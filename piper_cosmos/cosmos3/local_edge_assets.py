"""Strictly local asset bootstrap for Cosmos3-Edge Piper14 workflows.

Training and DCP inference require the full Cosmos3-Edge base snapshot.  A
self-contained HF inference export already carries the trained model,
processor, and vision tower, so its bootstrap deliberately validates only the
export, the Wan VAE, and the framework-owned reasoner config.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_WM_ROOT = REPO_ROOT.parents[1]
EDGE_REPOSITORY = "nvidia/Cosmos3-Edge"
WAN_VAE_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
ENV_EDGE_SNAPSHOT = "COSMOS3_EDGE_SNAPSHOT"
ENV_WAN_VAE_PATH = "COSMOS3_WAN_VAE_PATH"
ENV_REASONER_CONFIG = "COSMOS3_EDGE_REASONER_CONFIG"

DEFAULT_EDGE_SNAPSHOT = REPO_ROOT / "external" / "cosmos" / "checkpoints" / "Cosmos3-Edge"
DEFAULT_WAN_VAE_PATH = (
    PHYSICAL_WM_ROOT
    / "checkpoints"
    / "Wan-AI"
    / "Wan2.2-TI2V-5B"
    / "Wan2.2_VAE.pth"
)
DEFAULT_REASONER_CONFIG = (
    REPO_ROOT
    / "external"
    / "cosmos"
    / "packages"
    / "cosmos3"
    / "cosmos_framework"
    / "model"
    / "generator"
    / "reasoner"
    / "nemotron_3_dense_vl"
    / "configs"
    / "Nemotron-2B-Dense-VL.json"
)
REQUIRED_EDGE_FILES = (
    "config.json",
    "model_index.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
)
REQUIRED_EDGE_HF_INFERENCE_FILES = (
    "config.json",
    "checkpoint.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "vision_encoder/config.json",
    "vision_encoder/model.safetensors",
)

_LOCAL_REPOSITORY_PATHS: dict[str, Path] = {}
_ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD = None
_ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD = None


def _resolve_path(path: Path | None, env_var: str, default: Path) -> Path:
    if path is not None:
        return path
    raw = os.environ.get(env_var, "").strip()
    return Path(raw) if raw else default


def _set_common_offline_runtime_env(wan_vae_path: Path) -> None:
    cache_root = REPO_ROOT / "external" / "cosmos" / "cache"
    os.environ.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
    os.environ.setdefault("UV_TOOL_DIR", str(cache_root / "uv" / "tools"))
    os.environ.setdefault("XDG_DATA_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "external" / "cosmos" / "checkpoints" / "hf_home"))
    os.environ.setdefault("IMAGINAIRE_CACHE_DIR", str(cache_root / "imaginaire"))
    os.environ[ENV_WAN_VAE_PATH] = str(wan_vae_path)
    os.environ["WAN_VAE_PATH"] = str(wan_vae_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def set_offline_runtime_env(edge_snapshot: Path, wan_vae_path: Path) -> None:
    _set_common_offline_runtime_env(wan_vae_path)
    os.environ[ENV_EDGE_SNAPSHOT] = str(edge_snapshot)


def checkpoint_type(checkpoint: Path) -> str:
    """Return ``hf`` or ``dcp`` for a local Cosmos checkpoint."""

    checkpoint = checkpoint.expanduser().absolute()
    if (checkpoint / "config.json").is_file() and (
        (checkpoint / "model.safetensors.index.json").is_file()
        or any(checkpoint.glob("*.safetensors"))
    ):
        return "hf"
    model_dir = checkpoint / "model" if (checkpoint / "model").is_dir() else checkpoint
    if (model_dir / ".metadata").is_file() and any(model_dir.glob("*.distcp")):
        return "dcp"
    raise ValueError(f"Unrecognized checkpoint format: {checkpoint}")


def _missing_hf_shards(checkpoint: Path) -> list[str]:
    index_path = checkpoint / "model.safetensors.index.json"
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("weight_map must be a non-empty object")
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return [f"invalid HF shard index {index_path}: {exc}"]
    missing: list[str] = []
    for shard in weight_map.values():
        if not isinstance(shard, str):
            missing.append(f"invalid shard filename in {index_path}: {shard!r}")
        elif not (checkpoint / shard).is_file():
            missing.append(str(checkpoint / shard))
    return sorted(set(missing))


def resolve_edge_hf_inference_assets(
    *,
    checkpoint: Path,
    wan_vae_path: Path | None = None,
    reasoner_config: Path | None = None,
) -> dict[str, Path]:
    """Validate assets for a complete, local HF inference-only export."""

    checkpoint = checkpoint.expanduser().absolute()
    wan_vae_path = _resolve_path(wan_vae_path, ENV_WAN_VAE_PATH, DEFAULT_WAN_VAE_PATH)
    reasoner_config = _resolve_path(reasoner_config, ENV_REASONER_CONFIG, DEFAULT_REASONER_CONFIG)

    missing: list[str] = []
    if not checkpoint.is_dir():
        missing.append(f"missing HF inference export directory: {checkpoint}")
    else:
        missing.extend(
            str(checkpoint / relpath)
            for relpath in REQUIRED_EDGE_HF_INFERENCE_FILES
            if not (checkpoint / relpath).is_file()
        )
        if (checkpoint / "model.safetensors.index.json").is_file():
            missing.extend(_missing_hf_shards(checkpoint))
    if not wan_vae_path.is_file():
        missing.append(f"missing Wan VAE asset: {wan_vae_path}")
    if not reasoner_config.is_file():
        missing.append(f"missing framework reasoner config: {reasoner_config}")
    if missing:
        raise FileNotFoundError("HF inference asset validation failed: " + "; ".join(missing))

    return {
        "checkpoint": checkpoint,
        WAN_VAE_REPOSITORY: wan_vae_path,
        "framework_reasoner_config": reasoner_config,
    }


def edge_hf_inference_overrides(repository_paths: Mapping[str, Path]) -> tuple[str, ...]:
    """Build non-persistent path overrides for a portable HF export."""

    vae_path = json.dumps(str(repository_paths[WAN_VAE_REPOSITORY]))
    reasoner_config = json.dumps(str(repository_paths["framework_reasoner_config"]))
    return (
        f"model.config.tokenizer.vae_path={vae_path}",
        "model.config.tokenizer.bucket_name=",
        (
            "model.config.vlm_config.model_instance.config.base_config.json_file="
            f"{reasoner_config}"
        ),
    )


def bootstrap_edge_hf_inference_assets(
    *,
    checkpoint: Path,
    wan_vae_path: Path | None = None,
    reasoner_config: Path | None = None,
) -> dict[str, Path]:
    """Bootstrap a complete HF export without requiring the Edge base."""

    repository_paths = resolve_edge_hf_inference_assets(
        checkpoint=checkpoint,
        wan_vae_path=wan_vae_path,
        reasoner_config=reasoner_config,
    )
    _set_common_offline_runtime_env(repository_paths[WAN_VAE_REPOSITORY])
    os.environ[ENV_REASONER_CONFIG] = str(repository_paths["framework_reasoner_config"])
    return repository_paths


def resolve_local_edge_assets(
    *,
    edge_snapshot: Path | None = None,
    wan_vae_path: Path | None = None,
) -> dict[str, Path]:
    edge_snapshot = _resolve_path(edge_snapshot, ENV_EDGE_SNAPSHOT, DEFAULT_EDGE_SNAPSHOT)
    wan_vae_path = _resolve_path(wan_vae_path, ENV_WAN_VAE_PATH, DEFAULT_WAN_VAE_PATH)

    missing: list[str] = []
    if not edge_snapshot.is_dir():
        missing.append(f"missing Cosmos3-Edge snapshot directory: {edge_snapshot}")
    else:
        missing.extend(str(edge_snapshot / relpath) for relpath in REQUIRED_EDGE_FILES if not (edge_snapshot / relpath).is_file())
        if not list(edge_snapshot.glob("*.safetensors")) and not list((edge_snapshot / "transformer").glob("*.safetensors")):
            missing.append(f"missing Cosmos3-Edge safetensors shards under: {edge_snapshot}")
    if not wan_vae_path.is_file():
        missing.append(f"missing Wan VAE asset: {wan_vae_path}")
    if missing:
        raise FileNotFoundError("offline asset validation failed: " + "; ".join(missing))

    return {EDGE_REPOSITORY: edge_snapshot, WAN_VAE_REPOSITORY: wan_vae_path}


def seed_registry_paths(
    checkpoints: Iterable[object],
    repository_paths: Mapping[str, Path],
) -> dict[str, int]:
    seeded = {repository: 0 for repository in repository_paths}
    for checkpoint in checkpoints:
        hf = getattr(checkpoint, "hf", None)
        repository = getattr(hf, "repository", None)
        if repository not in repository_paths:
            continue
        hf._path = str(repository_paths[repository])
        seeded[repository] += 1
    return seeded


def _local_checkpoint_dir_hf_download(checkpoint: object) -> str:
    repository = getattr(checkpoint, "repository", "")
    if repository in _LOCAL_REPOSITORY_PATHS:
        path = _LOCAL_REPOSITORY_PATHS[repository]
        subdirectory = getattr(checkpoint, "subdirectory", "")
        return str(path / subdirectory) if subdirectory else str(path)
    if _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD is None:
        raise RuntimeError("original CheckpointDirHf._download is unavailable")
    return _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD(checkpoint)


def _local_checkpoint_file_hf_download(checkpoint: object) -> str:
    repository = getattr(checkpoint, "repository", "")
    if repository == WAN_VAE_REPOSITORY:
        return str(_LOCAL_REPOSITORY_PATHS[WAN_VAE_REPOSITORY])
    if _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD is None:
        raise RuntimeError("original CheckpointFileHf._download is unavailable")
    return _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD(checkpoint)


def patch_hf_downloads(repository_paths: Mapping[str, Path]) -> None:
    from cosmos_framework.utils.checkpoint_db import CheckpointDirHf, CheckpointFileHf

    global _LOCAL_REPOSITORY_PATHS
    global _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD
    global _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD

    _LOCAL_REPOSITORY_PATHS = dict(repository_paths)
    if _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD is None:
        _ORIGINAL_CHECKPOINT_DIR_HF_DOWNLOAD = CheckpointDirHf._download
        CheckpointDirHf._download = _local_checkpoint_dir_hf_download
    if _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD is None:
        _ORIGINAL_CHECKPOINT_FILE_HF_DOWNLOAD = CheckpointFileHf._download
        CheckpointFileHf._download = _local_checkpoint_file_hf_download


def bootstrap_local_edge_assets(
    *,
    edge_snapshot: Path | None = None,
    wan_vae_path: Path | None = None,
) -> dict[str, Path]:
    repository_paths = resolve_local_edge_assets(edge_snapshot=edge_snapshot, wan_vae_path=wan_vae_path)
    set_offline_runtime_env(repository_paths[EDGE_REPOSITORY], repository_paths[WAN_VAE_REPOSITORY])

    from cosmos_framework.inference.common.checkpoints import register_checkpoints
    from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

    register_checkpoints()
    seeded = seed_registry_paths(_CHECKPOINTS.values(), repository_paths)
    if seeded[EDGE_REPOSITORY] == 0:
        raise RuntimeError(f"No checkpoint registry entries were seeded for {EDGE_REPOSITORY}.")
    if seeded[WAN_VAE_REPOSITORY] == 0:
        raise RuntimeError(f"No checkpoint registry entries were seeded for {WAN_VAE_REPOSITORY}.")
    patch_hf_downloads(repository_paths)
    return repository_paths
