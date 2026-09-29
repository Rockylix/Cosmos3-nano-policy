"""Local-first Hugging Face asset bootstrap for Cosmos3 Piper workflows."""

from __future__ import annotations

import os
from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PHYSICAL_WM_ROOT = REPO_ROOT.parents[1]
QWEN_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
WAN_VAE_REPOSITORY = "Wan-AI/Wan2.2-TI2V-5B"
ENV_QWEN_SNAPSHOT = "COSMOS3_QWEN_SNAPSHOT"
ENV_WAN_VAE_PATH = "COSMOS3_WAN_VAE_PATH"
_ORIGINAL_DOWNLOAD_TOKENIZER_FILES = None

DEFAULT_QWEN_SNAPSHOT = (
    PHYSICAL_WM_ROOT
    / "checkpoints"
    / "hf_home"
    / "hub"
    / "models--Qwen--Qwen3-VL-8B-Instruct"
    / "snapshots"
    / "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
DEFAULT_WAN_VAE_PATH = (
    PHYSICAL_WM_ROOT
    / "checkpoints"
    / "Wan-AI"
    / "Wan2.2-TI2V-5B"
    / "Wan2.2_VAE.pth"
)
REQUIRED_QWEN_FILES = ("vocab.json", "merges.txt", "tokenizer_config.json")


def set_workspace_cache_env() -> None:
    cache_root = REPO_ROOT / "external" / "cosmos" / "cache"
    os.environ.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
    os.environ.setdefault("UV_TOOL_DIR", str(cache_root / "uv" / "tools"))
    os.environ.setdefault("XDG_DATA_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    os.environ.setdefault("HF_HOME", str(REPO_ROOT / "external" / "cosmos" / "checkpoints" / "hf_home"))
    os.environ.setdefault("IMAGINAIRE_CACHE_DIR", str(cache_root / "imaginaire"))


def set_local_first_hf_env(qwen_snapshot: Path, wan_vae_path: Path) -> None:
    os.environ[ENV_QWEN_SNAPSHOT] = str(qwen_snapshot)
    os.environ[ENV_WAN_VAE_PATH] = str(wan_vae_path)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _resolve_path(path: Path | None, env_var: str, default: Path) -> Path:
    if path is not None:
        return path
    raw = os.environ.get(env_var, "").strip()
    if raw:
        return Path(raw)
    return default


def resolve_local_hf_assets(
    *,
    qwen_snapshot: Path | None = None,
    wan_vae_path: Path | None = None,
) -> dict[str, Path]:
    qwen_snapshot = _resolve_path(qwen_snapshot, ENV_QWEN_SNAPSHOT, DEFAULT_QWEN_SNAPSHOT)
    wan_vae_path = _resolve_path(wan_vae_path, ENV_WAN_VAE_PATH, DEFAULT_WAN_VAE_PATH)

    missing_messages: list[str] = []
    if not qwen_snapshot.is_dir():
        missing_messages.append(f"missing Qwen snapshot directory: {qwen_snapshot}")
    else:
        missing_qwen_files = [str(qwen_snapshot / name) for name in REQUIRED_QWEN_FILES if not (qwen_snapshot / name).is_file()]
        if missing_qwen_files:
            missing_messages.append("missing Qwen tokenizer assets: " + ", ".join(missing_qwen_files))
    if not wan_vae_path.is_file():
        missing_messages.append(f"missing Wan VAE asset: {wan_vae_path}")
    if missing_messages:
        raise FileNotFoundError("; ".join(missing_messages))

    return {
        QWEN_REPOSITORY: qwen_snapshot,
        WAN_VAE_REPOSITORY: wan_vae_path,
    }


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


def seed_local_hf_registry(repository_paths: Mapping[str, Path]) -> dict[str, int]:
    from cosmos_framework.inference.common.checkpoints import register_checkpoints
    from cosmos_framework.utils.checkpoint_db import _CHECKPOINTS

    register_checkpoints()
    return seed_registry_paths(_CHECKPOINTS.values(), repository_paths)


def local_download_tokenizer_files(model_name: str, config_variant: str) -> str:
    if model_name == QWEN_REPOSITORY:
        return str(resolve_local_hf_assets()[QWEN_REPOSITORY])
    if _ORIGINAL_DOWNLOAD_TOKENIZER_FILES is None:
        raise RuntimeError("Original download_tokenizer_files is not initialized.")
    return _ORIGINAL_DOWNLOAD_TOKENIZER_FILES(model_name, config_variant)


def patch_qwen_tokenizer_loader() -> None:
    from cosmos_framework.configs.base import defaults
    try:
        from cosmos_framework.configs.base.defaults import vlm
    except ImportError:
        from cosmos_framework.configs.base.defaults import reasoner as vlm

    global _ORIGINAL_DOWNLOAD_TOKENIZER_FILES

    if getattr(vlm.download_tokenizer_files, "_cosmos3_local_hf_patched", False):
        return

    _ORIGINAL_DOWNLOAD_TOKENIZER_FILES = vlm.download_tokenizer_files
    local_download_tokenizer_files._cosmos3_local_hf_patched = True  # type: ignore[attr-defined]
    if hasattr(defaults, "vlm"):
        defaults.vlm.download_tokenizer_files = local_download_tokenizer_files
    if hasattr(defaults, "reasoner"):
        defaults.reasoner.download_tokenizer_files = local_download_tokenizer_files
    vlm.download_tokenizer_files = local_download_tokenizer_files


def bootstrap_local_hf_assets(
    *,
    qwen_snapshot: Path | None = None,
    wan_vae_path: Path | None = None,
) -> dict[str, Path]:
    set_workspace_cache_env()
    repository_paths = resolve_local_hf_assets(qwen_snapshot=qwen_snapshot, wan_vae_path=wan_vae_path)
    set_local_first_hf_env(
        repository_paths[QWEN_REPOSITORY],
        repository_paths[WAN_VAE_REPOSITORY],
    )
    seeded = seed_local_hf_registry(repository_paths)
    if seeded[QWEN_REPOSITORY] == 0:
        raise RuntimeError(f"No checkpoint registry entries were seeded for {QWEN_REPOSITORY}.")
    if seeded[WAN_VAE_REPOSITORY] == 0:
        raise RuntimeError(f"No checkpoint registry entries were seeded for {WAN_VAE_REPOSITORY}.")
    patch_qwen_tokenizer_loader()
    return repository_paths
