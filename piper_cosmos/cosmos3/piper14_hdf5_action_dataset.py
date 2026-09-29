"""Cosmos3 action SFT dataset for raw Battery Piper14 HDF5 episodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from piper_cosmos.data.hdf5_reader import (
    default_instruction,
    find_hdf5_files,
    hdf5_key,
    image_keys,
    load_config,
)
from piper_cosmos.cosmos3.domain import PIPER14_DOMAIN_ID, register_piper14_domain


@dataclass(frozen=True)
class _SampleIndex:
    episode_path: Path
    t: int


class Piper14HDF5ActionDataset(Dataset):
    """Return raw samples compatible with Cosmos3 ``ActionTransformPipeline``."""

    def __init__(
        self,
        *,
        root: str,
        config_path: str,
        fps: float = 30.0,
        chunk_length: int = 32,
        mode: str = "policy",
        use_state: bool = True,
        viewpoint: str = "concat_view",
        stride: int = 1,
    ) -> None:
        if viewpoint != "concat_view":
            raise NotImplementedError("Piper14 Cosmos3 dataset currently supports only concat_view.")
        if mode != "policy":
            raise NotImplementedError("Piper14 Battery SFT is policy-only for the first integration.")
        if not use_state:
            raise NotImplementedError("Piper14 Battery SFT expects use_state=True.")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        register_piper14_domain()
        self.root = Path(root)
        self.config = load_config(config_path)
        self.fps = float(fps)
        self.chunk_length = int(chunk_length)
        self.mode = mode
        self.use_state = bool(use_state)
        self.viewpoint = viewpoint
        self.stride = int(stride)
        self.instruction = default_instruction(self.config) or "Assemble the mouse's battery."
        self.image_keys = image_keys(self.config)
        self.action_key = hdf5_key(self.config, "action_key", "/action")
        self.qpos_key = hdf5_key(self.config, "qpos_key", "/observations/qpos")
        self.episode_paths = find_hdf5_files(self.root)
        self.index, self._shuffle_blocks = self._build_index()

    @property
    def action_dim(self) -> int:
        return 14

    @property
    def domain_id(self) -> int:
        return PIPER14_DOMAIN_ID

    def __len__(self) -> int:
        return len(self.index)

    def get_shuffle_blocks(self) -> list[tuple[int, int]]:
        return list(self._shuffle_blocks)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import h5py  # type: ignore

        sample = self.index[int(idx)]
        t = sample.t
        with h5py.File(sample.episode_path, "r") as h5:
            qpos = torch.as_tensor(h5[self.qpos_key][t], dtype=torch.float32)
            action = torch.as_tensor(h5[self.action_key][t : t + self.chunk_length], dtype=torch.float32)
            video = self._load_concat_video(h5, t)
        action_with_state = torch.cat([qpos[None, :], action], dim=0)
        return {
            "ai_caption": self.instruction,
            "video": video,
            "action": action_with_state,
            "conditioning_fps": torch.tensor(int(round(self.fps)), dtype=torch.long),
            "mode": self.mode,
            "domain_id": torch.tensor(PIPER14_DOMAIN_ID, dtype=torch.long),
            "viewpoint": self.viewpoint,
            "additional_view_description": (
                "The fixed overhead camera is on top; the left wrist camera is bottom-left; "
                "the right wrist camera is bottom-right."
            ),
            "idle_frames": torch.tensor(0, dtype=torch.long),
        }

    def _build_index(self) -> tuple[list[_SampleIndex], list[tuple[int, int]]]:
        import h5py  # type: ignore

        index: list[_SampleIndex] = []
        blocks: list[tuple[int, int]] = []
        for episode_path in self.episode_paths:
            with h5py.File(episode_path, "r") as h5:
                length = min(int(h5[self.action_key].shape[0]), int(h5[self.qpos_key].shape[0]))
            start_offset = len(index)
            last_t = length - self.chunk_length - 1
            if last_t < 0:
                continue
            for t in range(0, last_t + 1, self.stride):
                index.append(_SampleIndex(episode_path=episode_path, t=t))
            blocks.append((start_offset, len(index) - start_offset))
        return index, [block for block in blocks if block[1] > 0]

    def _load_concat_video(self, h5: Any, t: int) -> torch.Tensor:
        frame_slice = slice(t, t + self.chunk_length + 1)
        high = self._read_video_tensor(h5[self.image_keys["cam_high"]][frame_slice])
        left = self._read_video_tensor(h5[self.image_keys["cam_left_wrist"]][frame_slice])
        right = self._read_video_tensor(h5[self.image_keys["cam_right_wrist"]][frame_slice])
        _, _, high_h, high_w = high.shape
        half_h, half_w = max(1, high_h // 2), max(1, high_w // 2)
        left = F.interpolate(left, size=(half_h, half_w), mode="bilinear", align_corners=False)
        right = F.interpolate(right, size=(half_h, half_w), mode="bilinear", align_corners=False)
        bottom = torch.cat([left, right], dim=-1)
        video_tchw = torch.cat([high, bottom], dim=-2)
        return (video_tchw * 255.0).clamp(0, 255).to(torch.uint8).permute(1, 0, 2, 3).contiguous()

    @staticmethod
    def _read_video_tensor(frames: Any) -> torch.Tensor:
        tensor = torch.as_tensor(frames, dtype=torch.float32)
        if tensor.ndim != 4 or tensor.shape[-1] != 3:
            raise ValueError(f"Expected THWC RGB frames, got {tuple(tensor.shape)}")
        return tensor.permute(0, 3, 1, 2).contiguous() / 255.0


def get_piper14_hdf5_sft_dataset(
    *,
    root: str,
    config_path: str,
    fps: float = 30.0,
    chunk_length: int = 32,
    mode: str = "policy",
    use_state: bool = True,
    action_normalization: str | None = None,
    viewpoint: str = "concat_view",
    resolution: str | int = "480",
    max_action_dim: int = 64,
    tokenizer_config: dict | None = None,
    cfg_dropout_rate: float = 0.1,
    format_prompt_as_json: bool = False,
    iterable_shuffle: bool = False,
    episode_shuffle_seed: int = 42,
    stride: int = 1,
    **_: Any,
) -> Dataset:
    """Build the Piper14 action SFT dataset and official Cosmos3 transform."""

    if action_normalization is not None:
        raise NotImplementedError("Piper14 first integration uses raw absolute joint targets.")
    try:
        from cosmos_framework.data.generator.action.datasets.action_sft_dataset import (
            ActionIterableShuffleDataset,
            ActionSFTDataset,
        )
        from cosmos_framework.data.generator.action.transforms import ActionTransformPipeline
    except ModuleNotFoundError:
        from cosmos_framework.data.vfm.action.datasets.action_sft_dataset import (
            ActionIterableShuffleDataset,
            ActionSFTDataset,
        )
        from cosmos_framework.data.vfm.action.transforms import ActionTransformPipeline

    raw = Piper14HDF5ActionDataset(
        root=root,
        config_path=config_path,
        fps=fps,
        chunk_length=chunk_length,
        mode=mode,
        use_state=use_state,
        viewpoint=viewpoint,
        stride=stride,
    )
    transform = ActionTransformPipeline(
        tokenizer_config=tokenizer_config,
        cfg_dropout_rate=cfg_dropout_rate,
        max_action_dim=max_action_dim,
        append_viewpoint_info=True,
        append_duration_fps_timestamps=True,
        append_resolution_info=True,
        append_idle_frames=False,
        format_prompt_as_json=format_prompt_as_json,
    )
    sft = ActionSFTDataset(raw, transform, resolution)
    if iterable_shuffle:
        return ActionIterableShuffleDataset(sft, seed=episode_shuffle_seed)
    return sft
