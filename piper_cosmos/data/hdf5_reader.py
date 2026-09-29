#!/usr/bin/env python3
"""Read raw dual-Piper HDF5 episodes without modifying source files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_IMAGE_KEYS = {
    "cam_high": "/observations/images/cam_high",
    "cam_left_wrist": "/observations/images/cam_left_wrist",
    "cam_right_wrist": "/observations/images/cam_right_wrist",
}
DEFAULT_ACTION_KEY = "/action"
DEFAULT_QPOS_KEY = "/observations/qpos"
DEFAULT_QVEL_KEY = "/observations/qvel"


def load_config(path: Path | str | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit("PyYAML is required to read dataset config.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {config_path}")
    return data


def hdf5_key(config: dict[str, Any], name: str, default: str) -> str:
    hdf5 = config.get("hdf5", {})
    if isinstance(hdf5, dict):
        value = hdf5.get(name, default)
        if isinstance(value, str):
            return value
    return default


def image_keys(config: dict[str, Any]) -> dict[str, str]:
    hdf5 = config.get("hdf5", {})
    if isinstance(hdf5, dict):
        value = hdf5.get("image_keys", {})
        if isinstance(value, dict):
            parsed = {str(name): str(key) for name, key in value.items()}
            if parsed:
                return parsed
    return dict(DEFAULT_IMAGE_KEYS)


def default_instruction(config: dict[str, Any]) -> str:
    language = config.get("language", {})
    if isinstance(language, dict):
        value = language.get("default_instruction", "")
        if isinstance(value, str):
            return value
    return ""


def find_hdf5_files(data_root: Path | str) -> list[Path]:
    root = Path(data_root)
    if root.is_file():
        return [root]
    files = sorted(root.rglob("*.hdf5")) + sorted(root.rglob("*.h5"))
    return sorted(set(files))


@dataclass(frozen=True)
class EpisodeSummary:
    path: str
    length: int
    action_shape: tuple[int, ...]
    qpos_shape: tuple[int, ...]
    qvel_shape: tuple[int, ...]
    image_shapes: dict[str, tuple[int, ...]]


class HDF5EpisodeReader:
    """Small HDF5 episode reader scoped to numeric state/action and RGB frames."""

    def __init__(
        self,
        episode_path: Path | str,
        config: dict[str, Any] | None = None,
        image_size: int = 224,
    ) -> None:
        self.path = Path(episode_path)
        self.config = config or {}
        self.image_size = int(image_size)
        self.image_keys = image_keys(self.config)
        self.action_key = hdf5_key(self.config, "action_key", DEFAULT_ACTION_KEY)
        self.qpos_key = hdf5_key(self.config, "qpos_key", DEFAULT_QPOS_KEY)
        self.qvel_key = hdf5_key(self.config, "qvel_key", DEFAULT_QVEL_KEY)

    def summary(self) -> EpisodeSummary:
        import h5py  # type: ignore

        with h5py.File(self.path, "r") as h5:
            action = h5[self.action_key]
            qpos = h5[self.qpos_key]
            qvel = h5[self.qvel_key]
            image_shapes = {
                name: tuple(h5[key].shape)
                for name, key in self.image_keys.items()
                if key in h5
            }
            length = min(int(action.shape[0]), int(qpos.shape[0]), int(qvel.shape[0]))
        return EpisodeSummary(
            path=str(self.path),
            length=length,
            action_shape=tuple(action.shape),
            qpos_shape=tuple(qpos.shape),
            qvel_shape=tuple(qvel.shape),
            image_shapes=image_shapes,
        )

    def __len__(self) -> int:
        return self.summary().length

    def read_qpos(self, t: int) -> np.ndarray:
        return self._read_row(self.qpos_key, t)

    def read_qvel(self, t: int) -> np.ndarray:
        return self._read_row(self.qvel_key, t)

    def read_action_chunk(self, t: int, horizon: int) -> np.ndarray:
        import h5py  # type: ignore

        start = int(t)
        end = start + int(horizon)
        with h5py.File(self.path, "r") as h5:
            dataset = h5[self.action_key]
            if start < 0 or end > dataset.shape[0]:
                raise IndexError(
                    f"Action chunk [{start}:{end}] is outside episode length {dataset.shape[0]}"
                )
            return np.asarray(dataset[start:end], dtype=np.float32)

    def read_image_history(self, t: int, history_frames: int) -> dict[str, np.ndarray]:
        import h5py  # type: ignore

        end = int(t)
        start = end - int(history_frames) + 1
        if start < 0:
            raise IndexError(f"Image history start {start} is before frame 0")

        histories: dict[str, np.ndarray] = {}
        with h5py.File(self.path, "r") as h5:
            for name, key in self.image_keys.items():
                dataset = h5[key]
                if end >= dataset.shape[0]:
                    raise IndexError(
                        f"Image history end {end} is outside {name} length {dataset.shape[0]}"
                    )
                frames = [self._frame_to_chw(dataset[idx]) for idx in range(start, end + 1)]
                histories[name] = np.stack(frames, axis=0).astype(np.float32, copy=False)
        return histories

    def _read_row(self, key: str, t: int) -> np.ndarray:
        import h5py  # type: ignore

        idx = int(t)
        with h5py.File(self.path, "r") as h5:
            dataset = h5[key]
            if idx < 0 or idx >= dataset.shape[0]:
                raise IndexError(f"Index {idx} is outside dataset length {dataset.shape[0]}")
            return np.asarray(dataset[idx], dtype=np.float32)

    def _frame_to_chw(self, frame: np.ndarray) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"Expected RGB THWC frame with 3 channels, got {array.shape}")
        image = Image.fromarray(array.astype(np.uint8), mode="RGB")
        if self.image_size > 0 and image.size != (self.image_size, self.image_size):
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        normalized = np.asarray(image, dtype=np.float32) / 255.0
        return np.transpose(normalized, (2, 0, 1))
