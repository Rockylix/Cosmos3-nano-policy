# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Send one Piper14 HDF5 image tuple to a Nano DROID policy server.

This is deliberately a structural inference smoke only.  Piper14 stores a
14-D dual-arm state while the released Nano DROID policy expects a 7-D arm
state plus gripper.  The tool therefore requires an explicit acknowledgement
and sends a zero DROID state; it never reports the returned action as a valid
Piper14 policy result.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

DEFAULT_KEYS = {
    "high": "observations/images/cam_high",
    "left": "observations/images/cam_left_wrist",
    "right": "observations/images/cam_right_wrist",
}


def _dataset(handle: h5py.File, key: str) -> h5py.Dataset:
    normalized = key.lstrip("/")
    if normalized not in handle:
        raise KeyError(f"HDF5 dataset {key!r} was not found")
    value = handle[normalized]
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"HDF5 path {key!r} is not a dataset")
    return value


def _decode_rgb(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 1 and array.dtype == np.uint8:
        array = np.asarray(Image.open(io.BytesIO(array.tobytes())).convert("RGB"))
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected RGB [H,W,3], got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    return np.asarray(
        Image.fromarray(image).resize((width, height), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def compose_concat_view(high: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    high = _resize(high, 480, 640)
    left = _resize(left, 240, 320)
    right = _resize(right, 240, 320)
    return np.ascontiguousarray(np.concatenate((high, np.concatenate((left, right), axis=1)), axis=0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--prompt", default="Pick up the banana and place it in the bowl")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--high-key", default=DEFAULT_KEYS["high"])
    parser.add_argument("--left-key", default=DEFAULT_KEYS["left"])
    parser.add_argument("--right-key", default=DEFAULT_KEYS["right"])
    parser.add_argument("--output", type=Path, default=Path("/tmp/nano_hdf5_smoke.json"))
    parser.add_argument(
        "--allow-droid-zero-state-smoke",
        action="store_true",
        help="Acknowledge that Piper14 14-D state is not valid DROID state and use zeros for transport/model smoke.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_droid_zero_state_smoke:
        raise SystemExit(
            "Refusing to reinterpret Piper14 14-D state as DROID 8-D state. "
            "Pass --allow-droid-zero-state-smoke for a structural image-path smoke only."
        )
    with h5py.File(args.hdf5.expanduser().absolute(), "r") as handle:
        high_ds = _dataset(handle, args.high_key)
        left_ds = _dataset(handle, args.left_key)
        right_ds = _dataset(handle, args.right_key)
        length = min(len(high_ds), len(left_ds), len(right_ds))
        index = int(args.frame_index)
        if not 0 <= index < length:
            raise IndexError(f"frame-index {index} is outside [0,{length})")
        image = compose_concat_view(
            _decode_rgb(high_ds[index]),
            _decode_rgb(left_ds[index]),
            _decode_rgb(right_ds[index]),
        )

    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(args.host, int(args.port))
    response = client.infer(
        {
            "observation/image": image,
            "observation/joint_position": np.zeros(7, dtype=np.float32),
            "observation/gripper_position": np.zeros(1, dtype=np.float32),
            "prompt": str(args.prompt),
        }
    )
    action = np.asarray(response["action"], dtype=np.float32)
    if action.ndim != 2 or not np.isfinite(action).all():
        raise RuntimeError(f"Nano smoke returned invalid action shape/data: {action.shape}")
    report = {
        "scope": "structural image-path smoke; not a Piper14 policy evaluation",
        "hdf5": str(args.hdf5.expanduser().absolute()),
        "frame_index": int(args.frame_index),
        "server": f"{args.host}:{int(args.port)}",
        "image_shape": list(image.shape),
        "action_shape": list(action.shape),
        "action_finite": True,
        "action_mean": float(action.mean()),
        "action_std": float(action.std()),
    }
    args.output.expanduser().absolute().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().absolute().write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
