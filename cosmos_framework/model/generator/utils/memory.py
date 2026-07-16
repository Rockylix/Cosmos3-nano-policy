# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Abstract interfaces for persistent memory in the MoT transformer stack.

``MemoryState`` is a *mutable* Python object that lives **outside** the
``torch.compile`` boundary.  It is responsible for reading cached tensors
(``read_for_layer``) and writing new tensors back (``write_for_layer``).

``MemoryValue`` is a *read-only* tensor container that is safe to pass
**into** a compiled decoder layer.  Concrete implementations are plain
dataclasses whose fields are tensors (or None).  No methods on
``MemoryValue`` should mutate state.

``KVToStore`` is a type alias for the 4-tuple of tensors
``(gen_k, gen_v, und_k, und_v)`` returned by each compiled layer so
the caller can write them back into the cache outside the compile boundary.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

# (gen_k, gen_v, und_k, und_v) returned by each compiled layer for the caller
# to write back into the cache outside the torch.compile boundary.
KVToStore = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass
class MemoryValue(ABC):
    """Read-only tensor container safe to pass into ``torch.compile``.

    Concrete subclasses (e.g. ``ARMemoryValue``, ``KVTrainMemoryValue``)
    are plain dataclasses of tensors.  No methods on this class should
    mutate state or perform non-trivial computation.
    """

    @property
    def supports_context_parallel_attention(self) -> bool:
        """Whether this memory value is compatible with context-parallel attention.

        Overridden by ``KVTrainMemoryValue`` to return ``False``.  Used by
        ``ContextParallelDispatch`` to reject an unsupported combination
        without importing the concrete subclass.
        """
        return True


@dataclass
class DiffusionKVCacheMemoryValue(MemoryValue):
    """Read-only und K/V snapshot for diffusion gen-only denoising."""

    und_k: torch.Tensor | None = None
    und_v: torch.Tensor | None = None
    frame_idx: int = 0
    for_cuda_graphs: bool = False


class MemoryState(ABC):
    """Mutable persistent memory that lives outside ``torch.compile``.

    The outer loop in ``_impl_forward`` calls ``read_for_layer`` before
    each decoder layer and ``write_for_layer`` after.  The ``MemoryState``
    object itself is **never** passed into a compiled region.
    """

    @abstractmethod
    def init(self, hidden_states: dict, device: torch.device) -> None:
        """Initialization per training step.

        Called once before any transformer layers are processed.

        Args:
            hidden_states: The packed sequence (``SequencePack``).
            device: Target device for any new tensors.
        """

    @abstractmethod
    def read_for_layer(self, layer_idx: int) -> MemoryValue:
        """Produce a read-only tensor snapshot for *layer_idx*.

        Used to retrieve KV values from the cache.
        The returned ``MemoryValue`` is passed into the compiled decoder
        layer as ``memory_value``.
        """

    @abstractmethod
    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        """Store the K/V tensors produced by *layer_idx* back into the cache.

        Called outside the ``torch.compile`` boundary.
        """

    @abstractmethod
    def is_gen_only(self) -> bool:
        """Return ``True`` when only the generation pathway should run.

        When ``True``, the decoder layer assumes that the text caption has
        already been processed and cached in the MemoryState object.
        Used for autoregressive frame-by-frame generation of video.
        """

    def requires_natten_metadata(self) -> bool:
        """Whether the packed-sequence builder should create NATTEN metadata.

        Memory paths whose attention implementation handles temporal
        visibility itself return ``False``.
        """
        return True


class DiffusionKVCacheMemoryState(MemoryState):
    """Cache understanding-tower K/V once, then reuse it for gen-only denoising.

    A separate und-only prefill records the per-layer understanding K/V
    tensors. Every denoise step, including step 0, then returns cached K/V and
    marks the decoder as gen-only.
    """

    def __init__(self, num_layers: int, *, detach: bool = True) -> None:
        self.num_layers = num_layers
        self.detach = detach
        self._und_k: list[torch.Tensor | None] = [None] * num_layers
        self._und_v: list[torch.Tensor | None] = [None] * num_layers
        self._initialized = False
        self._device: torch.device | None = None
        self._debug_und_input: torch.Tensor | None = None

    def init(self, hidden_states: dict, device: torch.device) -> None:
        debug_validate_und = os.environ.get("COSMOS3_REASONER_KVCACHE_DEBUG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        reasoner_offloaded = os.environ.get("COSMOS3_REASONER_OFFLOAD", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if debug_validate_und and not (reasoner_offloaded and self.is_gen_only()):
            und_len = int(hidden_states["_num_causal_tokens"])
            und_input = hidden_states["causal_seq"][:und_len].detach()
            if self._debug_und_input is None:
                self._debug_und_input = und_input.clone()
            else:
                max_abs = (und_input - self._debug_und_input).abs().max().item()
                if max_abs != 0.0:
                    raise RuntimeError(
                        "Diffusion KV-cache invariant failed: decoder understanding input changed "
                        f"between denoise calls (max_abs={max_abs:.8g})."
                    )
        self._initialized = True
        self._device = device

    def read_for_layer(self, layer_idx: int) -> DiffusionKVCacheMemoryValue:
        self._check_layer_idx(layer_idx)
        return DiffusionKVCacheMemoryValue(
            und_k=self._und_k[layer_idx],
            und_v=self._und_v[layer_idx],
            frame_idx=1 if self.is_gen_only() else 0,
        )

    def write_for_layer(self, layer_idx: int, kv_to_store: KVToStore) -> None:
        self._check_layer_idx(layer_idx)
        _gen_k, _gen_v, und_k, und_v = kv_to_store
        if self.detach:
            und_k = und_k.detach()
            und_v = und_v.detach()
        self._und_k[layer_idx] = und_k
        self._und_v[layer_idx] = und_v

    def is_gen_only(self) -> bool:
        return self._initialized and all(k is not None and v is not None for k, v in zip(self._und_k, self._und_v))

    def reset(self) -> None:
        self._und_k = [None] * self.num_layers
        self._und_v = [None] * self.num_layers
        self._initialized = False
        self._device = None
        self._debug_und_input = None

    def _check_layer_idx(self, layer_idx: int) -> None:
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range for {self.num_layers} layers")
