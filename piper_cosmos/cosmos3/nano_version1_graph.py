"""Compile and whole-stack CUDA Graph backend for offloaded Nano Version1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from piper_cosmos.cosmos3.k112_sparse import _get_gen_seq
from piper_cosmos.cosmos3.nano_version1_sparse import NanoVersion1Controller

_COMPILED_LAYERS_ATTR = "_piper_nano_version1_compiled_layers"
_GRAPH_RUNNERS_ATTR = "_piper_nano_version1_graph_runners"


def _clone_pack(torch: Any, pack: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in pack.items()}


def _copy_pack_(torch: Any, target: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    if target.keys() != source.keys():
        raise RuntimeError("SequencePack keys changed after graph capture")
    for key, source_value in source.items():
        target_value = target[key]
        if torch.is_tensor(source_value):
            if not torch.is_tensor(target_value) or target_value.shape != source_value.shape:
                raise RuntimeError(f"SequencePack signature changed for {key}")
            target_value.copy_(source_value)
        elif target_value != source_value:
            raise RuntimeError(f"SequencePack metadata changed for {key}")


def _clone_memory_value(value: Any, *, for_cuda_graphs: bool) -> Any:
    if value.und_k is None or value.und_v is None:
        raise RuntimeError("Version1 compile/graph requires initialized UND K/V")
    return type(value)(
        und_k=value.und_k.clone(),
        und_v=value.und_v.clone(),
        frame_idx=int(value.frame_idx),
        for_cuda_graphs=for_cuda_graphs,
    )


def _copy_memory_(target: list[Any], source: list[Any]) -> None:
    for layer, (dst, src) in enumerate(zip(target, source, strict=True)):
        if src.und_k is None or src.und_v is None or dst.und_k is None or dst.und_v is None:
            raise RuntimeError(f"UND K/V missing at layer {layer}")
        if dst.und_k.shape != src.und_k.shape or dst.und_v.shape != src.und_v.shape:
            raise RuntimeError(f"UND K/V signature changed at layer {layer}")
        dst.und_k.copy_(src.und_k)
        dst.und_v.copy_(src.und_v)


@dataclass
class WholeStackGraphRunner:
    torch: Any
    static_hidden: dict[str, Any]
    static_rope: tuple[dict[str, Any], dict[str, Any]]
    static_memory: list[Any]
    graph: Any
    static_output: dict[str, Any]

    @classmethod
    def capture(
        cls,
        *,
        torch: Any,
        layers: tuple[Any, ...],
        hidden_states: Mapping[str, Any],
        position_embeddings: tuple[Mapping[str, Any], Mapping[str, Any]],
        attention_mask: Any,
        memory_values: list[Any],
    ) -> "WholeStackGraphRunner":
        static_hidden = _clone_pack(torch, hidden_states)
        static_rope = (_clone_pack(torch, position_embeddings[0]), _clone_pack(torch, position_embeddings[1]))
        static_memory = [_clone_memory_value(value, for_cuda_graphs=True) for value in memory_values]

        def execute() -> dict[str, Any]:
            output: Any = static_hidden
            for layer, memory_value in zip(layers, static_memory, strict=True):
                output, _lbl, _kv = layer(
                    output,
                    attention_mask,
                    static_rope,
                    natten_metadata=None,
                    memory_value=memory_value,
                    gen_only=True,
                )
            return output

        warm_stream = torch.cuda.Stream()
        warm_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm_stream):
            static_output = execute()
            static_output = execute()
        torch.cuda.current_stream().wait_stream(warm_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_output = execute()
        torch.cuda.synchronize()
        return cls(torch, static_hidden, static_rope, static_memory, graph, static_output)

    def replay(
        self,
        *,
        hidden_states: Mapping[str, Any],
        position_embeddings: tuple[Mapping[str, Any], Mapping[str, Any]],
        memory_values: list[Any],
    ) -> dict[str, Any]:
        _copy_pack_(self.torch, self.static_hidden, hidden_states)
        _copy_pack_(self.torch, self.static_rope[0], position_embeddings[0])
        _copy_pack_(self.torch, self.static_rope[1], position_embeddings[1])
        _copy_memory_(self.static_memory, memory_values)
        self.graph.replay()
        return self.static_output


class NanoVersion1CompiledController(NanoVersion1Controller):
    """Keep step 0 eager; use compiled GEN layers and optional stack graph later."""

    def __init__(
        self,
        *,
        torch: Any,
        net: Any,
        guidance: float,
        num_steps: int,
        output_dir: Any | None = None,
        use_cuda_graphs: bool,
    ) -> None:
        super().__init__(torch=torch, net=net, guidance=guidance, num_steps=num_steps, output_dir=output_dir)
        self.use_cuda_graphs = bool(use_cuda_graphs)
        self._memory_state: Any | None = None
        self._stack_output: Any | None = None
        self._step0_eager_block_calls = 0
        self._whole_stack_graph_captures = 0
        self._whole_stack_graph_replays = 0
        self._whole_stack_compile_calls = 0
        self._outer_identity_calls = 0

        compiled = getattr(self.model, _COMPILED_LAYERS_ATTR, None)
        if compiled is None:
            compiled = tuple(
                torch.compile(getattr(layer, "_orig_mod", layer), fullgraph=True, dynamic=False)
                for layer in self.layers
            )
            setattr(self.model, _COMPILED_LAYERS_ATTR, compiled)
        self._compiled_layers = compiled
        runners = getattr(self.model, _GRAPH_RUNNERS_ATTR, None)
        if runners is None:
            runners = {}
            setattr(self.model, _GRAPH_RUNNERS_ATTR, runners)
        self._graph_runners = runners

    def _network_pre_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        self._memory_state = kwargs.get("memory", args[1] if len(args) > 1 else None)
        super()._network_pre_hook(module, args, kwargs)

    def _network_post_hook(self, module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any], output: Any) -> None:
        try:
            super()._network_post_hook(module, args, kwargs, output)
        finally:
            self._memory_state = None

    def run_layer(
        self,
        *,
        block: int,
        decoder_layer: Any,
        hidden_states: Any,
        attention_mask: Any,
        memory_value: Any,
        gen_only: bool,
    ) -> tuple[Any, dict[str, Any], Any]:
        if self._current is None:
            raise RuntimeError("Version1 compiled controller has no active stack")
        step = int(self._current["step"])
        if step == 0:
            self._step0_eager_block_calls += 1
            return super().run_layer(
                block=block,
                decoder_layer=getattr(decoder_layer, "_orig_mod", decoder_layer),
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                memory_value=memory_value,
                gen_only=gen_only,
            )

        if block == 0:
            selected = self._prepare_selected_positions(_get_gen_seq(hidden_states).device)
            hidden_states, self._position_embeddings = self._slice_pack_and_rope(hidden_states, selected)
            self._active_positions = selected
            self._pack_count += 1
            if not gen_only or self._memory_state is None or not self._memory_state.is_gen_only():
                raise RuntimeError("Compiled sparse stack requires a completed reasoner prefill")
            memory_values = [self._memory_state.read_for_layer(index) for index in range(len(self.layers))]
            branch = str(self._current["branch"])
            if self.use_cuda_graphs:
                runner = self._graph_runners.get(branch)
                if runner is None:
                    runner = WholeStackGraphRunner.capture(
                        torch=self.torch,
                        layers=self._compiled_layers,
                        hidden_states=hidden_states,
                        position_embeddings=self._position_embeddings,
                        attention_mask=attention_mask,
                        memory_values=memory_values,
                    )
                    self._graph_runners[branch] = runner
                    self._whole_stack_graph_captures += 1
                self._stack_output = runner.replay(
                    hidden_states=hidden_states,
                    position_embeddings=self._position_embeddings,
                    memory_values=memory_values,
                )
                self._whole_stack_graph_replays += 1
            else:
                output: Any = hidden_states
                for layer, layer_memory in zip(self._compiled_layers, memory_values, strict=True):
                    output, _lbl, _kv = layer(
                        output,
                        attention_mask,
                        self._position_embeddings,
                        natten_metadata=None,
                        memory_value=layer_memory,
                        gen_only=True,
                    )
                self._stack_output = output
                self._whole_stack_compile_calls += 1
            self._block_calls += len(self.layers)
            return self._stack_output, {}, None
        if self._stack_output is None:
            raise RuntimeError("Outer loop reached B1 before whole-stack execution")
        self._outer_identity_calls += 1
        return self._stack_output, {}, None

    def end_stack(self, hidden_states: Any) -> Any:
        try:
            return super().end_stack(hidden_states)
        finally:
            self._stack_output = None

    def finish(self) -> dict[str, Any]:
        summary = super().finish()
        summary.update(
            {
                "execution_backend": "compiled_gen_layers_with_global_graph" if self.use_cuda_graphs else "compiled_gen_layers",
                "graph_boundary": "one_complete_sparse_36_block_stack_per_cfg_branch",
                "graph_start_step": 1,
                "step0_eager_block_calls": self._step0_eager_block_calls,
                "whole_stack_graph_captures": self._whole_stack_graph_captures,
                "whole_stack_graph_replays": self._whole_stack_graph_replays,
                "whole_stack_compile_calls": self._whole_stack_compile_calls,
                "outer_identity_calls": self._outer_identity_calls,
            }
        )
        return summary
