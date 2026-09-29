"""CPU hook-name contract for the measured offload runtime."""
from types import SimpleNamespace

import pytest

from cosmos_framework.inference.action_specprune_offload import NanoSpecPruneController


def make_controller():
    class Attention:
        _k112_attention_capture_callback = None

    controller = object.__new__(NanoSpecPruneController)
    controller.model = SimpleNamespace()
    controller.layers = [SimpleNamespace(self_attn=Attention())]
    return controller


def test_bridge_names_and_context_cleanup(monkeypatch):
    base = NanoSpecPruneController.__mro__[1]
    monkeypatch.setattr(base, "__enter__", lambda self: self)
    monkeypatch.setattr(base, "__exit__", lambda self, *args: None)
    controller = make_controller()
    with controller:
        assert controller.model._k112_sparse_controller is controller
        attention = controller.layers[0].self_attn
        callback = object()
        attention._nano_k112_attention_capture_callback = callback
        assert attention._k112_attention_capture_callback is callback
        attention._nano_k112_attention_capture_callback = None
        assert attention._k112_attention_capture_callback is None
    assert not hasattr(controller.model, "_k112_sparse_controller")


def test_bridge_rejects_occupied_runtime_hook():
    controller = make_controller()
    occupied = object()
    controller.model._k112_sparse_controller = occupied
    with pytest.raises(RuntimeError, match="occupied"):
        controller.__enter__()
    assert controller.model._k112_sparse_controller is occupied
