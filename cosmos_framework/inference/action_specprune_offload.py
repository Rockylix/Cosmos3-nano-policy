"""The same controller/callback-name bridge used by the measured HDF benchmark."""
from cosmos_framework.inference.action_specprune import NanoSpecPruneController as _Controller


class NanoSpecPruneController(_Controller):
    def __enter__(self):
        if hasattr(self.model, "_k112_sparse_controller"):
            raise RuntimeError("Offload runtime controller hook is occupied")
        for layer in self.layers:
            attention_type = type(layer.self_attn)
            if not hasattr(attention_type, "_nano_k112_attention_capture_callback"):
                attention_type._nano_k112_attention_capture_callback = property(
                    lambda self: self._k112_attention_capture_callback,
                    lambda self, value: setattr(self, "_k112_attention_capture_callback", value),
                )
        result = super().__enter__()
        self.model._k112_sparse_controller = self
        return result

    def __exit__(self, exc_type, exc, traceback):
        if getattr(self.model, "_k112_sparse_controller", None) is self:
            delattr(self.model, "_k112_sparse_controller")
        return super().__exit__(exc_type, exc, traceback)
