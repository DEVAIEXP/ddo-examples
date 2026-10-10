"""Resolve processor types from the selected LTX implementation, not globally."""
from importlib import import_module


def resolve_ltx_attention_types(model):
    """Support the protected fork runner and trusted modular local classes."""
    namespace = import_module(type(model).__module__)
    names = ("LTX2Attention", "LTX2AudioVideoAttnProcessor", "LTX2PerturbedAttnProcessor")
    try:
        types = tuple(getattr(namespace, name) for name in names)
    except AttributeError as error:
        raise ValueError("Selected transformer does not expose standard LTX processor types") from error
    if not all(isinstance(cls, type) for cls in types):
        raise ValueError("Selected transformer processor bindings must be classes")
    return types
