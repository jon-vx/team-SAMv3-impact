def __getattr__(name):
    if name in ("predict", "build_predictor"):
        from . import _inference_medsam3 as _m
        return getattr(_m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["predict", "build_predictor"]
