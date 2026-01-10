"""Pie - Core logic for the Pie Inference Engine."""

__version__ = "0.1.0"

# pie_rs is the compiled Rust extension module (built by maturin)
try:
    import pie_rs

    __all__ = ["pie_rs"]
except ImportError:
    # pie_rs not built yet - this is fine for pure Python usage
    __all__ = []
