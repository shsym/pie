"""PyCrust FFI - Direct FFI-based RPC for Python-Rust communication."""

from .worker import (
    Worker,
    STATUS_OK,
    STATUS_METHOD_NOT_FOUND,
    STATUS_INVALID_PARAMS,
    STATUS_INTERNAL_ERROR,
)

__all__ = [
    "Worker",
    "STATUS_OK",
    "STATUS_METHOD_NOT_FOUND",
    "STATUS_INVALID_PARAMS",
    "STATUS_INTERNAL_ERROR",
]
