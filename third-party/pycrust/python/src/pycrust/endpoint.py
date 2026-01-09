"""RPC Endpoint class for registering and handling methods."""

from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING

# Lazy import Pydantic only when needed
if TYPE_CHECKING:
    from pydantic import BaseModel

# Track if pydantic is available
_pydantic_available: Optional[bool] = None


def _check_pydantic() -> bool:
    """Check if pydantic is available, caching the result."""
    global _pydantic_available
    if _pydantic_available is None:
        try:
            import pydantic  # noqa: F401

            _pydantic_available = True
        except ImportError:
            _pydantic_available = False
    return _pydantic_available


def _validate_with_pydantic(schema: Type["BaseModel"], args: dict) -> dict:
    """Validate args using Pydantic model. Only called when schema is provided."""
    from pydantic import ValidationError as PydanticValidationError

    try:
        validated = schema(**args)
        return validated.model_dump()
    except PydanticValidationError as e:
        raise ValueError(f"Validation error: {e}") from e


class RpcEndpoint:
    """
    RPC Endpoint for handling method calls from Rust clients.

    The endpoint manages method registration and dispatching, with optional
    Pydantic validation for request arguments.

    Example:
        rpc = RpcEndpoint("my_service")

        def add(a: int, b: int) -> int:
            return a + b

        rpc.register("add", add)
        rpc.listen()
    """

    def __init__(self, service_name: str) -> None:
        """
        Initialize an RPC endpoint.

        Args:
            service_name: Unique name for this service (used for IPC topics)
        """
        self.service_name = service_name
        self._methods: Dict[str, Callable[..., Any]] = {}
        self._schemas: Dict[str, Optional[Type["BaseModel"]]] = {}

    def register(
        self,
        name: Optional[str] = None,
        func: Optional[Callable[..., Any]] = None,
        *,
        request_model: Optional[Type["BaseModel"]] = None,
    ) -> Union[None, Callable[[Callable[..., Any]], Callable[..., Any]]]:
        """
        Register an RPC method.

        Can be used in two ways:

        1. Direct registration:
            rpc.register("method_name", handler_function)
            rpc.register("validated", handler, request_model=MyModel)

        2. As a decorator (legacy):
            @rpc.register()
            def handler(): ...

        Args:
            name: Method name (required for direct registration)
            func: Handler function (for direct registration)
            request_model: Optional Pydantic model for request validation.
                          Requires pydantic to be installed.

        Returns:
            None for direct registration, decorator function for decorator usage

        Raises:
            ImportError: If request_model is provided but pydantic is not installed
        """
        # Validate that pydantic is available if request_model is specified
        if request_model is not None and not _check_pydantic():
            raise ImportError(
                "Pydantic is required for request validation. "
                "Install it with: pip install pydantic"
            )

        # Direct registration: register("name", func)
        if name is not None and func is not None:
            self._methods[name] = func
            self._schemas[name] = request_model
            return None

        # Decorator usage: @register() or @register(name="x")
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            method_name = name if name is not None else f.__name__
            self._methods[method_name] = f
            self._schemas[method_name] = request_model
            return f

        return decorator

    def _dispatch(self, method: str, args: Any) -> Any:
        """
        Internal dispatch function called by the native worker.

        Args:
            method: Method name to call
            args: Arguments (already deserialized from MessagePack)

        Returns:
            Result to be serialized and sent back

        Raises:
            ValueError: If method not found or validation fails
        """
        if method not in self._methods:
            raise ValueError(f"Method not found: {method}")

        func = self._methods[method]
        schema = self._schemas.get(method)

        # Validate arguments if schema is provided
        if schema is not None:
            if not isinstance(args, dict):
                raise ValueError(
                    f"Expected dict arguments for Pydantic validation, got {type(args).__name__}"
                )
            args = _validate_with_pydantic(schema, args)

        # Call the method with appropriate argument unpacking
        if isinstance(args, dict):
            return func(**args)
        elif isinstance(args, (list, tuple)):
            return func(*args)
        else:
            return func(args)

    def listen(self) -> None:
        """
        Start listening for RPC calls.

        This method blocks and runs the worker loop until interrupted
        (e.g., by Ctrl+C).
        """
        # Import the native extension
        try:
            from pycrust._worker import run_worker
        except ImportError as e:
            raise ImportError(
                "Failed to import native extension. "
                "Make sure pycrust-worker is installed: "
                "cd crates/pycrust-worker && maturin develop"
            ) from e

        print(f"[pycrust] Starting worker for service: {self.service_name}")
        print(f"[pycrust] Registered methods: {list(self._methods.keys())}")

        try:
            run_worker(self.service_name, self._dispatch)
        except KeyboardInterrupt:
            print("\n[pycrust] Worker stopped.")
