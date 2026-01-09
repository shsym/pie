"""Pydantic validation utilities."""

from typing import Any, List, Type, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError


class ValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, message: str, errors: List[Any] | None = None) -> None:
        super().__init__(message)
        self.errors = errors or []


T = TypeVar("T", bound=BaseModel)


def validate_args(model: Type[T], data: Any) -> T:
    """
    Validate arguments against a Pydantic model.

    Args:
        model: Pydantic model class
        data: Data to validate (dict or model instance)

    Returns:
        Validated model instance

    Raises:
        ValidationError: If validation fails

    Example:
        from pydantic import BaseModel

        class UserRequest(BaseModel):
            name: str
            age: int

        validated = validate_args(UserRequest, {"name": "Alice", "age": 30})
        print(validated.name)  # "Alice"
    """
    try:
        if isinstance(data, model):
            return data
        elif isinstance(data, dict):
            return model(**data)
        else:
            raise ValidationError(
                f"Expected dict or {model.__name__}, got {type(data).__name__}"
            )
    except PydanticValidationError as e:
        raise ValidationError(
            f"Validation failed for {model.__name__}", errors=e.errors()
        ) from e
