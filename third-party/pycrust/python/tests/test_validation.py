"""Tests for validation utilities."""

import pytest
from pydantic import BaseModel

from pycrust.validation import ValidationError, validate_args


class UserModel(BaseModel):
    name: str
    age: int
    email: str | None = None


class NestedModel(BaseModel):
    user: UserModel
    score: float


def test_validate_args_with_dict():
    """Test validating a dict against a model."""
    data = {"name": "Alice", "age": 30}
    result = validate_args(UserModel, data)

    assert isinstance(result, UserModel)
    assert result.name == "Alice"
    assert result.age == 30
    assert result.email is None


def test_validate_args_with_optional_field():
    """Test validating with an optional field provided."""
    data = {"name": "Bob", "age": 25, "email": "bob@example.com"}
    result = validate_args(UserModel, data)

    assert result.email == "bob@example.com"


def test_validate_args_with_model_instance():
    """Test validating an already-instantiated model."""
    user = UserModel(name="Charlie", age=35)
    result = validate_args(UserModel, user)

    assert result is user  # Should return the same instance


def test_validate_args_missing_required_field():
    """Test validation error for missing required field."""
    data = {"name": "Dave"}  # Missing 'age'

    with pytest.raises(ValidationError) as exc_info:
        validate_args(UserModel, data)

    assert "Validation failed for UserModel" in str(exc_info.value)
    assert exc_info.value.errors is not None


def test_validate_args_wrong_type():
    """Test validation error for wrong type."""
    data = {"name": "Eve", "age": "not_a_number"}

    with pytest.raises(ValidationError):
        validate_args(UserModel, data)


def test_validate_args_wrong_input_type():
    """Test validation error for completely wrong input type."""
    with pytest.raises(ValidationError, match="Expected dict or UserModel"):
        validate_args(UserModel, "not a dict")


def test_validate_args_nested_model():
    """Test validating nested models."""
    data = {
        "user": {"name": "Frank", "age": 40},
        "score": 95.5,
    }
    result = validate_args(NestedModel, data)

    assert isinstance(result, NestedModel)
    assert isinstance(result.user, UserModel)
    assert result.user.name == "Frank"
    assert result.score == 95.5


def test_validation_error_attributes():
    """Test ValidationError contains error details."""
    data = {"name": "Grace"}  # Missing 'age'

    try:
        validate_args(UserModel, data)
        pytest.fail("Expected ValidationError")
    except ValidationError as e:
        assert len(e.errors) > 0
        # Check that the error has location info
        error = e.errors[0]
        assert "loc" in error or "type" in error
