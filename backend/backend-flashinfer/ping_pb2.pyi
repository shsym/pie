from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Ping(_message.Message):
    __slots__ = ("correlation_id", "message")
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    correlation_id: int
    message: str
    def __init__(self, correlation_id: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...

class Pong(_message.Message):
    __slots__ = ("correlation_id", "message")
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    correlation_id: int
    message: str
    def __init__(self, correlation_id: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...
