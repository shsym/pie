from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Request(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Response(_message.Message):
    __slots__ = ("protocols",)
    PROTOCOLS_FIELD_NUMBER: _ClassVar[int]
    protocols: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, protocols: _Optional[_Iterable[str]] = ...) -> None: ...
