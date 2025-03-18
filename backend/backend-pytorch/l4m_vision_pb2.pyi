from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class EmbedImage(_message.Message):
    __slots__ = ("embedding_ids", "image_blob")
    EMBEDDING_IDS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_BLOB_FIELD_NUMBER: _ClassVar[int]
    embedding_ids: _containers.RepeatedScalarFieldContainer[int]
    image_blob: bytes
    def __init__(self, embedding_ids: _Optional[_Iterable[int]] = ..., image_blob: _Optional[bytes] = ...) -> None: ...

class BatchEmbedImage(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[EmbedImage]
    def __init__(self, items: _Optional[_Iterable[_Union[EmbedImage, _Mapping]]] = ...) -> None: ...

class Request(_message.Message):
    __slots__ = ("correlation_id", "embed_image")
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    EMBED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    correlation_id: int
    embed_image: BatchEmbedImage
    def __init__(self, correlation_id: _Optional[int] = ..., embed_image: _Optional[_Union[BatchEmbedImage, _Mapping]] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ("correlation_id",)
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    correlation_id: int
    def __init__(self, correlation_id: _Optional[int] = ...) -> None: ...
