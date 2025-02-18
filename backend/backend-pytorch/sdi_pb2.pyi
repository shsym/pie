from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObjectKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    OBJECT_KIND_UNSPECIFIED: _ClassVar[ObjectKind]
    OBJECT_KIND_KV_BLOCK: _ClassVar[ObjectKind]
    OBJECT_KIND_EMB: _ClassVar[ObjectKind]
    OBJECT_KIND_DIST: _ClassVar[ObjectKind]
OBJECT_KIND_UNSPECIFIED: ObjectKind
OBJECT_KIND_KV_BLOCK: ObjectKind
OBJECT_KIND_EMB: ObjectKind
OBJECT_KIND_DIST: ObjectKind

class Allocate(_message.Message):
    __slots__ = ("kind", "object_id_offset", "count")
    KIND_FIELD_NUMBER: _ClassVar[int]
    OBJECT_ID_OFFSET_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    kind: ObjectKind
    object_id_offset: int
    count: int
    def __init__(self, kind: _Optional[_Union[ObjectKind, str]] = ..., object_id_offset: _Optional[int] = ..., count: _Optional[int] = ...) -> None: ...

class BatchAllocate(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[Allocate]
    def __init__(self, items: _Optional[_Iterable[_Union[Allocate, _Mapping]]] = ...) -> None: ...

class BatchDeallocate(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[Allocate]
    def __init__(self, items: _Optional[_Iterable[_Union[Allocate, _Mapping]]] = ...) -> None: ...

class EmbedText(_message.Message):
    __slots__ = ("embedding_id", "token_id", "position_id")
    EMBEDDING_ID_FIELD_NUMBER: _ClassVar[int]
    TOKEN_ID_FIELD_NUMBER: _ClassVar[int]
    POSITION_ID_FIELD_NUMBER: _ClassVar[int]
    embedding_id: int
    token_id: int
    position_id: int
    def __init__(self, embedding_id: _Optional[int] = ..., token_id: _Optional[int] = ..., position_id: _Optional[int] = ...) -> None: ...

class BatchEmbedText(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[EmbedText]
    def __init__(self, items: _Optional[_Iterable[_Union[EmbedText, _Mapping]]] = ...) -> None: ...

class EmbedImage(_message.Message):
    __slots__ = ("embedding_ids", "url")
    EMBEDDING_IDS_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    embedding_ids: _containers.RepeatedScalarFieldContainer[int]
    url: str
    def __init__(self, embedding_ids: _Optional[_Iterable[int]] = ..., url: _Optional[str] = ...) -> None: ...

class BatchEmbedImage(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[EmbedImage]
    def __init__(self, items: _Optional[_Iterable[_Union[EmbedImage, _Mapping]]] = ...) -> None: ...

class FillBlock(_message.Message):
    __slots__ = ("block_id", "context_block_ids", "input_embedding_ids", "output_embedding_ids")
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_BLOCK_IDS_FIELD_NUMBER: _ClassVar[int]
    INPUT_EMBEDDING_IDS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_EMBEDDING_IDS_FIELD_NUMBER: _ClassVar[int]
    block_id: int
    context_block_ids: _containers.RepeatedScalarFieldContainer[int]
    input_embedding_ids: _containers.RepeatedScalarFieldContainer[int]
    output_embedding_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, block_id: _Optional[int] = ..., context_block_ids: _Optional[_Iterable[int]] = ..., input_embedding_ids: _Optional[_Iterable[int]] = ..., output_embedding_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class BatchFillBlock(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[FillBlock]
    def __init__(self, items: _Optional[_Iterable[_Union[FillBlock, _Mapping]]] = ...) -> None: ...

class MaskBlock(_message.Message):
    __slots__ = ("block_id", "mask")
    BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    block_id: int
    mask: _containers.RepeatedScalarFieldContainer[bool]
    def __init__(self, block_id: _Optional[int] = ..., mask: _Optional[_Iterable[bool]] = ...) -> None: ...

class BatchMaskBlock(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[MaskBlock]
    def __init__(self, items: _Optional[_Iterable[_Union[MaskBlock, _Mapping]]] = ...) -> None: ...

class CopyBlock(_message.Message):
    __slots__ = ("source_block_id", "destination_block_id", "source_start", "destination_start", "length")
    SOURCE_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_BLOCK_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_START_FIELD_NUMBER: _ClassVar[int]
    DESTINATION_START_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    source_block_id: int
    destination_block_id: int
    source_start: int
    destination_start: int
    length: int
    def __init__(self, source_block_id: _Optional[int] = ..., destination_block_id: _Optional[int] = ..., source_start: _Optional[int] = ..., destination_start: _Optional[int] = ..., length: _Optional[int] = ...) -> None: ...

class BatchCopyBlock(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[CopyBlock]
    def __init__(self, items: _Optional[_Iterable[_Union[CopyBlock, _Mapping]]] = ...) -> None: ...

class DecodeTokenDistribution(_message.Message):
    __slots__ = ("embedding_id", "distribution_id")
    EMBEDDING_ID_FIELD_NUMBER: _ClassVar[int]
    DISTRIBUTION_ID_FIELD_NUMBER: _ClassVar[int]
    embedding_id: int
    distribution_id: int
    def __init__(self, embedding_id: _Optional[int] = ..., distribution_id: _Optional[int] = ...) -> None: ...

class BatchDecodeTokenDistribution(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[DecodeTokenDistribution]
    def __init__(self, items: _Optional[_Iterable[_Union[DecodeTokenDistribution, _Mapping]]] = ...) -> None: ...

class SampleTopKRequest(_message.Message):
    __slots__ = ("distribution_id", "k")
    DISTRIBUTION_ID_FIELD_NUMBER: _ClassVar[int]
    K_FIELD_NUMBER: _ClassVar[int]
    distribution_id: int
    k: int
    def __init__(self, distribution_id: _Optional[int] = ..., k: _Optional[int] = ...) -> None: ...

class BatchSampleTopKRequest(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[SampleTopKRequest]
    def __init__(self, items: _Optional[_Iterable[_Union[SampleTopKRequest, _Mapping]]] = ...) -> None: ...

class SampleTopKResponse(_message.Message):
    __slots__ = ("token_ids",)
    TOKEN_IDS_FIELD_NUMBER: _ClassVar[int]
    token_ids: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, token_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class BatchSampleTopKResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[SampleTopKResponse]
    def __init__(self, items: _Optional[_Iterable[_Union[SampleTopKResponse, _Mapping]]] = ...) -> None: ...

class GetTokenDistributionRequest(_message.Message):
    __slots__ = ("distribution_id",)
    DISTRIBUTION_ID_FIELD_NUMBER: _ClassVar[int]
    distribution_id: int
    def __init__(self, distribution_id: _Optional[int] = ...) -> None: ...

class BatchGetTokenDistributionRequest(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[GetTokenDistributionRequest]
    def __init__(self, items: _Optional[_Iterable[_Union[GetTokenDistributionRequest, _Mapping]]] = ...) -> None: ...

class GetTokenDistributionResponse(_message.Message):
    __slots__ = ("distribution",)
    DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    distribution: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, distribution: _Optional[_Iterable[float]] = ...) -> None: ...

class BatchGetTokenDistributionResponse(_message.Message):
    __slots__ = ("items",)
    ITEMS_FIELD_NUMBER: _ClassVar[int]
    items: _containers.RepeatedCompositeFieldContainer[GetTokenDistributionResponse]
    def __init__(self, items: _Optional[_Iterable[_Union[GetTokenDistributionResponse, _Mapping]]] = ...) -> None: ...

class Request(_message.Message):
    __slots__ = ("correlation_id", "allocate", "deallocate", "embed_text", "embed_image", "fill_block", "mask_block", "copy_block", "decode_token_distribution", "sample_top_k_request", "get_token_distribution")
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    ALLOCATE_FIELD_NUMBER: _ClassVar[int]
    DEALLOCATE_FIELD_NUMBER: _ClassVar[int]
    EMBED_TEXT_FIELD_NUMBER: _ClassVar[int]
    EMBED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    FILL_BLOCK_FIELD_NUMBER: _ClassVar[int]
    MASK_BLOCK_FIELD_NUMBER: _ClassVar[int]
    COPY_BLOCK_FIELD_NUMBER: _ClassVar[int]
    DECODE_TOKEN_DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_TOP_K_REQUEST_FIELD_NUMBER: _ClassVar[int]
    GET_TOKEN_DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    correlation_id: int
    allocate: BatchAllocate
    deallocate: BatchDeallocate
    embed_text: BatchEmbedText
    embed_image: BatchEmbedImage
    fill_block: BatchFillBlock
    mask_block: BatchMaskBlock
    copy_block: BatchCopyBlock
    decode_token_distribution: BatchDecodeTokenDistribution
    sample_top_k_request: BatchSampleTopKRequest
    get_token_distribution: BatchGetTokenDistributionRequest
    def __init__(self, correlation_id: _Optional[int] = ..., allocate: _Optional[_Union[BatchAllocate, _Mapping]] = ..., deallocate: _Optional[_Union[BatchDeallocate, _Mapping]] = ..., embed_text: _Optional[_Union[BatchEmbedText, _Mapping]] = ..., embed_image: _Optional[_Union[BatchEmbedImage, _Mapping]] = ..., fill_block: _Optional[_Union[BatchFillBlock, _Mapping]] = ..., mask_block: _Optional[_Union[BatchMaskBlock, _Mapping]] = ..., copy_block: _Optional[_Union[BatchCopyBlock, _Mapping]] = ..., decode_token_distribution: _Optional[_Union[BatchDecodeTokenDistribution, _Mapping]] = ..., sample_top_k_request: _Optional[_Union[BatchSampleTopKRequest, _Mapping]] = ..., get_token_distribution: _Optional[_Union[BatchGetTokenDistributionRequest, _Mapping]] = ...) -> None: ...

class Response(_message.Message):
    __slots__ = ("correlation_id", "sample_top_k", "get_token_distribution")
    CORRELATION_ID_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_TOP_K_FIELD_NUMBER: _ClassVar[int]
    GET_TOKEN_DISTRIBUTION_FIELD_NUMBER: _ClassVar[int]
    correlation_id: int
    sample_top_k: BatchSampleTopKResponse
    get_token_distribution: BatchGetTokenDistributionResponse
    def __init__(self, correlation_id: _Optional[int] = ..., sample_top_k: _Optional[_Union[BatchSampleTopKResponse, _Mapping]] = ..., get_token_distribution: _Optional[_Union[BatchGetTokenDistributionResponse, _Mapping]] = ...) -> None: ...
