from dataclasses import dataclass
from enum import Enum


class ObjectKind(Enum):
    BLOCK = "Block"
    EMBED = "Embed"
    DIST = "Dist"


@dataclass
class AllocateCommand:
    kind: ObjectKind
    id_offset: int
    count: int


@dataclass
class EmbedTextCommand:
    embed_id: int
    token_id: int
    position_id: int


@dataclass
class EmbedImageCommand:
    embed_id: int
    url: str


@dataclass
class FillBlockCommand:
    block_id: int
    start: int
    end: int


@dataclass
class MaskBlockCommand:
    block_id: int
    start: int
    end: int


@dataclass
class CopyBlockCommand:
    src_block_id: int
    dst_block_id: int
    src_start: int
    dst_start: int
    length: int


@dataclass
class DecodeTokenDistributionCommand:
    dist_id: int
    token_id: int


@dataclass
class SampleTopKCommand:
    dist_id: int
    k: int


@dataclass
class SampleTopKResponse:
    token_id: int
    prob: float


@dataclass
class GetTokenDistributionCommand:
    dist_id: int


@dataclass
class GetTokenDistributionResponse:
    token_id: int
    prob: float


class Engine:
    def __init__(self):
        self.embeds = {}

        # llm
        # tokens

    def allocate(self, cmds: list[AllocateCommand]):
        # in current implementation, all allocations are already done in the constructor.
        # but in the future, we may want to allocate more blocks than the GPU capacity, by offloading some of the blocks to the CPU memory.
        # This logic should handle that case.
        ...

    def deallocate(self, cmds: list[AllocateCommand]):
        ...

    def embed_text(self, cmds: list[EmbedTextCommand]):
        ...

    def embed_image(self, cmds: list[EmbedImageCommand]):
        # if the input/output embeds are not "in use", do it one the second buffer.
        ...

    def fill_block(self, cmds: list[FillBlockCommand]):
        ...

    def mask_block(self, cmds: list[MaskBlockCommand]):
        ...

    def copy_block(self, cmds: list[CopyBlockCommand]):
        ...

    def decode_token_distribution(self, cmds: list[DecodeTokenDistributionCommand]):
        ...

    def sample_top_k_request(self, cmds: list[SampleTopKCommand]) -> list[SampleTopKResponse]:
        ...

    def get_token_distribution(self, cmds: list[GetTokenDistributionCommand]) -> list[GetTokenDistributionResponse]:
        ...
