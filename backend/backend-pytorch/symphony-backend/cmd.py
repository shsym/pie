from dataclasses import dataclass
from enum import StrEnum
from typing import Union

from blocks import BlockManager


# =============================================================================
# 1) Enums & Dataclasses for Commands/Responses
# =============================================================================

class CommandKind(StrEnum):
    ALLOCATE_BLOCKS = "AllocateBlocks"
    ALLOCATE_BLOCK = "AllocateBlock"
    COPY = "Copy"
    DROP = "Drop"
    FREE_BLOCK = "FreeBlock"
    FREE_BLOCKS = "FreeBlocks"
    AVAILABLE_BLOCKS = "AvailableBlocks"
    # Add more if needed


@dataclass
class AllocateBlocksCmd:
    num_blocks: int


@dataclass
class AllocateBlockCmd:
    pass


@dataclass
class CopyCmd:
    src_block_id: int
    dst_block_id: int
    src_start: int
    dst_start: int
    length: int


@dataclass
class DropCmd:
    block_id: int
    start: int
    end: int


@dataclass
class FreeBlockCmd:
    block_id: int


@dataclass
class FreeBlocksCmd:
    block_id_offset: int
    count: int


@dataclass
class AvailableBlocksCmd:
    pass


CommandPayload = Union[
    AllocateBlocksCmd,
    AllocateBlockCmd,
    CopyCmd,
    DropCmd,
    FreeBlockCmd,
    FreeBlocksCmd,
    AvailableBlocksCmd,
]


class ServerState:
    block_mgr: BlockManager
    ...


def process_commands(state: ServerState, cmd: CommandPayload):


    ...
