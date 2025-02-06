from __future__ import annotations

from enum import Enum
import numpy as np
import torch

type BlockId = int
type BlockPointer = int


class BlockError(Exception):
    ...


class BlockManager:
    addr_space: dict[BlockId, Block]

    # in the future, we may want to have multiple block storages
    storage: BlockStorage

    def __init__(self, storage: BlockStorage):
        self.addr_space = {}
        self.storage = storage

    def num_free_blocks(self) -> int:
        return self.storage.num_free_blocks()

    def get_block(self, block_id: BlockId) -> Block:
        return self.get_blocks([block_id])[0]

    def create_block(self) -> BlockId:
        return self.create_blocks(1)[0]

    def delete_block(self, block: BlockId):
        self.delete_blocks([block])

    def copy_block(self, src: BlockId, dst: BlockId, src_offset: int, dst_offset: int, size: int):
        src_block = self.get_block(src)
        dst_block = self.get_block(dst)
        self.storage.copy(self.storage, src_block.pointer, dst_block.pointer, src_offset, dst_offset, size)

    def get_blocks(self, block_ids: list[BlockId]) -> list[Block]:
        blocks = []
        for block_id in block_ids:
            if block_id not in self.addr_space:
                raise BlockError(f"Block with id {block_id} does not exist")
            blocks.append(self.addr_space[block_id])
        return blocks

    def create_blocks(self, num_blocks: int) -> list[BlockId]:
        # first, allocate the blocks in the storage
        block_ptrs = self.storage.allocate(num_blocks)

        # then, create the block objects
        blocks = []
        for block_ptr in block_ptrs:
            block = Block(block_ptr, BlockLocation.GPU)
            self.addr_space[block_ptr] = block
            blocks.append(block)

        return block_ptrs

    def delete_blocks(self, block_ids: list[BlockId]):

        for block_id in block_ids:

            if block_id not in self.addr_space:
                raise BlockError(f"Block with id {block_id} does not exist")

            block = self.addr_space[block_id]
            block.decrease_ref_count()

            if block.ref_count <= 0:
                self.storage.free([block.pointer])
                del self.addr_space[block_id]


class BlockStorage:
    ptr: list[torch.Tensor]
    _num_blocks: int
    device: torch.device

    addr_map: dict[BlockId, int]
    index: np.ndarray

    num_layers: int

    def __init__(self,
                 num_layers: int,
                 max_capacity: int,
                 num_head: int,
                 block_size: int,
                 block_dim: int,
                 device: torch.device,
                 dtype=torch.bfloat16):
        self.max_capacity = max_capacity
        self.num_layers = num_layers
        self.block_size = block_size
        self.device = device
        self.addr_map = {}
        self.index = np.ones((max_capacity,), dtype=np.bool_)

        self.base_ptr = torch.empty((num_layers, max_capacity, num_head, block_size * 2, block_dim), device=device, dtype=dtype)
        self.ptr = [self.base_ptr[i] for i in range(num_layers)]

    def allocate(self, num_blocks: int) -> list[BlockPointer]:

        if self.num_free_blocks() == 0:
            raise BlockError("No free blocks available")

        allocated = []
        for i in range(self.max_capacity):

            if len(allocated) == num_blocks:
                break

            if self.index[i]:
                allocated.append(i)

        if len(allocated) < num_blocks:
            raise BlockError(f"Not enough free blocks available. Requested: {num_blocks}, Available: {len(allocated)}")

        for block_ptr in allocated:
            self.index[block_ptr] = False

        return allocated

    def free(self, block_ptrs: list[BlockPointer]):
        for block_ptr in block_ptrs:
            self.index[block_ptr] = True

    def copy(self, src: BlockStorage, src_ptr: BlockPointer, dst_ptr: BlockPointer, src_offset: int, dst_offset: int, size: int):

        # fast path
        if size == self.block_size:
            for i in range(self.num_layers):
                self.ptr[i][dst_ptr].copy_(src.ptr[i][src_ptr], non_blocking=True)

        else:
            for i in range(self.num_layers):
                self.ptr[i][dst_ptr, :, dst_offset:dst_offset + size].copy_(src.ptr[i][src_ptr, :, src_offset:src_offset + size], non_blocking=True)
                self.ptr[i][dst_ptr, :, dst_offset + self.block_size:dst_offset + self.block_size + size].copy_(
                    src.ptr[i][src_ptr, :, src_offset + self.block_size:src_offset + self.block_size + size], non_blocking=True)

    def num_free_blocks(self) -> int:
        return self.index.sum()

    def num_bytes(self) -> int:
        return sum([ptr.numel() * ptr.element_size() for ptr in self.ptr])


class BlockLocation(Enum):
    CPU = 0
    GPU = 1
    REMOVED = 2
    SCHEDULED = 3


class Block:
    _location: BlockLocation
    _pointer: BlockPointer

    _position_indices: list[int]
    _occupancy: list[bool]

    _ref_count: int
    _last_used: int

    def __init__(self, ptr: BlockPointer, location: BlockLocation):
        self._pointer = ptr
        self._location = location
        self._position_indices = []
        self._occupancy = []
        self.increase_ref_count()

    @property
    def pointer(self):
        return self._pointer

    @property
    def location(self):
        return self._location

    @property
    def ref_count(self):
        return self._ref_count

    def increase_ref_count(self):
        self._ref_count += 1

    def decrease_ref_count(self):
        self._ref_count -= 1
