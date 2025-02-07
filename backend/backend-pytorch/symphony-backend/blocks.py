from __future__ import annotations

from enum import Enum
import numpy as np
import torch

type InstanceId = bytes
type KvBlockId = int
type KvBlockPointer = int


class BlockError(Exception):
    ...


class InstanceAddrSpace:
    blocks: dict[KvBlockId, KvBlockPointer]
    id_counter: int

    def __init__(self):
        self.blocks = {}
        self.id_counter = 0

    def translate(self, block_id: KvBlockId) -> KvBlockPointer:
        return self.blocks[block_id]

    def map(self, block_ptr: KvBlockPointer) -> KvBlockId:
        # create a new id
        block_id = self.id_counter
        self.id_counter += 1
        self.blocks[block_id] = block_ptr

        return block_id

    def unmap(self, block_id: KvBlockId):
        del self.blocks[block_id]

    def block_ids(self) -> list[KvBlockId]:
        return list(self.blocks.keys())


class KvBlockManager:
    # block manager's job is to keep safe address space and storage per each user

    global_addr_space: dict[KvBlockPointer, KvBlock]
    virtual_addr_space: dict[InstanceId, InstanceAddrSpace]

    # in the future, we may want to have multiple block storages
    storage: KvBlockStorage

    def __init__(self, storage: KvBlockStorage):
        self.global_addr_space = {}
        self.virtual_addr_space = {}
        self.storage = storage

    def num_free_blocks(self) -> int:
        return self.storage.num_free_blocks()

    ## --- Methods to manage address space ---
    def create_address_space(self, inst_id: InstanceId):
        self.virtual_addr_space[inst_id] = InstanceAddrSpace()

    def destroy_address_space(self, inst_id: InstanceId):
        for block_id in self.virtual_addr_space[inst_id].block_ids():
            self.delete_block(inst_id, block_id)
        del self.virtual_addr_space[inst_id]

    ## --------------------------------------------------------

    ## --- Just helper functions to make the code more readable ----

    def get_block(self, inst_id: InstanceId, block_id: KvBlockId) -> KvBlock:
        return self.get_blocks(inst_id, [block_id])[0]

    def create_block(self, inst_id: InstanceId) -> KvBlockId:
        return self.create_blocks(inst_id, 1)[0]

    def delete_block(self, inst_id: InstanceId, block: KvBlockId):
        self.delete_blocks(inst_id, [block])

    ## --------------------------------------------------------

    def copy_tokens(self, inst_id: InstanceId, src: KvBlockId, dst: KvBlockId, src_offset: int, dst_offset: int, size: int):
        src_block = self.get_block(inst_id, src)
        dst_block = self.get_block(inst_id, dst)
        self.storage.copy(self.storage, src_block.pointer, dst_block.pointer, src_offset, dst_offset, size)

    def drop_tokens(self, inst_id: InstanceId, block: KvBlockId, start: int, end: int):
        block = self.get_block(inst_id, block)
        block.drop(start, end)

    ## --- Methods to manage blocks --------------------------------

    def get_blocks(self, inst_id: InstanceId, block_ids: list[KvBlockId]) -> list[KvBlock]:
        addr_space = self.virtual_addr_space[inst_id]

        blocks = []

        for block_id in block_ids:
            b_ptr = addr_space.translate(block_id)
            blocks.append(self.global_addr_space[b_ptr])
        return blocks

    def create_blocks(self, inst_id: InstanceId, num_blocks: int) -> list[KvBlockId]:

        addr_space = self.virtual_addr_space[inst_id]

        # first, allocate the blocks in the storage
        block_ptrs = self.storage.allocate(num_blocks)
        block_size = self.storage.block_size

        # then, create the block objects
        blocks_ids = []
        for block_ptr in block_ptrs:
            self.global_addr_space[block_ptr] = KvBlock(block_ptr, StorageTier.GPU, block_size)
            block_id = addr_space.map(block_ptr)
            blocks_ids.append(block_id)

        return blocks_ids

    def create_linked_blocks(self, inst_id: InstanceId, src_inst_id: InstanceId, src_block_ids: list[KvBlockId]) -> list[KvBlockId]:

        dst_addr_space = self.virtual_addr_space[inst_id]
        src_addr_space = self.virtual_addr_space[src_inst_id]

        dst_block_ids = []
        for src_block_id in src_block_ids:
            src_block_ptr = src_addr_space.translate(src_block_id)
            src_block = self.global_addr_space[src_block_ptr]
            src_block.increase_ref_count()

            dst_block_id = dst_addr_space.map(src_block_ptr)
            dst_block_ids.append(dst_block_id)

        return dst_block_ids

    def delete_blocks(self, inst_id: InstanceId, block_ids: list[KvBlockId]):

        addr_space = self.virtual_addr_space[inst_id]

        for block_id in block_ids:

            block_ptr = addr_space.translate(block_id)
            block = self.global_addr_space[block_ptr]
            block.decrease_ref_count()

            addr_space.unmap(block_id)

            if block.ref_count <= 0:
                self.storage.free([block_ptr])
                del self.global_addr_space[block_ptr]

    ## --------------------------------------------------------


class KvBlockStorage:
    ptr: list[torch.Tensor]
    _num_blocks: int
    device: torch.device

    addr_map: dict[KvBlockPointer, int]
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

    def allocate(self, num_blocks: int) -> list[KvBlockPointer]:

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

    def free(self, block_ptrs: list[KvBlockPointer]):
        for block_ptr in block_ptrs:
            self.index[block_ptr] = True

    def copy(self, src: KvBlockStorage, src_ptr: KvBlockPointer, dst_ptr: KvBlockPointer, src_offset: int, dst_offset: int, size: int):

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


class StorageTier(Enum):
    CPU = 0
    GPU = 1
    REMOVED = 2
    SCHEDULED = 3


class KvBlock:
    _location: StorageTier
    _pointer: KvBlockPointer

    _position_ids: list[int]
    _occupancy: list[bool]

    _ref_count: int
    _last_used: int
    _filled: bool

    def __init__(self, ptr: KvBlockPointer, location: StorageTier, block_size: int):
        self._pointer = ptr
        self._location = location
        self._position_ids = [0] * block_size
        self._occupancy = [False] * block_size
        self._filled = False
        self.increase_ref_count()

    def drop(self, start: int, end: int):
        for i in range(start, end):
            self._occupancy[i] = False

    @property
    def size(self) -> int:
        return len(self._position_ids)

    @property
    def pointer(self) -> KvBlockPointer:
        return self._pointer

    @property
    def location(self) -> StorageTier:
        return self._location

    @property
    def ref_count(self) -> int:
        return self._ref_count

    @property
    def position_ids(self) -> list[int]:
        return self._position_ids

    @property
    def occupancy(self) -> list[bool]:
        return self._occupancy

    @property
    def filled(self) -> bool:
        return self._filled

    def set_occupancy(self, occupancy: list[bool]):
        if len(occupancy) != self.size:
            raise BlockError("Invalid occupancy size")
        self._occupancy = occupancy

    def set_position_ids(self, position_ids: list[int]):
        if len(position_ids) != self.size:
            raise BlockError("Invalid position_ids size")
        self._position_ids = position_ids

    def set_filled(self, filled: bool):
        self._filled = filled

    def increase_ref_count(self):
        self._ref_count += 1

    def decrease_ref_count(self):
        self._ref_count -= 1


type TokenEmbeddingPointer = int


# Token-level Embedding. This can be either input embedding out output embedding.

class Embedding:
    _location: StorageTier
    _pointer: TokenEmbeddingPointer


class EmbeddingManager:
    storage: EmbeddingStorage

    def __init__(self, storage: EmbeddingStorage):
        self.storage = storage

    # some stubs

    def is_ready(self, inst_id: InstanceId, url: str) -> bool:
        ...


class EmbeddingStorage:
    ptr: torch.Tensor
    device: torch.device

    addr_map: dict[TokenEmbeddingPointer, int]
    index: np.ndarray

    num_layers: int

    def __init__(self,
                 max_capacity: int,
                 block_dim: int,
                 device: torch.device,
                 dtype=torch.bfloat16):

        self.max_capacity = max_capacity
        self.device = device
        self.addr_map = {}
        self.index = np.ones((max_capacity,), dtype=np.bool_)

        self.ptr = torch.empty((max_capacity, block_dim), device=device, dtype=dtype)

    def allocate(self, num_tokens: int) -> list[TokenEmbeddingPointer]:

        if self.num_free_tokens() == 0:
            raise BlockError("No free blocks available")

        allocated = []
        for i in range(self.max_capacity):

            if len(allocated) == num_tokens:
                break

            if self.index[i]:
                allocated.append(i)

        if len(allocated) < num_tokens:
            raise BlockError(f"Not enough free tokens available. Requested: {num_tokens}, Available: {len(allocated)}")

        for te_ptr in allocated:
            self.index[te_ptr] = False

        return allocated

    def free(self, te_ptrs: list[TokenEmbeddingPointer]):
        for te_ptr in te_ptrs:
            self.index[te_ptr] = True

    def num_free_tokens(self) -> int:
        return self.index.sum()

    def num_bytes(self) -> int:
        return sum([ptr.numel() * ptr.element_size() for ptr in self.ptr])
