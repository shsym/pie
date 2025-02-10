from __future__ import annotations

import abc
import time
from itertools import count

import numpy as np
import torch

type InstanceId = bytes

type Address = int
type BlockPointer = int


class BlockError(Exception):
    ...


class AddressMap:
    mapping: dict[Address, Address]
    _id_generator: count

    def __init__(self):
        self.mapping = {}
        self._id_generator = count(start=0, step=1)

    def resolve(self, virtual_addr: Address) -> Address:
        return self.mapping[virtual_addr]

    def register(self, addr: Address) -> Address:
        # create a new id
        virtual_addr = next(self._id_generator)
        self.mapping[virtual_addr] = addr

        return virtual_addr

    def unregister(self, virtual_addr: Address):
        del self.mapping[virtual_addr]


class Block:
    ptr: int  # physical address
    ptr_secondary: int | None

    reference_count: int
    last_used: float

    def __init__(self, ptr: int, ptr_secondary: int | None = None):
        self.ptr = ptr
        self.ptr_secondary = ptr_secondary
        self.reference_count = 0

        self.last_used = 0  # 0 means never used

    def refresh_last_used(self):
        self.last_used = time.monotonic()

    ## --------------------------------------------------------


class BlockStorage:
    max_capacity: int
    index: np.ndarray

    def __init__(self,
                 max_capacity: int):

        self.max_capacity = max_capacity
        self.index = np.ones((max_capacity,), dtype=np.bool_)

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

    def deallocate(self, block_ptrs: list[BlockPointer]):
        for block_ptr in block_ptrs:
            self.index[block_ptr] = True

    def num_free_blocks(self) -> int:
        return self.index.sum()


class BlockManager[BT:Block, ST:BlockStorage]:
    # block manager's job is to keep safe address space and storage per each user

    blocks: dict[Address, BT]
    addr_assignment: count

    addr_space: dict[InstanceId, AddressMap]

    storage: ST  # primary storage in the GPU
    storage_secondary: ST | None  # secondary storage in the CPU

    def __init__(self, storage: ST, storage_secondary: ST | None = None):
        self.blocks = {}
        self.addr_space = {}
        self.storage = storage
        self.storage_secondary = storage_secondary

        self.addr_assignment = count(start=0, step=1)

    def num_free_blocks(self) -> int:
        return self.storage.num_free_blocks()

    ## --- Methods to manage address space ---
    def create_address_space(self, inst_id: InstanceId):
        self.addr_space[inst_id] = AddressMap()

    def destroy_address_space(self, inst_id: InstanceId):
        for block_id in self.addr_space[inst_id].mapping.values():
            self.deallocate_block(inst_id, block_id)
        del self.addr_space[inst_id]

    ## --------------------------------------------------------

    ## --- Just helper functions to make the code more readable ----

    def get_block(self, inst_id: InstanceId, v_addr: Address) -> BT:
        return self.get_blocks(inst_id, [v_addr])[0]

    def allocate_block(self, inst_id: InstanceId) -> Address:
        return self.allocate_blocks(inst_id, 1)[0]

    def deallocate_block(self, inst_id: InstanceId, v_addr: Address):
        self.delete_blocks(inst_id, [v_addr])

    ## --- Methods to manage blocks --------------------------------

    @abc.abstractmethod
    def create_block(self, ptr: BlockPointer) -> BT:
        ...

    def get_blocks(self, inst_id: InstanceId, v_addrs: list[Address]) -> list[BT]:
        addr_space = self.addr_space[inst_id]

        blocks = []

        for v_addr in v_addrs:
            g_addr = addr_space.resolve(v_addr)
            blocks.append(self.blocks[g_addr])
        return blocks

    def allocate_blocks(self, inst_id: InstanceId, num_blocks: int) -> list[Address]:

        addr_space = self.addr_space[inst_id]

        # first, allocate the blocks in the (primary) storage
        ptrs = self.storage.allocate(num_blocks)

        # then, create the block objects
        v_addrs = []
        for ptr in ptrs:
            block = self.create_block(ptr)
            block.reference_count += 1

            g_addr = next(self.addr_assignment)

            self.blocks[g_addr] = block
            v_addr = addr_space.register(g_addr)
            v_addrs.append(v_addr)

        return v_addrs

    def allocate_linked_blocks(self, inst_id: InstanceId, src_inst_id: InstanceId, src_v_addrs: list[Address]) -> list[Address]:

        dst_addr_space = self.addr_space[inst_id]
        src_addr_space = self.addr_space[src_inst_id]

        dst_v_addrs = []
        for src_v_addr in src_v_addrs:
            g_addr = src_addr_space.resolve(src_v_addr)
            block = self.blocks[g_addr]
            block.reference_count += 1

            dst_v_addr = dst_addr_space.register(g_addr)
            dst_v_addrs.append(dst_v_addr)

        return dst_v_addrs

    def delete_blocks(self, inst_id: InstanceId, v_addrs: list[Address]):

        addr_space = self.addr_space[inst_id]

        for v_addr in v_addrs:

            g_addr = addr_space.resolve(v_addr)
            block = self.blocks[g_addr]
            block.reference_count -= 1

            addr_space.unregister(v_addr)

            if block.reference_count <= 0:

                if block.ptr is not None:
                    self.storage.deallocate([block.ptr])
                if block.ptr_secondary is not None:
                    self.storage_secondary.deallocate([block.ptr_secondary])

                del self.blocks[g_addr]


## ------------------------------------------------------------------------------------------------------------------------------


class KvBlock(Block):
    position_ids: list[int]
    occupancy: list[bool]

    filled: bool

    def __init__(self, ptr: Address):
        super().__init__(ptr)

        self.position_ids = []
        self.occupancy = []
        self.filled = False


class KvBlockStorage(BlockStorage):
    base_ptr: torch.Tensor
    ptr: list[torch.Tensor]

    num_layers: int
    num_head: int
    block_size: int
    block_dim: int

    device: torch.device
    dtype: torch.dtype

    def __init__(self,
                 max_capacity: int,

                 num_layers: int,
                 num_head: int,
                 block_size: int,
                 block_dim: int,
                 device: torch.device,
                 dtype=torch.bfloat16):

        super().__init__(max_capacity)

        self.num_layers = num_layers
        self.num_head = num_head
        self.block_size = block_size
        self.block_dim = block_dim

        self.device = device
        self.dtype = dtype

        self.base_ptr = torch.empty((num_layers, max_capacity, num_head, block_size * 2, block_dim), device=device, dtype=dtype)
        self.ptr = [self.base_ptr[i] for i in range(num_layers)]

    def copy(self, src: KvBlockStorage, src_ptr: Address, dst_ptr: Address, src_offset: int, dst_offset: int, size: int):

        # fast path
        if size == self.block_size:
            for i in range(self.num_layers):
                self.ptr[i][dst_ptr].copy_(src.ptr[i][src_ptr], non_blocking=True)

        else:
            for i in range(self.num_layers):
                self.ptr[i][dst_ptr, :, dst_offset:dst_offset + size].copy_(src.ptr[i][src_ptr, :, src_offset:src_offset + size], non_blocking=True)
                self.ptr[i][dst_ptr, :, dst_offset + self.block_size:dst_offset + self.block_size + size].copy_(
                    src.ptr[i][src_ptr, :, src_offset + self.block_size:src_offset + self.block_size + size], non_blocking=True)

    def num_bytes(self) -> int:
        return sum([ptr.numel() * ptr.element_size() for ptr in self.ptr])


class KvBlockManager(BlockManager[KvBlock, KvBlockStorage]):

    def __init__(self, storage: KvBlockStorage):
        super().__init__(storage)

    def create_block(self, ptr: BlockPointer) -> KvBlock:
        block = KvBlock(ptr)

        block.position_ids = [0] * self.storage.block_size
        block.occupancy = [False] * self.storage.block_size

        return KvBlock(ptr)

    def copy_tokens(self, inst_id: InstanceId, src: Address, dst: Address, src_offset: int, dst_offset: int, size: int):
        src_block = self.get_block(inst_id, src)
        dst_block = self.get_block(inst_id, dst)
        self.storage.copy(self.storage, src_block.ptr, dst_block.ptr, src_offset, dst_offset, size)

    def drop_tokens(self, inst_id: InstanceId, block: Address, start: int, end: int):
        block = self.get_block(inst_id, block)
        block.occupancy[start:end] = [False] * (end - start)


## ------------------------------------------------------------------------------------------------------------------------------


# Token-level Embedding. This can be either input embedding out output embedding.

class TokenEmbed(Block):

    def __init__(self, ptr: Address):
        super().__init__(ptr)


class TokenEmbedStorage(BlockStorage):
    ptr: torch.Tensor

    block_dim: int

    device: torch.device
    dtype: torch.dtype

    def __init__(self,
                 max_capacity: int,
                 block_dim: int,
                 device: torch.device,
                 dtype=torch.bfloat16):
        super().__init__(max_capacity)

        self.block_dim = block_dim

        self.device = device
        self.dtype = dtype

        self.ptr = torch.empty((max_capacity, block_dim), device=device, dtype=dtype)

    def num_bytes(self) -> int:
        return sum([ptr.numel() * ptr.element_size() for ptr in self.ptr])


class TokenEmbedManager(BlockManager[TokenEmbed, TokenEmbedStorage]):

    def __init__(self, storage: TokenEmbedStorage):
        super().__init__(storage)

    def create_block(self, ptr: BlockPointer) -> TokenEmbed:
        emb = TokenEmbed(ptr)
        return emb
