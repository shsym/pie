from __future__ import annotations

from enum import Enum
import numpy as np
import torch

from .common import BlockId, ThreadId


class BlockStorage:
    ptr: list[torch.Tensor]
    _num_blocks: int
    device: torch.device

    addr_map: dict[BlockId, int]
    index: np.ndarray

    num_layers: int

    def __init__(self, num_layers: int, num_blocks: int, num_head: int, block_size: int, block_dim: int, device: torch.device, dtype=torch.float16):
        self._num_blocks = num_blocks
        self.num_layers = num_layers
        self.device = device
        self.addr_map = {}
        self.index = np.ones((num_blocks,), dtype=np.bool_)

        self.base_ptr = torch.empty((num_layers, num_blocks, num_head, block_size * 2, block_dim), device=device, dtype=dtype)
        self.ptr = [self.base_ptr[i] for i in range(num_layers)]

    def translate_addr(self, b_id: BlockId | list[BlockId]) -> int | list[int]:
        if isinstance(b_id, list):
            return [self.addr_map[b] for b in b_id]
        return self.addr_map[b_id]

    def create(self, b_id: BlockId) -> bool:

        # check if the block is already in the storage
        if b_id in self.addr_map:
            return False

        if self.num_free_blocks() == 0:
            return False

        # get the smallest index that is not used
        idx = -1
        for i in range(self._num_blocks):
            if self.index[i]:
                idx = i
                break

        self.addr_map[b_id] = idx
        self.index[idx] = False

    def delete(self, b_id: BlockId) -> bool:
        if b_id not in self.addr_map:
            return False

        idx = self.addr_map[b_id]
        self.index[idx] = True
        del self.addr_map[b_id]

    def has_block(self, b_id: BlockId) -> bool:
        return b_id in self.addr_map

    def num_blocks(self) -> int:
        return self._num_blocks

    def num_free_blocks(self) -> int:
        return self._num_blocks - len(self.addr_map)

    def optimize(self):
        ...

    def relocate(self, b_id: BlockId, storage: BlockStorage) -> bool:

        # first check if the block is in the storage
        if not self.has_block(b_id):
            return False

        # check if there is enough space in the target storage
        if storage.num_free_blocks() == 0:
            return False

        src_addr = self.addr_map[b_id]

        # move the block to the target storage
        storage.create(b_id)
        dst_addr = storage.addr_map[b_id]

        for i in range(self.num_layers):
            src_ptr = self.ptr[i]
            dst_ptr = storage.ptr[i]
            dst_ptr[dst_addr].copy_(src_ptr[src_addr], non_blocking=True)

    def num_bytes(self) -> int:
        return sum([ptr.numel() * ptr.element_size() for ptr in self.ptr])


class BlockLocation(Enum):
    CPU = 0
    GPU = 1
    REMOVED = 2
    SCHEDULED = 3


class Block:
    token_ids: list[int]
    location: BlockLocation
    idle_ticks: int
    refs: list[ThreadId]
    context_hash: str

    # next token distribution
    next_token_id: np.ndarray | None
    next_token_probs: np.ndarray | None
    next_token_probs_rem: np.ndarray | None

    def __init__(self, token_ids: list[int] = None, prev: Block | None = None):
        self.token_ids = token_ids
        self.location = BlockLocation.SCHEDULED
        self.idle_ticks = 0
        self.refs = []

        self.next_token_id = None
        self.next_token_probs = None
        self.next_token_probs_rem = None

        if prev:
            self.context_hash = str(hash(prev.context_hash + str(token_ids)))
        else:
            self.context_hash = str(hash(str(token_ids)))
