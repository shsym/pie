"""
Memory pooling system for performance optimization.

Provides efficient memory management to minimize allocation overhead
and achieve <5% performance impact during debugging operations.
"""

import threading
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict, deque
import weakref
import gc
from dataclasses import dataclass
import time


@dataclass
class MemoryBlock:
    """Represents a memory block in the pool."""
    size: int
    data: bytearray
    in_use: bool
    allocated_at: float
    last_used: float


class ManagedByteArray:
    """Bytearray wrapper that tracks its parent for memory pool deallocation."""

    def __init__(self, data: bytearray, parent: bytearray):
        self.data = data
        self.parent = parent

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __bytes__(self):
        return bytes(self.data)

    def __repr__(self):
        return f"ManagedByteArray({self.data!r})"


class MemoryPool:
    """
    Thread-safe memory pool for reducing allocation overhead.
    """

    def __init__(
        self,
        block_sizes: List[int] = None,
        max_blocks_per_size: int = 10,
        cleanup_interval: float = 60.0
    ):
        """
        Initialize memory pool.

        Args:
            block_sizes: List of block sizes to pre-allocate (defaults to common sizes)
            max_blocks_per_size: Maximum number of blocks per size category
            cleanup_interval: Interval in seconds for automatic cleanup
        """
        self.block_sizes = block_sizes or [
            1024,      # 1KB
            4096,      # 4KB
            16384,     # 16KB
            65536,     # 64KB
            262144,    # 256KB
            1048576,   # 1MB
            4194304    # 4MB
        ]

        self.max_blocks_per_size = max_blocks_per_size
        self.cleanup_interval = cleanup_interval

        # Thread-safe storage
        self._lock = threading.RLock()
        self._pools: Dict[int, deque] = defaultdict(deque)
        self._allocated_blocks: Dict[id, MemoryBlock] = {}
        self._last_cleanup = time.time()

        # Statistics
        self._stats = {
            'total_allocations': 0,
            'pool_hits': 0,
            'pool_misses': 0,
            'bytes_allocated': 0,
            'bytes_freed': 0
        }

    def allocate(self, size: int) -> 'ManagedByteArray':
        """
        Allocate memory block from pool.

        Args:
            size: Size in bytes

        Returns:
            Memory block as ManagedByteArray that knows its parent
        """
        with self._lock:
            self._stats['total_allocations'] += 1

            # Find appropriate block size (next power of 2 or closest available)
            block_size = self._find_block_size(size)

            # Try to get from pool first
            pool = self._pools[block_size]
            if pool:
                block = pool.popleft()
                block.in_use = True
                block.last_used = time.time()
                self._allocated_blocks[id(block.data)] = block
                self._stats['pool_hits'] += 1
                self._stats['bytes_allocated'] += block_size

                # Return managed bytearray that tracks its parent
                return ManagedByteArray(block.data[:size], block.data)

            # Pool miss - create new block
            self._stats['pool_misses'] += 1
            self._stats['bytes_allocated'] += block_size

            data = bytearray(block_size)
            block = MemoryBlock(
                size=block_size,
                data=data,
                in_use=True,
                allocated_at=time.time(),
                last_used=time.time()
            )

            self._allocated_blocks[id(data)] = block

            # Periodic cleanup
            if time.time() - self._last_cleanup > self.cleanup_interval:
                self._cleanup_expired_blocks()

            return ManagedByteArray(data[:size], data)

    def deallocate(self, data) -> bool:
        """
        Return memory block to pool.

        Args:
            data: Memory block to deallocate (ManagedByteArray or bytearray)

        Returns:
            True if successfully returned to pool
        """
        with self._lock:
            # Get parent buffer if this is a ManagedByteArray
            if isinstance(data, ManagedByteArray):
                parent_buffer = data.parent
            else:
                parent_buffer = data

            block_id = id(parent_buffer)

            if block_id not in self._allocated_blocks:
                return False

            block = self._allocated_blocks.pop(block_id)
            block.in_use = False
            block.last_used = time.time()

            self._stats['bytes_freed'] += block.size

            # Return to appropriate pool if not at capacity
            pool = self._pools[block.size]
            if len(pool) < self.max_blocks_per_size:
                # Clear the memory for security
                block.data[:] = b'\x00' * block.size
                pool.append(block)
                return True

            # Pool at capacity - let GC handle it
            return True

    def _find_block_size(self, requested_size: int) -> int:
        """Find the best block size for requested size."""
        for block_size in self.block_sizes:
            if block_size >= requested_size:
                return block_size

        # If larger than any predefined size, round up to next MB
        return ((requested_size + 1048575) // 1048576) * 1048576

    def _cleanup_expired_blocks(self, max_age_seconds: float = 300.0) -> None:
        """Clean up old unused blocks."""
        current_time = time.time()
        self._last_cleanup = current_time

        for block_size, pool in self._pools.items():
            # Remove blocks older than max_age_seconds
            expired_count = 0
            pool_size = len(pool)

            for _ in range(pool_size):
                if pool and current_time - pool[0].last_used > max_age_seconds:
                    pool.popleft()
                    expired_count += 1
                else:
                    break

            if expired_count > 0:
                # Force garbage collection periodically
                if expired_count > 5:
                    gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            pool_stats = {}
            total_pooled_blocks = 0
            total_pooled_bytes = 0

            for size, pool in self._pools.items():
                count = len(pool)
                pool_stats[f'{size}_bytes'] = count
                total_pooled_blocks += count
                total_pooled_bytes += size * count

            stats = self._stats.copy()
            stats.update({
                'active_blocks': len(self._allocated_blocks),
                'pooled_blocks': total_pooled_blocks,
                'pooled_bytes': total_pooled_bytes,
                'pool_efficiency': self._stats['pool_hits'] / max(self._stats['total_allocations'], 1),
                'pool_stats': pool_stats
            })

            return stats

    def force_cleanup(self) -> None:
        """Force immediate cleanup of all unused blocks."""
        with self._lock:
            for pool in self._pools.values():
                pool.clear()
            gc.collect()

    def resize_pools(self, new_max_blocks: int) -> None:
        """Resize pool capacity."""
        with self._lock:
            self.max_blocks_per_size = new_max_blocks

            # Trim existing pools if necessary
            for pool in self._pools.values():
                while len(pool) > new_max_blocks:
                    pool.pop()


# Global memory pool instance
_global_memory_pool = None
_pool_lock = threading.Lock()


def get_memory_pool() -> MemoryPool:
    """Get global memory pool instance (singleton)."""
    global _global_memory_pool

    if _global_memory_pool is None:
        with _pool_lock:
            if _global_memory_pool is None:
                _global_memory_pool = MemoryPool()

    return _global_memory_pool


class PooledTensorBuffer:
    """
    Tensor buffer using memory pool for efficient allocation.
    """

    def __init__(self, size: int, dtype_size: int = 4):
        """
        Initialize pooled tensor buffer.

        Args:
            size: Number of elements
            dtype_size: Size of each element in bytes (4 for float32)
        """
        self.size = size
        self.dtype_size = dtype_size
        self.byte_size = size * dtype_size

        self._pool = get_memory_pool()
        self._buffer = self._pool.allocate(self.byte_size)
        self._released = False

    def as_bytes(self) -> bytearray:
        """Get buffer as bytes."""
        if self._released:
            raise RuntimeError("Buffer has been released")
        return self._buffer

    def release(self) -> None:
        """Release buffer back to pool."""
        if not self._released:
            self._pool.deallocate(self._buffer)
            self._released = True

    def __del__(self):
        """Automatic cleanup on destruction."""
        if not self._released:
            try:
                self.release()
            except:
                pass  # Ignore errors during cleanup

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


def allocate_tensor_buffer(size: int, dtype_size: int = 4) -> PooledTensorBuffer:
    """
    Allocate tensor buffer from memory pool.

    Args:
        size: Number of elements
        dtype_size: Size of each element in bytes

    Returns:
        Pooled tensor buffer
    """
    return PooledTensorBuffer(size, dtype_size)


def get_pool_stats() -> Dict[str, Any]:
    """Get global memory pool statistics."""
    return get_memory_pool().get_stats()


def cleanup_memory_pool() -> None:
    """Force cleanup of global memory pool."""
    get_memory_pool().force_cleanup()