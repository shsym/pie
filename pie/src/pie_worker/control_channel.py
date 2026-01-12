"""
IPC-based control channel for multi-GPU coordination.

This module provides a lightweight alternative to GLOO for broadcasting
metadata/control messages from rank 0 to worker processes. Uses
multiprocessing.Queue for cross-process communication.
"""

from __future__ import annotations

from multiprocessing import Queue
from typing import Any


class ControlChannel:
    """
    IPC-based control channel for multi-GPU metadata broadcasts.

    Replaces GLOO's broadcast_object_list for control messages.
    Rank 0 uses send() to broadcast to all workers.
    Workers use recv() to receive from rank 0.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        queues: list[Queue],
    ):
        """
        Initialize control channel for this rank.

        Args:
            rank: This process's rank (0 = sender, >0 = receiver)
            world_size: Total number of processes
            queues: List of queues, one per worker (indexed by rank-1)
        """
        self.rank = rank
        self.world_size = world_size
        self.queues = queues  # queues[i] is the queue for worker rank i+1

    def send(self, data: Any) -> None:
        """
        Broadcast data to all workers (called by rank 0 only).

        Args:
            data: Any picklable Python object
        """
        if self.rank != 0:
            raise RuntimeError("Only rank 0 can send on the control channel")

        # Put data into each worker's queue
        for q in self.queues:
            q.put(data)

    def recv(self, timeout: float | None = None) -> Any:
        """
        Receive data from rank 0 (called by workers only).

        Args:
            timeout: Optional timeout in seconds (None = block forever)

        Returns:
            The data sent by rank 0

        Raises:
            queue.Empty: If timeout expires
        """
        if self.rank == 0:
            raise RuntimeError("Rank 0 should not receive on the control channel")

        # Each worker has its own queue (indexed by rank-1)
        queue_idx = self.rank - 1
        return self.queues[queue_idx].get(timeout=timeout)

    def cleanup(self) -> None:
        """
        Close all queues to prevent 'leaked semaphore' warnings.

        Should be called by rank 0 during shutdown.
        """
        for q in self.queues:
            # cancel_join_thread prevents the process from hanging if the
            # background thread that feeds the queue is still alive
            q.cancel_join_thread()
            q.close()


def create_control_channels(world_size: int) -> list[Queue]:
    """
    Create the underlying queues for control channels.

    Call this in the parent process BEFORE spawning workers.
    Pass the returned queues to each worker's ControlChannel constructor.

    Args:
        world_size: Total number of processes (including rank 0)

    Returns:
        List of queues, one per worker (world_size - 1 queues)
    """
    # Create one queue per worker (ranks 1 through world_size-1)
    return [Queue() for _ in range(world_size - 1)]
