"""
IPC-based control channel for multi-GPU coordination.

This module provides a lightweight alternative to GLOO for broadcasting
metadata/control messages from rank 0 to worker processes. Uses
multiprocessing.Queue for cross-process communication.
"""

from __future__ import annotations

import torch.multiprocessing as mp
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
        group_topology: list[list[int]] | None = None,
    ):
        """
        Initialize control channel for this rank.

        Args:
            rank: This process's rank (0 = sender, >0 = receiver)
            world_size: Total number of processes
            queues: List of queues, one per worker (indexed by rank-1)
            group_topology: Optional list of groups, where each group is a list of ranks.
                            If None, assumes a single group containing all ranks.
        """
        self.rank = rank
        self.world_size = world_size
        self.queues = queues  # queues[i] is the queue for worker rank i+1

        if group_topology is None:
            # Default: Single group with all ranks
            self.group_topology = [list(range(world_size))]
        else:
            self.group_topology = group_topology

    def send(self, data: Any, destination_group: int | None = None) -> None:
        """
        Broadcast data to workers (called by rank 0 only).

        Args:
            data: Any picklable Python object
            destination_group: If defined, sends only to workers in this group.
                               If None, broadcasts to ALL workers.
        """
        if self.rank != 0:
            raise RuntimeError("Only rank 0 can send on the control channel")

        if destination_group is None:
            # Broadcast to all workers
            for q in self.queues:
                q.put(data)
        else:
            # Send only to workers in specific group
            if destination_group >= len(self.group_topology):
                raise ValueError(f"Invalid destination group {destination_group}")

            target_ranks = self.group_topology[destination_group]
            for r in target_ranks:
                if r == 0:
                    continue  # Skip self (Rank 0)

                # Worker queue index is rank - 1
                q_idx = r - 1
                if 0 <= q_idx < len(self.queues):
                    self.queues[q_idx].put(data)

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

    def get_group_id(self) -> int:
        """Return the group ID this rank belongs to."""
        for i, group in enumerate(self.group_topology):
            if self.rank in group:
                return i
        raise ValueError(f"Rank {self.rank} not found in any group topology")


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
    # Use spawn context to match torch.multiprocessing.spawn
    ctx = mp.get_context("spawn")
    return [ctx.Queue() for _ in range(world_size - 1)]
