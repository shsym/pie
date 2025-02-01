from __future__ import annotations

import collections
from enum import Enum

from .command import Command
from .common import BlockId


class ThreadState(Enum):
    NEW = -1  # not started yet
    SUSPENDED = 0  # waiting for notification
    RUNNING = 2  # in progress
    TERMINATED = 3


class Thread:
    state: ThreadState

    block_ids: list[BlockId]

    command_queue: collections.deque[Command]
    # minimally guaranteed ticks for the thread. this is needed to prevent too frequent context switching
    ticks: int
    min_ticks: int

    idle_time: float

    def __init__(self):
        self.state = ThreadState.NEW
        self.block_ids = []
        self.command_queue = collections.deque()
        self.ticks = 0
        self.min_ticks = 6
        self.ticks_idle = 0
        self.idle_time = 0

    def ran_min_ticks(self):
        return self.ticks % self.min_ticks == 0 and self.ticks > 0
