from __future__ import annotations

import asyncio
from abc import ABC


class Command(ABC):
    event: ResponseEvent | None

    def __init__(self, event: ResponseEvent | None = None):
        self.event = event

    def is_blocking(self) -> bool:
        return self.event is not None

    async def response(self) -> dict:
        if self.event is None:
            return {}
        await self.event.event.wait()
        return self.event.response

    def resolve(self, data: dict = None):
        if self.event is not None:
            self.event(data)


class Transform(Command):
    token_ids: list[int]

    def __init__(self, token_ids: list[int]):
        super().__init__()
        self.token_ids = token_ids


### task split


# HINT: place the thread in the cpu memory
class Yield(Command):

    def __init__(self):
        super().__init__()


class Terminate(Command):
    def __init__(self):
        super().__init__()


class Next(Command):
    n: int

    def __init__(self, n: int):
        super().__init__(ResponseEvent())
        self.n = n


class Read(Command):
    indices: list[int]

    def __init__(self, indices: list[int]):
        super().__init__(ResponseEvent())
        self.indices = indices


class Purge(Command):
    def __init__(self):
        super().__init__()


class ResponseEvent:
    event: asyncio.Event
    response: dict

    def __init__(self):
        self.event = asyncio.Event()

    def __call__(self, data: dict = None):
        self.response = data
        self.event.set()
