# Simple websocket-based RPC

import asyncio
import collections
import json
from typing import Any

import websockets
import websockets.exceptions

uid_counter = collections.defaultdict(int)

SessionId = int
AwkId = int


def get_uid(cat: str) -> int:
    uid_counter[cat] += 1
    return uid_counter[cat]


class ResponseEvent(asyncio.Event):
    response: Any

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.response = None


class Client:
    socket: websockets.WebSocketClientProtocol
    connected: bool

    sid: SessionId

    cmd_buffer: list[dict]
    handler: asyncio.Task
    events: dict[AwkId, ResponseEvent]

    def __init__(self):
        self.connected = False
        self.cmd_buffer = []
        self.events = {}

    async def connect(self, host: str, port: int):
        if self.connected:
            raise ValueError("already connected")

        self.socket = await websockets.connect(f"ws://{host}:{port}")

        # Initial handshake
        await self.socket.send(json.dumps({
            "handler": "init",
        }))
        self.sid = int(await self.socket.recv())

        self.handler = asyncio.create_task(self.handle())
        self.connected = True

    async def disconnect(self):
        await self.socket.close()
        self.connected = False

    async def handle(self):
        while True:
            try:
                message = json.loads(await self.socket.recv())
                awk_id = message["awk_id"]
                if awk_id in self.events:
                    self.events[awk_id].response = message["response"]
                    self.events[awk_id].set()

            except websockets.exceptions.ConnectionClosed:
                self.connected = False
                break

    async def flush(self):
        try:
            await self.socket.send(json.dumps(self.cmd_buffer))
            self.cmd_buffer.clear()

        except websockets.exceptions.ConnectionClosed:
            raise ConnectionError("connection closed")

    def send(self, handler: str, data: dict):
        self.cmd_buffer.append({
            "handler": handler,
            "data": data
        })

    async def query(self, fn: str, args: dict) -> Any:

        awk_id = get_uid("awk")

        self.cmd_buffer.append({
            "awk_id": awk_id,
            "handler": fn,
            "data": args
        })

        await self.flush()

        self.events[awk_id] = ResponseEvent()
        await self.events[awk_id].wait()

        response = self.events[awk_id].response
        del self.events[awk_id]

        return response


class Session:
    sid: SessionId

    def __init__(self, sid: SessionId):
        self.sid = sid


class Server:
    server_socket: websockets.WebSocketServerProtocol
    handlers: dict[str, callable]
    sessions: dict[SessionId, Session]
    events: dict[AwkId, ResponseEvent]

    connected: bool

    def __init__(self):
        self.connected = False
        self.cmd_buffer = []
        self.events = {}
        self.handlers = {}
        self.sessions = {}

        # register all the handlers

    async def start(self, host: str, port: int):
        self.server_socket = await websockets.serve(self.handle, host, port)

    async def close(self):
        await self.server_socket.disconnect()

    def has_handler(self, event: str) -> bool:
        return event in self.handlers

    def set_handler(self, event: str, handler: callable):

        # make sure func is async
        if not asyncio.iscoroutinefunction(handler):
            raise ValueError("handler functions must be async")

        self.handlers[event] = handler

    async def handle(self, socket):
        sid = get_uid("sid")
        session = Session(sid)
        self.sessions[sid] = session
        await self.begin_session(session)
        try:
            async for payload in socket:
                message_list = json.loads(payload)

                if not isinstance(message_list, list):
                    message_list = [message_list]

                for message in message_list:

                    if "handler" not in message:
                        break

                    match message["handler"]:
                        case "init":
                            await socket.send(str(sid))
                            continue
                        case "close":
                            break
                        case event:
                            fn = getattr(self, "_handle_" + event)
                            response = await fn(session, **message["data"])
                            if "awk_id" in message:
                                packet = {
                                    "awk_id": message["awk_id"],
                                    "response": response
                                }
                                await socket.send(json.dumps(packet))
        except Exception as e:
            print(e)

        await self.end_session(session)
        del self.sessions[sid]

    async def begin_session(self, session: Session):
        ...

    async def end_session(self, session: Session):
        ...


def handle(event: str):
    class HandlerWrapper:
        def __init__(self, fn):
            self.fn = fn

        def __set_name__(self, owner, name):
            setattr(owner, "_handle_" + event, self.fn)

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

    return HandlerWrapper
