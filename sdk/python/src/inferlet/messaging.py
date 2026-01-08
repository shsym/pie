"""
Messaging functions for communicating with the remote user client.
"""

from wit_world.imports import message as _message
from wit_world.imports import inferlet_core_common as _common
from .async_utils import await_future


class Blob:
    """Represents a binary blob that can be sent/received."""

    def __init__(self, inner: _common.Blob) -> None:
        self._inner = inner

    @classmethod
    def new(cls, data: bytes) -> "Blob":
        """Create a new Blob from binary data."""
        return cls(_common.Blob(data))

    @classmethod
    def _from_inner(cls, inner: _common.Blob) -> "Blob":
        """
        Internal factory to create Blob from WIT binding.

        This is used when receiving blobs from WIT functions that return
        blob resources directly (e.g., download_adapter).
        """
        blob = cls.__new__(cls)
        blob._inner = inner
        return blob

    @property
    def size(self) -> int:
        """Get the size of the blob."""
        return self._inner.size()

    def read(self, offset: int, n: int) -> bytes:
        """Read n bytes from the blob starting at offset."""
        return self._inner.read(offset, n)

    def __bytes__(self) -> bytes:
        """Convert entire blob to bytes."""
        return self.read(0, self.size)


def send(message: str, *, streaming: bool = False) -> None:
    """
    Sends a message to the remote user client.

    Args:
        message: The message to send
        streaming: If True, indicates this is a partial/streaming update
                   (Note: streaming flag is for semantic clarity, actual behavior
                   is the same - messages are sent immediately)
    """
    _message.send(message)


def receive(*, timeout: float | None = None) -> str:
    """
    Receives an incoming message from the remote user client.
    Blocks until a message arrives.

    Args:
        timeout: Optional timeout in seconds (not yet implemented)

    Returns:
        The received message
    """
    # Note: timeout not yet implemented in WIT interface
    result = _message.receive()
    return await_future(result, "receive() returned None")


def send_blob(blob: Blob) -> None:
    """Sends a blob to the remote user client."""
    _message.send_blob(blob._inner)


def receive_blob() -> Blob:
    """
    Receives an incoming blob from the remote user client.
    Blocks until a blob arrives.
    """
    result = _message.receive_blob()
    inner = await_future(result, "receive_blob() returned None")
    return Blob(inner)


def broadcast(topic: str, message: str) -> None:
    """
    Publishes a message to a topic (broadcast to all subscribers).

    Args:
        topic: The topic to broadcast to
        message: The message to broadcast
    """
    _message.broadcast(topic, message)


class Subscription:
    """Subscription to a broadcast topic."""

    def __init__(self, inner: _message.Subscription) -> None:
        self._inner = inner

    def get(self) -> str | None:
        """Get next message if available, returns None if no message ready."""
        return self._inner.get()

    def unsubscribe(self) -> None:
        """Cancel the subscription."""
        self._inner.unsubscribe()

    def __enter__(self) -> "Subscription":
        return self

    def __exit__(self, *args: object) -> None:
        self.unsubscribe()


def subscribe(topic: str) -> Subscription:
    """
    Subscribes to a topic and returns a subscription handle.

    Args:
        topic: The topic to subscribe to

    Returns:
        A Subscription that can be used to receive messages
    """
    return Subscription(_message.subscribe(topic))
