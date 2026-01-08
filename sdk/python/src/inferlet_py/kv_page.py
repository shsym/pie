"""
KV Page management for caching key-value pairs in attention.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Queue


class KvPage:
    """
    Reference-counted KV cache page.

    KV pages store key-value pairs for attention caching.
    Use as a context manager for automatic cleanup.

    Reference counting allows multiple contexts to share pages
    (copy-on-write for beam search). Pages are only deallocated
    when the reference count reaches 0.
    """

    def __init__(self, queue: "Queue", ptr: int) -> None:
        self._queue = queue
        self._ptr = ptr
        self._ref_count = 1
        self._released = False

    @property
    def ptr(self) -> int:
        """The raw pointer to the KV page resource."""
        if self._released:
            raise RuntimeError("KV page has been released")
        return self._ptr

    def ref(self) -> None:
        """Increment reference count for shared ownership."""
        if self._released:
            raise RuntimeError("Cannot ref a released KV page")
        self._ref_count += 1

    def retain(self) -> None:
        """Alias for ref() - increment reference count."""
        self.ref()

    def release(self) -> None:
        """
        Decrement reference count and deallocate if it reaches 0.

        Safe to call multiple times - only deallocates once when
        ref count reaches 0.
        """
        if self._released:
            return

        self._ref_count -= 1
        if self._ref_count <= 0:
            self._queue.deallocate_kv_page(self._ptr)
            self._released = True

    def __enter__(self) -> "KvPage":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    def __del__(self) -> None:
        if not self._released and self._ref_count > 0:
            try:
                # Force release on garbage collection
                self._ref_count = 1
                self.release()
            except Exception:
                pass  # Ignore errors during cleanup


class KvPageManager:
    """
    Manager for KV cache pages.

    Handles allocation, deallocation, and lifecycle of KV pages.
    Tracks the total number of tokens in the cache.
    """

    def __init__(self, queue: "Queue", page_size: int) -> None:
        self._queue = queue
        self._page_size = page_size
        self._pages: list[KvPage] = []
        self._last_page_len: int = 0

    @property
    def page_count(self) -> int:
        """Number of allocated pages."""
        return len(self._pages)

    @property
    def last_page_len(self) -> int:
        """Length of the last KV page (tokens stored in it)."""
        return self._last_page_len

    @property
    def total_tokens(self) -> int:
        """Total number of tokens in the KV cache."""
        if len(self._pages) == 0:
            return self._last_page_len
        return (len(self._pages) - 1) * self._page_size + self._last_page_len

    @property
    def pages(self) -> list[KvPage]:
        """List of all managed pages."""
        return [p for p in self._pages if not p._released]

    @property
    def page_ptrs(self) -> list[int]:
        """List of all managed page pointers."""
        return [p.ptr for p in self._pages if not p._released]

    def grow(self, num_tokens: int) -> None:
        """
        Grow the KV cache to accommodate more tokens.

        Args:
            num_tokens: Number of tokens to add
        """
        self._adjust(num_tokens)

    def shrink(self, num_tokens: int) -> None:
        """
        Shrink the KV cache.

        Args:
            num_tokens: Number of tokens to remove
        """
        self._adjust(-num_tokens)

    def _adjust(self, num_tokens: int) -> None:
        """Adjust the KV cache size by the given number of tokens."""
        if num_tokens == 0:
            return

        current_tokens = self.total_tokens
        new_total_tokens = current_tokens + num_tokens

        if new_total_tokens < 0:
            raise ValueError("Token count adjustment resulted in underflow")

        current_pages = len(self._pages)
        if new_total_tokens > 0:
            required_pages = math.ceil(new_total_tokens / self._page_size)
        else:
            required_pages = 0

        if required_pages > current_pages:
            # Grow: Allocate new pages
            new_pages_needed = required_pages - current_pages
            ptrs = self._queue.allocate_kv_pages(new_pages_needed)
            for ptr in ptrs:
                self._pages.append(KvPage(self._queue, ptr))
        elif required_pages < current_pages:
            # Shrink: Release excess pages
            pages_to_release = self._pages[required_pages:]
            self._pages = self._pages[:required_pages]
            for page in pages_to_release:
                page.release()

        # Update the length of the last page
        last_page_len = new_total_tokens % self._page_size
        if last_page_len == 0 and new_total_tokens > 0:
            self._last_page_len = self._page_size
        else:
            self._last_page_len = last_page_len

    def import_pages_from_ptrs(self, ptrs: list[int], last_page_len: int) -> None:
        """
        Import pages from raw pointers (e.g., from cache restore).

        Args:
            ptrs: List of page pointers
            last_page_len: Length of the last page
        """
        self._pages = [KvPage(self._queue, ptr) for ptr in ptrs]
        self._last_page_len = last_page_len

    def export(self, name: str) -> None:
        """
        Export all managed pages with a name for later import.

        Args:
            name: Name to associate with the exported pages
        """
        ptrs = self.page_ptrs
        if ptrs:
            self._queue.export_resources(
                self._queue.KV_PAGE_TYPE, ptrs, name
            )

    def import_pages(self, name: str) -> list[KvPage]:
        """
        Import previously exported pages by name.

        Args:
            name: Name of the exported pages

        Returns:
            List of imported KvPage objects
        """
        ptrs = self._queue.import_resources(self._queue.KV_PAGE_TYPE, name)
        pages = [KvPage(self._queue, ptr) for ptr in ptrs]
        self._pages.extend(pages)
        return pages

    def import_pages_from_state(self, ptrs: list[int], last_page_len: int) -> None:
        """
        Import KV page pointers into the manager.

        This is used to restore a context's state from imported KV pages.
        Unlike import_pages_from_ptrs which wraps ptrs in new KvPage objects,
        this method directly sets the internal state.

        Args:
            ptrs: List of KV page pointers
            last_page_len: Length of the last page
        """
        self._pages = [KvPage(self._queue, ptr) for ptr in ptrs]
        self._last_page_len = last_page_len

    def release_all(self) -> None:
        """Release all managed KV pages."""
        for page in self._pages:
            page.release()
        self._pages.clear()
        self._last_page_len = 0

    def fork(self) -> tuple["KvPageManager", int]:
        """
        Fork with copy-on-write sharing for beam search.

        Only shares FULL pages. Partial pages are dropped because:
        - Shared pages point to the same memory
        - If two contexts write to the same position, they corrupt each other
        - Full pages are read-only (new writes go to new pages)
        - Partial pages would be written to, so must be unique per context

        Returns:
            (forked_manager, dropped_token_count)
        """
        # Only keep full pages (all except possibly the last)
        kept_page_count = max(0, len(self._pages) - 1)
        kept_tokens = kept_page_count * self._page_size
        dropped_token_count = self.total_tokens - kept_tokens

        forked = KvPageManager(self._queue, self._page_size)
        forked._pages = self._pages[:kept_page_count]
        forked._last_page_len = self._page_size if kept_page_count > 0 else 0

        # Increment ref count for shared pages
        for page in forked._pages:
            page.ref()

        return forked, dropped_token_count

    def adopt(self, other: "KvPageManager") -> None:
        """
        Adopt pages from another manager (for beam search winner).

        Releases current pages, then takes ownership of other's pages
        by incrementing their reference counts.

        Args:
            other: The KvPageManager to adopt pages from
        """
        self.release_all()
        self._pages = list(other._pages)
        self._last_page_len = other._last_page_len

        # Increment ref count since we're now an owner
        for page in self._pages:
            page.ref()

    def remove_page_at(self, index: int) -> None:
        """
        Remove a page at the specified index.

        Used by drop_masked_kv_pages to remove fully-masked pages.

        Args:
            index: The index of the page to remove
        """
        if index < 0 or index >= len(self._pages):
            raise IndexError(f"Page index {index} out of bounds")

        removed = self._pages.pop(index)
        removed.release()

        # Recalculate last page length
        new_total_tokens = self.total_tokens
        last_page_len = new_total_tokens % self._page_size
        self._last_page_len = (
            self._page_size if last_page_len == 0 and new_total_tokens > 0 else last_page_len
        )

    def recalculate_last_page_len(self, total_tokens: int) -> None:
        """
        Recalculate last page length based on current page count and total tokens.

        Used after external modifications to pages list.

        Args:
            total_tokens: The total number of committed tokens
        """
        last_page_len = total_tokens % self._page_size
        self._last_page_len = (
            self._page_size if last_page_len == 0 and total_tokens > 0 else last_page_len
        )

    def __enter__(self) -> "KvPageManager":
        return self

    def __exit__(self, *args: object) -> None:
        self.release_all()

    def __len__(self) -> int:
        """Returns total number of tokens in the cache."""
        return self.total_tokens
