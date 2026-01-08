"""
Binary Run-Length Encoding (BRLE) for attention masks.

This efficiently stores boolean sequences by encoding run lengths.
Mirrors the Rust Brle from inferlet/src/brle.rs and JS from inferlet-js/src/brle.ts.
"""

from __future__ import annotations


class Brle:
    """
    Binary Run-Length Encoding for attention masks.

    Encodes boolean sequences as alternating run lengths of false/true.
    In attention mask context: false = visible (can attend).

    Examples:
        [false, false, true, true, true, false] → [2, 3, 1]
        [true, true, false] → [0, 2, 1]
    """

    def __init__(self, buffer: list[int], total_size: int) -> None:
        self.buffer = buffer
        self.total_size = total_size

    @classmethod
    def new(cls, size: int) -> "Brle":
        """
        Create a Brle with `size` visible positions (all can attend).

        In attention mask semantics: false = visible.
        """
        if size == 0:
            return cls([], 0)
        # All false values = [size] means size visible positions
        return cls([size], size)

    @classmethod
    def from_array(cls, values: list[bool]) -> "Brle":
        """Create a Brle from an array of booleans."""
        if not values:
            return cls.new(0)

        buffer: list[int] = []
        current_val = False
        count = 0

        # Handle initial run of false
        if values[0]:
            buffer.append(0)
            current_val = True

        for val in values:
            if val == current_val:
                count += 1
            else:
                buffer.append(count)
                current_val = val
                count = 1

        buffer.append(count)
        return cls(buffer, len(values))

    def __len__(self) -> int:
        return self.total_size

    def clone(self) -> "Brle":
        """Create a copy of this Brle."""
        return Brle(self.buffer.copy(), self.total_size)

    def append(self, value: bool) -> None:
        """Append a single boolean value."""
        if self.total_size == 0:
            if value:
                self.buffer = [0, 1]
            else:
                self.buffer = [1]
            self.total_size = 1
            return

        self.total_size += 1
        is_last_run_true = len(self.buffer) % 2 == 0

        if value == is_last_run_true:
            self.buffer[-1] += 1
        else:
            self.buffer.append(1)

    def truncate(self, new_size: int) -> "Brle":
        """Return a new Brle truncated to new_size."""
        if new_size >= self.total_size:
            return self.clone()
        if new_size <= 0:
            return Brle.new(0)

        new_buffer: list[int] = []
        remaining = new_size

        for run_len in self.buffer:
            if remaining <= 0:
                break
            if run_len <= remaining:
                new_buffer.append(run_len)
                remaining -= run_len
            else:
                new_buffer.append(remaining)
                remaining = 0

        return Brle(new_buffer, new_size)

    def _slice(self, start: int, end: int) -> "Brle":
        """
        Create a new Brle representing a slice [start, end) of this one.

        Internal method used by remove_range.
        """
        end = min(end, self.total_size)
        if start >= end:
            return Brle.new(0)

        new_size = end - start
        new_buffer: list[int] = []

        current_pos = 0
        for i, run_len in enumerate(self.buffer):
            value = i % 2 != 0  # Even indices are false, odd are true
            r_start = current_pos
            r_end = current_pos + run_len
            current_pos = r_end

            # Calculate intersection of [r_start, r_end) and [start, end)
            slice_r_start = max(r_start, start)
            slice_r_end = min(r_end, end)

            if slice_r_start < slice_r_end:
                length = slice_r_end - slice_r_start

                if not new_buffer:
                    # First run of the new slice
                    if value:
                        new_buffer.append(0)  # Start with zero-length false run
                    new_buffer.append(length)
                else:
                    # Check if we can merge with previous run
                    last_run_is_true = (len(new_buffer) - 1) % 2 != 0
                    if last_run_is_true == value:
                        new_buffer[-1] += length
                    else:
                        new_buffer.append(length)

        return Brle(new_buffer, new_size)

    def remove_range(self, start: int, end: int) -> None:
        """
        Remove a range of boolean values [start, end).

        Modifies this Brle in place by concatenating the parts before
        and after the removed range.
        """
        end = min(end, self.total_size)
        if start >= end:
            return

        head = self._slice(0, start)
        tail = self._slice(end, self.total_size)

        # Merge head and tail
        if head.total_size == 0:
            self.buffer = tail.buffer.copy()
            self.total_size = tail.total_size
        elif tail.total_size == 0:
            self.buffer = head.buffer.copy()
            self.total_size = head.total_size
        else:
            # Check if last run of head and first run of tail have same value
            head_last_is_true = (len(head.buffer) - 1) % 2 != 0
            tail_first_is_true = tail.buffer[0] == 0 and len(tail.buffer) > 1

            if head_last_is_true == tail_first_is_true:
                # Merge the runs
                tail_first_run_len = tail.buffer[1] if tail_first_is_true else tail.buffer[0]
                tail_slice_start = 2 if tail_first_is_true else 1

                new_buffer = head.buffer.copy()
                new_buffer[-1] += tail_first_run_len
                new_buffer.extend(tail.buffer[tail_slice_start:])
                self.buffer = new_buffer
            else:
                # No merge needed
                if tail_first_is_true:
                    self.buffer = head.buffer + tail.buffer[1:]
                else:
                    self.buffer = head.buffer + tail.buffer

            self.total_size = head.total_size + tail.total_size


    def is_empty(self) -> bool:
        """Check if the sequence is empty."""
        return self.total_size == 0

    def iter_runs(self):
        """
        Iterate over the runs, yielding (value, start_index, end_index).

        Yields:
            Tuples of (bool, int, int) representing (value, start, end)
            where value is the boolean value of the run.
        """
        current_pos = 0
        for i, run_len in enumerate(self.buffer):
            value = i % 2 != 0  # Even indices are false, odd are true
            start = current_pos
            end = current_pos + run_len
            current_pos = end
            yield (value, start, end)

    def is_masked(self, indices: list[int]) -> list[bool]:
        """
        Check the boolean values at a given set of indices.

        Returns whether each index is masked (TRUE = masked/hidden from attention).
        This is optimized to check multiple indices in a single pass over the data.

        Args:
            indices: A list of indices to check. Does not need to be sorted.

        Returns:
            A list of booleans containing the value at each corresponding index.

        Raises:
            IndexError: If any index is out of bounds.
        """
        if not indices:
            return []

        # Pair indices with their original position for result ordering
        indexed_indices = [(i, idx) for i, idx in enumerate(indices)]
        indexed_indices.sort(key=lambda x: x[1])

        results = [False] * len(indices)
        run_iter = iter(self.iter_runs())
        current_run = next(run_iter, None)

        for original_pos, query_index in indexed_indices:
            if query_index >= self.total_size:
                raise IndexError(
                    f"Index {query_index} is out of bounds for Brle of length {self.total_size}"
                )

            # Advance through runs until we find the one containing the query_index
            while current_run is not None:
                value, run_start, run_end = current_run
                if query_index >= run_start and query_index < run_end:
                    results[original_pos] = value
                    break
                current_run = next(run_iter, None)

        return results

    def is_range_all_value(self, start: int, end: int, expected_value: bool) -> bool:
        """
        Check if all boolean values within a specified range [start, end)
        are equal to a given expected_value.

        Args:
            start: The starting index of the range (inclusive).
            end: The ending index of the range (exclusive).
            expected_value: The boolean value to check for.

        Returns:
            True if every value in the range is expected_value, False otherwise.
            Returns True for an empty range (start >= end).
        """
        if start >= end:
            return True
        if end > self.total_size:
            return False

        # Track how much of the target range we've verified
        pos_covered = start

        for run_value, run_start, run_end in self.iter_runs():
            # Find intersection of current run [run_start, run_end) and [pos_covered, end)
            intersect_start = max(run_start, pos_covered)
            intersect_end = min(run_end, end)

            if intersect_start < intersect_end:
                # There's a valid intersection - check if value matches
                if run_value != expected_value:
                    return False
                # Update coverage marker
                pos_covered = intersect_end

                # If we've covered the entire target range, we're done
                if pos_covered >= end:
                    return True

            # Optimization: if current run extends beyond target, no need to check more
            if run_end >= end:
                break

        return pos_covered >= end

    def to_array(self) -> list[bool]:
        """
        Convert to a list of booleans.

        Returns:
            List of boolean values represented by this Brle
        """
        result: list[bool] = []
        current_pos = 0

        for i, run_len in enumerate(self.buffer):
            value = i % 2 != 0  # Even = false, odd = true
            result.extend([value] * run_len)
            current_pos += run_len

        return result

    def mask(self, indices: list[int], value: bool) -> None:
        """
        Set specific indices to the given value.

        Args:
            indices: Indices to modify
            value: Value to set (True = masked, False = visible)
        """
        if not indices:
            return

        # Sort and deduplicate
        sorted_indices = sorted(set(indices))

        # Group into contiguous ranges
        ranges: list[tuple[int, int]] = []
        range_start = sorted_indices[0]
        range_end = range_start + 1

        for idx in sorted_indices[1:]:
            if idx == range_end:
                range_end = idx + 1
            else:
                ranges.append((range_start, range_end))
                range_start = idx
                range_end = idx + 1
        ranges.append((range_start, range_end))

        # Apply each range
        for start, end in ranges:
            self.mask_range(start, end, value)

    def mask_range(self, start: int, end: int, value: bool) -> None:
        """
        Set a range of indices to the given value.

        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
            value: Value to set (True = masked, False = visible)
        """
        if start >= end or start >= self.total_size:
            return

        end = min(end, self.total_size)

        # Rebuild the buffer with the range modified
        new_buffer: list[int] = []
        current_pos = 0

        for i, run_len in enumerate(self.buffer):
            is_true = i % 2 != 0
            r_start = current_pos
            r_end = current_pos + run_len
            current_pos = r_end

            # Check overlap with [start, end)
            overlap_start = max(r_start, start)
            overlap_end = min(r_end, end)

            if overlap_start >= overlap_end:
                # No overlap - keep original run
                self._append_run(new_buffer, is_true, run_len)
            else:
                # Has overlap - split into up to 3 parts
                # Part 1: before overlap
                if r_start < overlap_start:
                    self._append_run(new_buffer, is_true, overlap_start - r_start)

                # Part 2: the overlap (with new value)
                self._append_run(new_buffer, value, overlap_end - overlap_start)

                # Part 3: after overlap
                if overlap_end < r_end:
                    self._append_run(new_buffer, is_true, r_end - overlap_end)

        self.buffer = new_buffer

    @staticmethod
    def _append_run(buffer: list[int], value: bool, length: int) -> None:
        """
        Helper to append a run to a buffer, merging if possible.

        Args:
            buffer: The buffer to append to
            value: The boolean value of the run
            length: The length of the run
        """
        if length <= 0:
            return

        if not buffer:
            if value:
                buffer.extend([0, length])
            else:
                buffer.append(length)
        else:
            last_is_true = (len(buffer) - 1) % 2 != 0
            if last_is_true == value:
                buffer[-1] += length
            else:
                buffer.append(length)

    def extend(self, other: "Brle") -> None:
        """
        Extend this Brle with another Brle's contents.

        Args:
            other: Brle to append
        """
        if other.total_size == 0:
            return

        if self.total_size == 0:
            self.buffer = other.buffer.copy()
            self.total_size = other.total_size
            return

        # Check if last run of self and first run of other have same value
        self_last_is_true = (len(self.buffer) - 1) % 2 != 0
        other_first_is_true = other.buffer[0] == 0 and len(other.buffer) > 1

        if self_last_is_true == other_first_is_true:
            # Merge the runs
            first_run_len = other.buffer[1] if other_first_is_true else other.buffer[0]
            slice_start = 2 if other_first_is_true else 1

            self.buffer[-1] += first_run_len
            self.buffer.extend(other.buffer[slice_start:])
        else:
            # No merge - just concatenate
            if other_first_is_true:
                self.buffer.extend(other.buffer[1:])
            else:
                self.buffer.extend(other.buffer)

        self.total_size += other.total_size


def causal_mask(num_total_tokens: int, num_input_tokens: int) -> list[Brle]:
    """
    Create causal attention masks for the given parameters.

    Each token can only attend to tokens at positions <= its own position.
    For prefill, this means progressively longer masks covering full kv_len.

    BRLE format: [visible_count, masked_count] where:
    - visible (false) = can attend to these positions
    - masked (true) = cannot attend to these positions

    Args:
        num_total_tokens: Total number of tokens in the KV cache after processing
        num_input_tokens: Number of new input tokens being processed

    Returns:
        List of Brle masks, one per input token, each of length num_total_tokens
    """
    masks: list[Brle] = []
    offset = num_total_tokens - num_input_tokens

    for i in range(num_input_tokens):
        # Token at position (offset + i) can attend to positions 0..(offset + i) inclusive
        visible = offset + i + 1
        masked = num_total_tokens - visible

        if masked > 0:
            # [visible_count, masked_count]
            masks.append(Brle([visible, masked], num_total_tokens))
        else:
            # Last token can see everything - just [visible_count]
            masks.append(Brle([visible], num_total_tokens))

    return masks


def causal_mask_raw(num_total_tokens: int, num_input_tokens: int) -> list[list[int]]:
    """
    Create causal attention masks as raw buffers (for WIT interface).

    Args:
        num_total_tokens: Total number of tokens in the KV cache after processing
        num_input_tokens: Number of new input tokens being processed

    Returns:
        List of raw BRLE buffers (list[list[int]]) for passing to WIT
    """
    masks = causal_mask(num_total_tokens, num_input_tokens)
    return [m.buffer for m in masks]
