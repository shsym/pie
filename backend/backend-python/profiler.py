from __future__ import annotations

from collections import defaultdict
from contextlib import ContextDecorator
from dataclasses import dataclass, field
import numpy as np
import torch


class _TorchProfiler:
    """The internal profiler class. Users should interact with the global API."""

    @dataclass
    class Node:
        name: str
        parent: _TorchProfiler.Node | None
        children: list[_TorchProfiler.Node] = field(default_factory=list)
        times: list[float] = field(
            default_factory=list
        )  # Raw timings for this node (self time)
        # Metrics calculated from timings
        count: int = 0
        mean: float = 0.0
        std: float = 0.0
        total_mean: float = 0.0  # Includes children's total_mean

    def __init__(self):
        self._node_map: dict[str, _TorchProfiler.Node] = {}
        self.root = self.Node(name="root", parent=None)
        self.active_node = self.root

    def _get_full_path(self, name: str) -> str:
        if self.active_node is self.root:
            return name
        return f"{self.active_node.name}.{name}"

    def start(self, name: str) -> ContextDecorator:
        """Creates a new profiling scope context manager."""
        full_path = self._get_full_path(name)
        if full_path not in self._node_map:
            new_node = self.Node(name=full_path, parent=self.active_node)
            self.active_node.children.append(new_node)
            self._node_map[full_path] = new_node

        return self.Timer(self, self._node_map[full_path])

    class Timer(ContextDecorator):
        def __init__(self, profiler: _TorchProfiler, node: _TorchProfiler.Node):
            self.profiler = profiler
            self.node = node
            self.start_event = None

        def __enter__(self):
            self.profiler.active_node = self.node
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
            return self

        def __exit__(self, *exc):
            stop_event = torch.cuda.Event(enable_timing=True)
            stop_event.record()
            torch.cuda.synchronize()
            elapsed_ms = self.start_event.elapsed_time(stop_event)
            self.node.times.append(elapsed_ms)

            if self.node.parent:
                self.profiler.active_node = self.node.parent
            return False

    def report(self):
        """Calculates statistics and prints the hierarchical report."""
        if not self.root.children:
            print("No profiling data collected.")
            return

        for node in self._node_map.values():
            if node.times:
                node.count = len(node.times)
                node.mean = np.mean(node.times)
                node.std = np.std(node.times)

        def calculate_total(node: _TorchProfiler.Node) -> float:
            children_total = sum(calculate_total(child) for child in node.children)
            node.total_mean = node.mean + children_total
            return node.total_mean

        calculate_total(self.root)

        print("\n--- 🌲 Performance Report 🌲 ---")
        print(
            f"{'Operation':<50} {'Avg Latency (ms)':<20} {'% of Parent':<15} {'Std Dev (ms)':<20} {'Samples':<10}"
        )
        print("-" * 115)

        def print_node(
            node: _TorchProfiler.Node, indent: str = "", is_last: bool = True
        ):
            if not node.name.startswith("root"):
                name_part = node.name.split(".")[-1]
                line_char = "└── " if is_last else "├── "
                parent_total = (
                    node.parent.total_mean if node.parent else node.total_mean
                )
                percent_str = (
                    f"{(node.total_mean / parent_total * 100):.1f}%"
                    if parent_total > 1e-6
                    else ""
                )

                print(
                    f"{indent}{line_char}{name_part:<{46 - len(indent)}}"
                    f"{node.total_mean:<20.4f}"
                    f"{percent_str:<15}"
                    f"{node.std:<20.4f}"
                    f"{node.count:<10}"
                )
                child_indent = indent + ("    " if is_last else "│   ")
                for i, child in enumerate(node.children):
                    print_node(child, child_indent, i == len(node.children) - 1)

        for i, child in enumerate(self.root.children):
            print_node(child, is_last=i == len(self.root.children) - 1)

        print("-" * 115)

    def reset(self):
        """Clears all collected data for a fresh run."""
        self._node_map.clear()
        self.root = self.Node(name="root", parent=None)
        self.active_node = self.root


# --- GLOBAL PROFILER API ---
PROFILER = _TorchProfiler()


def start_profile(name: str) -> ContextDecorator:
    """
    Starts a new profiling scope. Use as a context manager.
    Example:
        with start_profile("my_operation"):
            ...
    """
    return PROFILER.start(name)


def report_profiling_results():
    """Calculates and prints the final hierarchical report."""
    PROFILER.report()


def reset_profiler():
    """Resets all profiling data."""
    PROFILER.reset()
