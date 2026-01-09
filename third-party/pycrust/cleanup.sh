#!/bin/bash
# Cleanup script for stuck benchmark processes

echo "Killing benchmark processes..."
pkill -9 -f rust_benchmark
pkill -9 -f benchmark_worker

echo "Cleaning iceoryx2 shared memory..."
rm -rf /dev/shm/iox2_*

echo "Verifying cleanup..."
ps aux | grep -E "(rust_benchmark|benchmark_worker)" | grep -v grep || echo "No benchmark processes running"
ls /dev/shm/iox2_* 2>/dev/null || echo "No iceoryx2 files in /dev/shm"

echo "Cleanup complete!"
