#!/usr/bin/env python3

import argparse
import asyncio
import aiohttp
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import psutil  # For resource monitoring
import os
from collections import deque
from threading import Lock
import concurrent.futures  # For thread-pool based operations
import multiprocessing  # For multiprocessing support
import pickle  # For serializing data between processes

class BenchmarkMetrics:
    """Thread-safe container for benchmark metrics"""
    def __init__(self):
        self.lock = Lock()
        self.request_times = []
        self.status_counts = {}
        self.throughput_data = []
        self.resource_usage = []  # CPU, memory usage over time
    
    def add_latency(self, latency, status):
        with self.lock:
            self.request_times.append(latency)
            self.status_counts[status] = self.status_counts.get(status, 0) + 1
    
    def add_throughput(self, time_point, throughput):
        with self.lock:
            self.throughput_data.append((time_point, throughput))
    
    def add_resource_usage(self, time_point, cpu_percent, memory_percent):
        with self.lock:
            self.resource_usage.append((time_point, cpu_percent, memory_percent))
    
    def get_request_count(self):
        with self.lock:
            return len(self.request_times)
    
    def get_results(self):
        with self.lock:
            return (
                list(self.request_times), 
                dict(self.status_counts), 
                list(self.throughput_data),
                list(self.resource_usage)
            )

async def send_request(session, url, headers, data):
    """Send a single request and measure the latency"""
    start_time = time.time()
    try:
        # For curl-like mode, use GET request instead of POST
        if data is None:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                await response.text()
                return time.time() - start_time, response.status
        else:
            # Use POST for when a payload is provided
            async with session.post(url, headers=headers, json=data, timeout=aiohttp.ClientTimeout(total=10)) as response:
                await response.text()
                return time.time() - start_time, response.status
    except asyncio.TimeoutError:
        print(f"Request timeout")
        return time.time() - start_time, -2
    except aiohttp.ClientConnectionError as e:
        print(f"Connection error: {e}")
        return time.time() - start_time, -3
    except Exception as e:
        print(f"Request error: {e}")
        return time.time() - start_time, -1

async def worker(session, url, headers, data, metrics, active):
    """Worker that continuously sends requests"""
    while active[0]:
        latency, status = await send_request(session, url, headers, data)
        metrics.add_latency(latency, status)

async def run_benchmark(args):
    """Run the benchmark with the specified parameters"""
    headers = {}
    
    # Only add auth header if API key is provided and not a placeholder
    if args.api_key and args.api_key != "YOUR_API_KEY" and args.api_key != "YOUR_ACTUAL_API_KEY":
        headers["Authorization"] = f"Bearer {args.api_key}"
    
    # Set Content-Type only for POST requests
    if not args.empty_message:
        headers["Content-Type"] = "application/json"
    
    # Load payload from file if specified
    if args.payload_file and os.path.exists(args.payload_file):
        try:
            with open(args.payload_file, 'r') as f:
                data = json.load(f)
            print(f"Using custom payload from {args.payload_file}")
        except Exception as e:
            print(f"Error loading payload file: {e}")
            return None
    elif args.empty_message:
        # In curl-like mode, don't send a payload (use GET request)
        data = None
        print("Using empty message payload (curl-like mode)")
    else:
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the weather like today?"}
            ],
            "temperature": 0.7
        }
    
    metrics = BenchmarkMetrics()
    active = [True]  # Using a list to make it mutable for the workers
    
    # Track current worker count for auto-scaling
    current_workers = args.concurrent
    
    print(f"Starting benchmark with {current_workers} concurrent requests for {args.duration} seconds")
    
    # Configure TCP connector with appropriate limits for high concurrency
    connector = aiohttp.TCPConnector(
        limit=args.connection_limit,
        limit_per_host=args.connections_per_host, 
        ttl_dns_cache=args.ttl_dns_cache,
        ssl=False
    )
    
    # Configure timeout settings
    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=30)
    
    # Create client session with the configured connector and timeout
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Start workers
        tasks = []
        for _ in range(current_workers):
            tasks.append(asyncio.create_task(
                worker(session, args.url, headers, data, metrics, active)))
        
        # Record throughput and resource usage every second
        start_time = time.time()
        last_count = 0
        
        while time.time() - start_time < args.duration:
            await asyncio.sleep(1)
            current_count = metrics.get_request_count()
            throughput = current_count - last_count
            elapsed = time.time() - start_time
            
            # Get current CPU utilization for auto-scaling
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            print(f"Time: {elapsed:.1f}s, Throughput: {throughput} req/s, Total: {current_count}, "
                  f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
            
            metrics.add_throughput(elapsed, throughput)
            metrics.add_resource_usage(elapsed, cpu_percent, memory_percent)
            last_count = current_count
            
            # Auto-scale workers based on system load if enabled
            if args.auto_scale:
                if cpu_percent > args.max_cpu_percent and current_workers > 1:
                    # Scale down if CPU usage is too high
                    workers_to_remove = max(1, int(current_workers * 0.1))  # Remove 10% of workers
                    for _ in range(workers_to_remove):
                        if tasks:
                            task = tasks.pop()
                            task.cancel()
                            current_workers -= 1
                    print(f"CPU load too high ({cpu_percent:.1f}%), scaled down to {current_workers} workers")
                
                elif cpu_percent < args.max_cpu_percent * 0.7:  # Have 30% headroom before adding workers
                    # Scale up if CPU usage is low
                    workers_to_add = max(1, int(current_workers * 0.1))  # Add 10% more workers
                    for _ in range(workers_to_add):
                        task = asyncio.create_task(worker(session, args.url, headers, data, metrics, active))
                        tasks.append(task)
                        current_workers += 1
                    print(f"CPU load low ({cpu_percent:.1f}%), scaled up to {current_workers} workers")
        
        # Signal workers to stop
        active[0] = False
        
        # Wait for all tasks to complete or be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)
    
    return metrics.get_results()

def plot_results(request_times, status_counts, throughput_data, resource_usage, concurrent):
    """Plot and display benchmark results"""
    # Create a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate statistics
    if not request_times:
        print("No requests completed. Cannot generate statistics.")
        return

    request_times.sort()
    
    plt.figure(figsize=(12, 8))
    
    # Plot latency CDF
    plt.subplot(2, 2, 1)
    y = np.linspace(0, 1, len(request_times))
    plt.plot(request_times, y)
    plt.title(f'Latency CDF (Concurrent Requests: {concurrent})')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('CDF')
    plt.grid(True)
    
    # Plot throughput over time
    time_points, throughput_points = zip(*throughput_data)
    plt.subplot(2, 2, 2)
    plt.plot(time_points, throughput_points)
    plt.title(f'Throughput over Time (Concurrent Requests: {concurrent})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Requests per second')
    plt.grid(True)
    
    # Plot CPU usage over time
    resource_time_points, cpu_usage, memory_usage = zip(*resource_usage)
    plt.subplot(2, 2, 3)
    plt.plot(resource_time_points, cpu_usage, label='CPU Usage (%)')
    plt.plot(resource_time_points, memory_usage, label='Memory Usage (%)')
    plt.title('Resource Usage over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Usage (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    output_file = f"benchmark_results_{timestamp}.pdf"
    plt.savefig(output_file)
    print(f"Saved results chart to {output_file}")
    
    # Print summary statistics
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"Total requests: {len(request_times)}")
    print(f"Average throughput: {len(request_times)/max(time_points):.2f} req/s")
    print(f"Average latency: {np.mean(request_times):.4f} seconds")
    print(f"Min latency: {min(request_times):.4f} seconds")
    print(f"Max latency: {max(request_times):.4f} seconds")
    print(f"P50 latency: {np.percentile(request_times, 50):.4f} seconds")
    print(f"P95 latency: {np.percentile(request_times, 95):.4f} seconds")
    print(f"P99 latency: {np.percentile(request_times, 99):.4f} seconds")
    
    print("\nStatus Codes:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

def run_process(args, process_id):
    """Run the benchmark in a single process"""
    print(f"Process {process_id} starting")
    return asyncio.run(run_benchmark(args))

def merge_results(result_list):
    """Merge results from multiple processes"""
    all_request_times = []
    all_status_counts = {}
    all_throughput_data = []
    all_resource_usage = []
    
    for request_times, status_counts, throughput_data, resource_usage in result_list:
        all_request_times.extend(request_times)
        
        # Merge status counts
        for status, count in status_counts.items():
            all_status_counts[status] = all_status_counts.get(status, 0) + count
        
        all_throughput_data.extend(throughput_data)
        all_resource_usage.extend(resource_usage)
    
    return all_request_times, all_status_counts, all_throughput_data, all_resource_usage

def main():
    """Parse command line arguments and start the benchmark"""
    parser = argparse.ArgumentParser(description='Benchmark API requests')
    parser.add_argument('--concurrent', '-c', type=int, default=10,
                        help='Number of concurrent requests per process (default: 10)')
    parser.add_argument('--duration', '-d', type=int, default=10,
                        help='Duration of the benchmark in seconds (default: 10)')
    parser.add_argument('--url', '-u', type=str, default="http://localhost:8080/",
                        help='URL to send requests to (default: http://localhost:8080/)')
    parser.add_argument('--api-key', '-k', type=str, default="YOUR_API_KEY",
                        help='API key for authorization')
    parser.add_argument('--empty-message', '-e', action='store_true',
                        help='Send empty message payload (like curl -v), useful for connection testing')
    
    # Advanced configuration for scalability
    parser.add_argument('--connection-limit', type=int, default=0,
                        help='Maximum number of connections (0 for unlimited, default: 0)')
    parser.add_argument('--connections-per-host', type=int, default=0,
                        help='Maximum number of connections per host (0 for unlimited, default: 0)')
    parser.add_argument('--ttl-dns-cache', type=int, default=10,
                        help='Time-to-live for DNS cache in seconds (default: 10)')
    parser.add_argument('--payload-file', type=str,
                        help='Path to JSON file containing the request payload')
    parser.add_argument('--auto-scale', action='store_true',
                        help='Automatically adjust worker count based on system load')
    parser.add_argument('--max-cpu-percent', type=float, default=80.0,
                        help='Maximum CPU utilization percent before scaling down workers (default: 80%)')
    parser.add_argument('--processes', '-p', type=int, default=1,
                        help='Number of processes to use for the benchmark (default: 1)')
    
    args = parser.parse_args()
    
    if args.processes <= 1:
        # Single process mode
        request_times, status_counts, throughput_data, resource_usage = asyncio.run(run_benchmark(args))
        plot_results(request_times, status_counts, throughput_data, resource_usage, args.concurrent)
    else:
        # Multi-process mode
        print(f"Starting benchmark with {args.processes} processes, {args.concurrent} concurrent requests per process")
        
        # Create a pool of worker processes
        with multiprocessing.Pool(args.processes) as pool:
            # Prepare arguments for each process
            process_args = [(args, i) for i in range(args.processes)]
            
            # Run the benchmark in parallel
            results = pool.starmap(run_process, process_args)
            
            # Merge the results
            merged_results = merge_results(results)
            
            # Plot the results
            plot_results(*merged_results, args.concurrent * args.processes)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # For Windows support
    main()