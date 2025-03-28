#!/usr/bin/env python3

import argparse
import asyncio
import aiohttp
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

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

async def worker(session, url, headers, data, request_times, status_counts, active):
    """Worker that continuously sends requests"""
    while active[0]:
        latency, status = await send_request(session, url, headers, data)
        request_times.append(latency)
        status_counts[status] = status_counts.get(status, 0) + 1

async def run_benchmark(concurrent, duration, url, api_key, empty_message=False):
    """Run the benchmark with the specified parameters"""
    headers = {}
    
    # Only add auth header if API key is provided and not a placeholder
    if api_key and api_key != "YOUR_API_KEY" and api_key != "YOUR_ACTUAL_API_KEY":
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Set Content-Type only for POST requests
    if not empty_message:
        headers["Content-Type"] = "application/json"
    
    if empty_message:
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
    
    request_times = []
    status_counts = {}
    throughput_data = []
    active = [True]  # Using a list to make it mutable for the workers
    
    print(f"Starting benchmark with {concurrent} concurrent requests for {duration} seconds")
    
    # Configure TCP connector with appropriate limits for high concurrency
    # Setting limit_per_host to None removes the limit on simultaneous connections to the same host
    # Setting limit to a high value (0 or None means unlimited) to allow all concurrent connections
    connector = aiohttp.TCPConnector(limit=0, limit_per_host=0, ssl=False)
    
    # Configure timeout settings
    timeout = aiohttp.ClientTimeout(total=30, connect=10, sock_connect=10, sock_read=30)
    
    # Create client session with the configured connector and timeout
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Start workers
        tasks = []
        for _ in range(concurrent):
            tasks.append(asyncio.create_task(
                worker(session, url, headers, data, request_times, status_counts, active)))
        
        # Record throughput every second
        start_time = time.time()
        last_count = 0
        
        while time.time() - start_time < duration:
            await asyncio.sleep(1)
            current_count = len(request_times)
            throughput = current_count - last_count
            elapsed = time.time() - start_time
            print(f"Time: {elapsed:.1f}s, Throughput: {throughput} req/s, Total: {current_count}")
            throughput_data.append((elapsed, throughput))
            last_count = current_count
        
        # Signal workers to stop
        active[0] = False
        
        # Wait for all tasks to be cancelled
        await asyncio.gather(*tasks, return_exceptions=True)
    
    return request_times, status_counts, throughput_data

def plot_results(request_times, status_counts, throughput_data, concurrent):
    """Plot and display benchmark results"""
    # Create a timestamp for the output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate statistics
    if not request_times:
        print("No requests completed. Cannot generate statistics.")
        return

    request_times.sort()
    
    plt.figure(figsize=(12, 6))
    
    # Plot latency CDF
    plt.subplot(1, 2, 1)
    y = np.linspace(0, 1, len(request_times))
    plt.plot(request_times, y)
    plt.title(f'Latency CDF (Concurrent Requests: {concurrent})')
    plt.xlabel('Latency (seconds)')
    plt.ylabel('CDF')
    plt.grid(True)
    
    # Plot throughput over time
    time_points, throughput_points = zip(*throughput_data)
    plt.subplot(1, 2, 2)
    plt.plot(time_points, throughput_points)
    plt.title(f'Throughput over Time (Concurrent Requests: {concurrent})')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Requests per second')
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

async def main_async(args):
    """Main async function to run the benchmark"""
    request_times, status_counts, throughput_data = await run_benchmark(
        args.concurrent, 
        args.duration, 
        args.url, 
        args.api_key,
        args.empty_message
    )
    plot_results(request_times, status_counts, throughput_data, args.concurrent)

def main():
    """Parse command line arguments and start the benchmark"""
    parser = argparse.ArgumentParser(description='Benchmark API requests')
    parser.add_argument('--concurrent', '-c', type=int, default=10,
                        help='Number of concurrent requests (default: 10)')
    parser.add_argument('--duration', '-d', type=int, default=10,
                        help='Duration of the benchmark in seconds (default: 10)')
    parser.add_argument('--url', '-u', type=str, default="http://localhost:8080/",
                        help='URL to send requests to (default: http://localhost:8080/)')
    parser.add_argument('--api-key', '-k', type=str, default="YOUR_API_KEY",
                        help='API key for authorization')
    parser.add_argument('--empty-message', '-e', action='store_true',
                        help='Send empty message payload (like curl -v), useful for connection testing')
    
    args = parser.parse_args()
    asyncio.run(main_async(args))

if __name__ == "__main__":
    main()