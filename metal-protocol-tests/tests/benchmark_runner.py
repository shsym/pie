#!/usr/bin/env python3

import subprocess
import sys
import re
import time
from datetime import datetime

class BenchmarkRunner:
    def __init__(self):
        self.log_file = "attn_bench.log"
        self.errors = []
        self.current_test = 0
        self.total_tests = 7  # Number of benchmark configurations
        self.test_configs = []
        
    def log_to_file(self, message):
        """Log message to file with timestamp"""
        with open(self.log_file, "a") as f:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            f.write(f"[{timestamp}] {message}\n")
            f.flush()
    
    def show_progress(self, operation=""):
        """Display progress bar"""
        if self.total_tests == 0:
            progress = 0.0
        else:
            progress = self.current_test / self.total_tests
        
        width = 50
        completed = int(width * progress)
        
        bar = "█" * completed + "▌" if completed < width else "█" * width
        bar = bar.ljust(width)
        
        percentage = progress * 100
        
        # Clear line and show progress
        sys.stdout.write(f"\r[{bar}] {percentage:5.1f}%")
        if operation:
            sys.stdout.write(f" - {operation}")
        sys.stdout.flush()
    
    def parse_line(self, line):
        """Parse output line and update progress"""
        # Log everything to file
        self.log_to_file(line)
        
        # Check for test start
        if "🔄 Benchmarking:" in line:
            self.current_test += 1
            config = line.replace("🔄 Benchmarking:", "").strip()
            self.test_configs.append(config)
            self.show_progress(f"Testing: {config}")
            
        # Check for kernel testing phases
        elif "📋 Testing baseline kernel" in line:
            self.show_progress("Running baseline kernel...")
        elif "⚡ Testing simdgroup kernel" in line:
            self.show_progress("Running SIMD kernel...")
            
        # Check for errors
        elif any(keyword in line for keyword in ["FAILED", "ERROR", "❌", "Metal command buffer failed"]):
            error_msg = line.strip()
            if error_msg not in self.errors:  # Avoid duplicates
                self.errors.append(error_msg)
                self.log_to_file(f"ERROR DETECTED: {error_msg}")
    
    def run_benchmark(self):
        """Run the benchmark with progress tracking"""
        print("🏁 Metal FlashAttention Performance Benchmark")
        print("=" * 50)
        print()
        print(f"📝 Detailed logs will be saved to: {self.log_file}")
        print()
        
        # Clear log file
        with open(self.log_file, "w") as f:
            f.write(f"=== Metal FlashAttention Performance Benchmark Started at {datetime.now()} ===\n")
        
        # Start benchmark process
        try:
            cmd = ["./bin/test_attention_performance_benchmark"]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            print("🚀 Starting benchmark...")
            self.show_progress("Initializing...")
            
            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                self.parse_line(line.rstrip())
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Final progress
            self.current_test = self.total_tests
            self.show_progress("Completed!")
            print("\n")
            
            return return_code
            
        except FileNotFoundError:
            print("❌ Error: ./bin/test_attention_performance_benchmark not found")
            print("Please make sure you're in the tests directory and the binary is built.")
            return 1
        except KeyboardInterrupt:
            print("\n⚠️  Benchmark interrupted by user")
            if 'process' in locals():
                process.terminate()
            return 130
    
    def show_summary(self):
        """Show benchmark summary"""
        print()
        print("📊 BENCHMARK SUMMARY:")
        print("=" * 30)
        
        try:
            with open(self.log_file, "r") as f:
                content = f.read()
            
            # Count results
            good_count = len(re.findall(r"✅ GOOD", content))
            modest_count = len(re.findall(r"⚠️ MODEST", content))
            slow_count = len(re.findall(r"❌ SLOWER", content))
            neutral_count = len(re.findall(r"➖ NEUTRAL", content))
            failed_count = len(re.findall(r"FAILED", content))
            
            print(f"✅ Good speedups: {good_count}")
            print(f"⚠️  Modest speedups: {modest_count}")
            print(f"❌ Slower results: {slow_count}")
            print(f"➖ Neutral results: {neutral_count}")
            print(f"💥 Failed tests: {failed_count}")
            
            # Extract speedups
            print("\n🚀 SPEEDUP DETAILS:")
            speedup_matches = re.findall(r"🚀 Speedup: ([\d.]+)x ([✅⚠️❌➖] \w+)", content)
            for i, (speedup, status) in enumerate(speedup_matches):
                config = self.test_configs[i] if i < len(self.test_configs) else f"Test {i+1}"
                print(f"   {config}: {speedup}x {status}")
                
        except Exception as e:
            print(f"⚠️  Could not parse summary from log file: {e}")
        
        # Show errors
        if self.errors:
            print(f"\n⚠️  ERRORS ENCOUNTERED ({len(self.errors)}):")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"   ❌ {error}")
            if len(self.errors) > 5:
                print(f"   ... and {len(self.errors) - 5} more (see {self.log_file})")
        
        print(f"\n📋 Full details available in: {self.log_file}")

def main():
    runner = BenchmarkRunner()
    
    try:
        return_code = runner.run_benchmark()
        runner.show_summary()
        return return_code
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())