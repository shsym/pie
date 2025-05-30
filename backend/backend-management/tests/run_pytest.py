#!/usr/bin/env python3
"""
Pytest test runner for Symphony Management Service tests.
"""

import sys
import subprocess
import os

def run_tests():
    """Run all pytest tests."""
    print("Running Symphony Management Service Tests with pytest")
    print("=" * 60)
    
    # Change to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    # Run pytest with coverage if available
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "tests/test_management_service_pytest.py",
        "tests/test_management_cli_pytest.py",
        "tests/test_integration_pytest.py"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running tests: {e}")
        return False

def run_unit_tests_only():
    """Run only unit tests."""
    print("Running Unit Tests Only")
    print("=" * 40)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short", 
        "tests/test_management_service_pytest.py",
        "tests/test_management_cli_pytest.py"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running unit tests: {e}")
        return False

def run_integration_tests_only():
    """Run only integration tests."""
    print("Running Integration Tests Only")
    print("=" * 40)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    os.chdir(parent_dir)
    
    cmd = [
        sys.executable, "-m", "pytest",
        "-v",
        "--tb=short",
        "tests/test_integration_pytest.py"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"Error running integration tests: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Symphony Management Service tests")
    parser.add_argument("--unit-only", action="store_true", 
                       help="Run only unit tests")
    parser.add_argument("--integration-only", action="store_true",
                       help="Run only integration tests")
    
    args = parser.parse_args()
    
    if args.unit_only:
        success = run_unit_tests_only()
    elif args.integration_only:
        success = run_integration_tests_only()
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)
