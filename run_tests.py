#!/usr/bin/env python3
"""
Script untuk menjalankan unit tests untuk Komodo Mlipir Algorithm.

Usage:
    python run_tests.py [options]
    
Options:
    --full      Run all tests including slow tests
    --unit      Run only unit tests
    --coverage  Generate coverage report
    --quick     Run quick subset of tests
"""

import sys
import subprocess
import argparse


def run_tests(args):
    """Menjalankan pytest dengan opsi yang diberikan."""
    
    # Base command
    cmd = ["pytest"]
    
    # Add common options
    cmd.extend(["-v", "--tb=short"])
    
    # Handle different test modes
    if args.full:
        # Run all tests
        print("Running all tests (including slow tests)...")
        cmd.append("test_kma.py")
    elif args.unit:
        # Run only unit tests
        print("Running unit tests only...")
        cmd.extend(["-m", "not slow and not integration"])
    elif args.quick:
        # Run quick subset
        print("Running quick test subset...")
        cmd.extend(["-k", "not Performance and not Integration"])
    else:
        # Default: run all except slow tests
        print("Running standard test suite...")
        cmd.extend(["-m", "not slow"])
    
    # Add coverage if requested
    if args.coverage:
        print("Generating coverage report...")
        cmd.extend([
            "--cov=kma",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Execute pytest
    try:
        result = subprocess.run(cmd, check=True)
        print("\n✅ All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated in htmlcov/index.html")
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Tests failed with exit code {e.returncode}")
        sys.exit(1)


def check_dependencies():
    """Memeriksa apakah dependencies terinstall."""
    required_packages = ["pytest", "pytest-cov", "numpy"]
    missing = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies:", ", ".join(missing))
        print("\nPlease install them using:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run unit tests for Komodo Mlipir Algorithm"
    )
    
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Run all tests including slow tests"
    )
    parser.add_argument(
        "--unit",
        action="store_true", 
        help="Run only unit tests"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick subset of tests"
    )
    
    args = parser.parse_args()
    
    # Check dependencies first
    if not check_dependencies():
        sys.exit(1)
    
    # Run tests
    run_tests(args)


if __name__ == "__main__":
    main()


# ============================================
# Requirements file content (requirements-test.txt)
# ============================================
"""
# Testing dependencies for Komodo Mlipir Algorithm

# Core dependencies
numpy>=1.20.0

# Testing framework
pytest>=6.0.0
pytest-cov>=2.12.0
pytest-mock>=3.6.0

# Optional performance testing
pytest-benchmark>=3.4.0

# Code quality (optional)
pylint>=2.10.0
black>=21.6b0
flake8>=3.9.0
mypy>=0.910

# Documentation (optional)
sphinx>=4.0.0
sphinx-rtd-theme>=0.5.0
"""