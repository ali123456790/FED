#!/usr/bin/env python3
"""
Run all Partially Homomorphic Encryption (PHE) tests and generate a report.

This script runs all the PHE-related tests and generates a summary report
of test results and performance benchmarks.
"""

import os
import sys
import unittest
import time
import argparse
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'

def print_header(text):
    """Print a formatted header."""
    print(f"\n{BOLD}{YELLOW}{'='*80}{RESET}")
    print(f"{BOLD}{YELLOW}= {text}{RESET}")
    print(f"{BOLD}{YELLOW}{'='*80}{RESET}\n")

def run_test_suite(test_class, verbosity=2, enable_performance_tests=True):
    """Run a test suite and return results."""
    if not enable_performance_tests and "Performance" in test_class.__name__:
        print(f"{YELLOW}Skipping performance tests: {test_class.__name__}{RESET}")
        return None
    
    print(f"{BOLD}Running {test_class.__name__}...{RESET}")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(test_class)
    
    start_time = time.time()
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    end_time = time.time()
    
    duration = end_time - start_time
    
    # Print result summary
    if result.wasSuccessful():
        print(f"{GREEN}✓ All tests passed! ({len(result.successes)} tests, {duration:.2f}s){RESET}")
    else:
        print(f"{RED}✗ {len(result.failures)} failures, {len(result.errors)} errors{RESET}")
    
    return result, duration

def main():
    """Run all PHE-related tests."""
    parser = argparse.ArgumentParser(description="Run PHE-related tests")
    parser.add_argument('--no-performance', action='store_true', help='Skip performance tests')
    parser.add_argument('--verbosity', type=int, default=2, help='Test verbosity level (1-3)')
    args = parser.parse_args()
    
    print_header("PHE Test Suite")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Performance tests: {'Disabled' if args.no_performance else 'Enabled'}")
    
    # Import test modules
    from tests.security.test_encoding import TestFixedPointEncoding
    from tests.security.test_encryption import TestPaillierHomomorphicEncryption
    from tests.integration.test_phe_integration import TestPHEIntegration
    
    # Only import performance tests if enabled
    if not args.no_performance:
        from tests.security.test_encryption_performance import TestEncryptionPerformance
    
    # Record test results
    all_results = []
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_duration = 0
    
    # Run encoding tests
    test_result = run_test_suite(TestFixedPointEncoding, verbosity=args.verbosity)
    if test_result:
        result, duration = test_result
        all_results.append(("Fixed-Point Encoding", result, duration))
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        total_duration += duration
    
    # Run encryption tests
    test_result = run_test_suite(TestPaillierHomomorphicEncryption, verbosity=args.verbosity)
    if test_result:
        result, duration = test_result
        all_results.append(("Paillier Homomorphic Encryption", result, duration))
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        total_duration += duration
    
    # Run integration tests
    test_result = run_test_suite(TestPHEIntegration, verbosity=args.verbosity)
    if test_result:
        result, duration = test_result
        all_results.append(("PHE Integration", result, duration))
        total_tests += result.testsRun
        total_failures += len(result.failures)
        total_errors += len(result.errors)
        total_duration += duration
    
    # Run performance tests if enabled
    if not args.no_performance:
        test_result = run_test_suite(TestEncryptionPerformance, verbosity=args.verbosity)
        if test_result:
            result, duration = test_result
            all_results.append(("Encryption Performance", result, duration))
            total_tests += result.testsRun
            total_failures += len(result.failures)
            total_errors += len(result.errors)
            total_duration += duration
    
    # Print summary report
    print_header("Test Summary")
    print(f"{'Test Suite':<30} {'Status':<10} {'Tests':<10} {'Failures':<10} {'Errors':<10} {'Duration':<10}")
    print("-" * 80)
    
    for name, result, duration in all_results:
        status = f"{GREEN}PASS{RESET}" if result.wasSuccessful() else f"{RED}FAIL{RESET}"
        print(f"{name:<30} {status:<10} {result.testsRun:<10} {len(result.failures):<10} {len(result.errors):<10} {duration:.2f}s")
    
    print("-" * 80)
    overall_status = f"{GREEN}PASS{RESET}" if total_failures == 0 and total_errors == 0 else f"{RED}FAIL{RESET}"
    print(f"{'TOTAL':<30} {overall_status:<10} {total_tests:<10} {total_failures:<10} {total_errors:<10} {total_duration:.2f}s")
    
    # Generate a report file
    report_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"phe_test_report_{report_time}.txt"
    
    with open(report_file, 'w') as f:
        f.write(f"PHE Test Suite Report\n")
        f.write(f"=====================\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Performance tests: {'Disabled' if args.no_performance else 'Enabled'}\n\n")
        
        f.write(f"Test Summary\n")
        f.write(f"------------\n")
        f.write(f"{'Test Suite':<30} {'Status':<10} {'Tests':<10} {'Failures':<10} {'Errors':<10} {'Duration':<10}\n")
        f.write("-" * 80 + "\n")
        
        for name, result, duration in all_results:
            status = "PASS" if result.wasSuccessful() else "FAIL"
            f.write(f"{name:<30} {status:<10} {result.testsRun:<10} {len(result.failures):<10} {len(result.errors):<10} {duration:.2f}s\n")
        
        f.write("-" * 80 + "\n")
        overall_status = "PASS" if total_failures == 0 and total_errors == 0 else "FAIL"
        f.write(f"{'TOTAL':<30} {overall_status:<10} {total_tests:<10} {total_failures:<10} {total_errors:<10} {total_duration:.2f}s\n\n")
        
        # Write details of failures and errors
        if total_failures > 0 or total_errors > 0:
            f.write("Failure and Error Details\n")
            f.write("------------------------\n\n")
            
            for name, result, _ in all_results:
                if result.failures or result.errors:
                    f.write(f"{name}:\n")
                    f.write("-" * len(name) + "\n")
                    
                    if result.failures:
                        f.write("\nFailures:\n")
                        for test, trace in result.failures:
                            f.write(f"- {test}\n")
                            f.write(f"{trace}\n\n")
                    
                    if result.errors:
                        f.write("\nErrors:\n")
                        for test, trace in result.errors:
                            f.write(f"- {test}\n")
                            f.write(f"{trace}\n\n")
    
    print(f"\nTest report written to: {report_file}")
    return 0 if overall_status == f"{GREEN}PASS{RESET}" else 1

if __name__ == "__main__":
    sys.exit(main()) 