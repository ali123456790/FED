#!/usr/bin/env python3
"""
Main test runner for FIDS (Federated Learning for IoT Device Security).
Discovers and runs all tests in the test directory.
"""

import unittest
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_all_tests():
    """Discover and run all tests in the tests directory."""
    # Initialize the test loader
    loader = unittest.TestLoader()
    
    # Discover all tests in the tests directory
    start_dir = 'tests'
    pattern = 'test_*.py'
    
    # Find and load all tests
    suite = loader.discover(start_dir, pattern=pattern)
    
    # Initialize the test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run the tests
    logger.info("Starting test execution...")
    result = runner.run(suite)
    logger.info("Test execution completed.")
    
    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
