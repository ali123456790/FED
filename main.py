#!/usr/bin/env python3
"""
FIDS - Federated Learning for IoT Device Security
Main entry point for the application.
"""

import argparse
import yaml
import logging
import os
from utils.logging_utils import setup_logging
from utils.cli_utils import parse_arguments

# Import components based on mode
def import_components(mode):
    if mode == "server":
        from server.server import FlowerServer
        return FlowerServer
    elif mode == "client":
        from client.client import FlowerClient
        return FlowerClient
    elif mode == "edge":
        from edge.edge_aggregator import EdgeAggregator
        return EdgeAggregator
    else:
        raise ValueError(f"Unknown mode: {mode}")

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config["logging"])
    
    # Import appropriate components based on mode
    component_class = import_components(args.mode)
    
    # Initialize and run the component
    if args.mode == "server":
        server = component_class(config)
        server.start()
    elif args.mode == "client":
        client = component_class(config, client_id=args.client_id)
        client.start()
    elif args.mode == "edge":
        edge = component_class(config, edge_id=args.edge_id)
        edge.start()

if __name__ == "__main__":
    main()

