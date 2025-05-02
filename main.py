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

def setup_logging(config):
    """
    Set up logging configuration.
    
    Args:
        config: Application configuration
    """
    log_dir = "logs"
    log_file = os.path.join(log_dir, "fids.log")
    os.makedirs(log_dir, exist_ok=True)
    
    log_level = getattr(logging, config.get("logging", {}).get("level", "INFO").upper())
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Also set the level for client.wrapper logger specifically
    logging.getLogger("client.wrapper").setLevel(logging.DEBUG)
    
    # Set Flower's logger level as well (it can be quite verbose)
    logging.getLogger("flwr").setLevel(log_level)

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    
    # Import appropriate components based on mode
    component_class = import_components(args.mode)
    
    # Initialize and run the component
    if args.mode == "server":
        server = component_class(config)
        server.start()
    elif args.mode == "client":
        from client.client import FlowerClient
        from client.client import setup_logging as setup_client_logging
        
        # Additional client setup
        setup_client_logging()
        
        # Create client
        client = FlowerClient(
            config=config,
            client_id=args.client_id
        )
        client.start()
    elif args.mode == "edge":
        edge = component_class(config, edge_id=args.edge_id)
        edge.start()

if __name__ == "__main__":
    main()

