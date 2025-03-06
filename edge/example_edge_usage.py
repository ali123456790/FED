#!/usr/bin/env python3
"""
Example script to demonstrate usage of the Edge Module.

This script shows how to initialize and use the Edge Module components
in a federated learning system for IoT device security.
"""

import os
import time
import logging
import yaml
import argparse
from edge.edge_aggregator import EdgeAggregator
from edge.utils import NetworkTopology

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Edge Module Example")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config.yaml", 
        help="Path to configuration file"
    )
    parser.add_argument(
        "--edge-id", 
        type=str, 
        default="edge_0", 
        help="Edge node identifier"
    )
    parser.add_argument(
        "--blocking", 
        action="store_true", 
        help="Run in blocking mode"
    )
    parser.add_argument(
        "--num-clients", 
        type=int, 
        default=20, 
        help="Number of clients in synthetic topology"
    )
    parser.add_argument(
        "--num-edges", 
        type=int, 
        default=3, 
        help="Number of edge nodes in synthetic topology"
    )
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        # Create default config if not exists
        config = {
            "server": {
                "address": "localhost",
                "port": 8080,
                "rounds": 10,
                "min_clients": 2,
                "min_available_clients": 2,
                "timeout": 60,
                "aggregation_strategy": "fedavg"
            },
            "edge": {
                "enabled": True,
                "nodes": 3,
                "aggregation_strategy": "weighted_average",
                "client_assignment": "proximity",
                "base_port": 8090,
                "forward_frequency": 1
            },
            "security": {
                "encryption": {
                    "enabled": False,
                    "type": "tls"
                }
            }
        }
        
        # Create directory if not exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Save default config
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        
        return config
    else:
        # Load config from file
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

def main():
    """Main function to demonstrate Edge Module usage."""
    # Parse command line arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config["edge"]["nodes"] = args.num_edges
    config["num_clients"] = args.num_clients
    
    logger.info(f"Starting Edge Module example with edge_id={args.edge_id}")
    
    # Generate synthetic network topology
    logger.info("Generating synthetic network topology...")
    topology = NetworkTopology(args.edge_id)
    topology.generate_synthetic_topology(
        num_clients=args.num_clients,
        num_edges=args.num_edges
    )
    
    # Create edge aggregator
    logger.info("Creating edge aggregator...")
    edge = EdgeAggregator(
        config=config,
        edge_id=args.edge_id,
        certificates_dir=None  # No TLS in this example
    )
    
    # Start edge server
    logger.info(f"Starting edge server (blocking={args.blocking})...")
    if not args.blocking:
        # Start in non-blocking mode
        edge.start(blocking=False)
        
        # Wait for some time to simulate running
        logger.info("Edge server running in background. Press Ctrl+C to stop.")
        try:
            while True:
                # Get client status every 5 seconds
                client_status = edge.get_client_status()
                logger.info(f"Client status: {len(client_status)} clients")
                
                # Get aggregation stats
                agg_stats = edge.get_aggregation_stats()
                if agg_stats:
                    logger.info(f"Latest aggregation stats: Round {agg_stats[-1]['round']} with {agg_stats[-1]['num_clients']} clients")
                
                # Sleep for a bit
                time.sleep(5)
        except KeyboardInterrupt:
            logger.info("Stopping edge server...")
            edge.stop()
            logger.info("Edge server stopped.")
    else:
        # Start in blocking mode
        edge.start(blocking=True)
    
    logger.info("Edge Module example completed")

if __name__ == "__main__":
    main()