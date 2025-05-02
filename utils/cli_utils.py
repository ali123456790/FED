import argparse


def parse_arguments():
    """Parse command line arguments for the FIDS project."""
    parser = argparse.ArgumentParser(description="FIDS Federated Learning Runner")
    parser.add_argument("--mode", type=str, required=True, choices=["server", "client", "edge"],
                        help="Mode to run: server, client, or edge")
    parser.add_argument("--config", type=str, default="config.yaml", 
                        help="Path to configuration file")
    parser.add_argument("--client_id", type=str, help="Client ID (required for client mode)")
    parser.add_argument("--edge_id", type=str, help="Edge ID (required for edge mode)")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == "client" and not args.client_id:
        parser.error("--client_id is required when using --mode client")
    
    if args.mode == "edge" and not args.edge_id:
        parser.error("--edge_id is required when using --mode edge")
    
    return args 