import argparse


def parse_arguments():
    """Parse command line arguments for the FIDS project."""
    parser = argparse.ArgumentParser(description="FIDS Federated Learning Runner")
    parser.add_argument("--mode", type=str, default="server", help="Mode to run: server, client, or edge")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--client_id", type=str, default="client_1", help="Client ID (if mode is client)")
    parser.add_argument("--edge_id", type=str, default="edge_1", help="Edge ID (if mode is edge)")
    return parser.parse_args() 