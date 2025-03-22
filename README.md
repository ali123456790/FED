# FIDS - Federated Learning for IoT Device Security

## Overview
This project implements a federated learning framework for IoT device security, focusing on anomaly detection using the N-BaIoT dataset. The system supports both traditional machine learning and deep learning models, with privacy-preserving mechanisms, secure aggregation protocols, and edge computing capabilities.

## Features
- Federated learning server and client implementation using Flower
- Support for heterogeneous IoT devices
- Edge/fog node aggregation for improved efficiency
- Multiple ML models (Random Forest, Naive Bayes, LSTM, Bi-LSTM)
- Privacy-preserving mechanisms (differential privacy)
- Secure aggregation protocols
- Automatic and robust model initialization on the client side
- Comprehensive evaluation metrics and detailed logging

## Data Organization
All data is organized in the following structure:
- `data/raw/` - Raw datasets and original source files
- `data/processed/` - Processed and preprocessed data ready for training
- `metrics/` - Performance metrics from federated learning rounds
- `logs/` - Log files for debugging and monitoring

## Installation

### Environment Setup (macOS - M1/M2)
For macOS users (including M1/M2 Apple Silicon), follow these instructions:
1. Install Homebrew if not already installed:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. Install pyenv and pyenv-virtualenv:
   ```
   brew install pyenv
   brew install pyenv-virtualenv
   ```
3. Make the provided setup script executable and run it:
   ```
   chmod +x setup_pyenv.sh
   ./setup_pyenv.sh
   ```
This script installs Python 3.8.13, creates a virtual environment, installs all dependencies from requirements.txt (excluding TensorFlow), and installs Apple's optimized TensorFlow packages (tensorflow-macos and tensorflow-metal).

### Dependencies
The project uses two requirements files:
- `requirements.txt` - Standard dependencies for all platforms
- `requirements_no_tensorflow.txt` - Dependencies without TensorFlow (for M1/M2 Macs where TensorFlow is installed separately)

Ensure the following packages (and compatible versions) are installed:
```
flwr>=1.0.0
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0 (or tensorflow-macos/tensorflow-metal for M1/M2)
tensorflow-privacy>=0.7.0 (optional)
torch>=1.10.0
pyyaml>=6.0
matplotlib>=3.5.0
seaborn>=0.11.0
cryptography>=37.0.0
```

## Running the System

The system is driven by the `main.py` entry point. Ensure your configuration file (`config.yaml`) matches your deployment environment.

### Starting the Server
Run the following command to start the federated learning server:
```
python main.py --mode server
```

### Starting the Clients
Clients now use the updated Flower API. In the client code, we wrap the client logic using `FlowerClientWrapper` to work with `flwr.client.start_client()`. To start a client, run:
```
python main.py --mode client --client_id <your_client_id>
```
Make sure your client_id ends with a numeric value (e.g., client1, client2) to set a reproducible random seed.

## Monitoring and Detailed Analysis

### Logging
The system is extensively instrumented with logging. The log messages include:
- Server initialization details (model creation, aggregation strategy, secure aggregation status).
- Client startup messages (device information, data loading, model creation, and model building events).
- Detailed messages from Flower indicating when initial parameters are requested, evaluations are performed, and training rounds start and complete.

### What to Look For
- **Server Logs:** Verify messages indicating the creation of the global model, the starting of the server in insecure mode (if TLS is not configured), and messages about the beginning of FL rounds. For example, logs should show:
  - `Starting FIDS Flower server...`
  - `Requesting initial parameters from one random client`
- **Client Logs:** Ensure the client logs show that the data is loaded from cache or preprocessed, the model is created, and if the underlying Keras model is not built, it is automatically built using the training data. The logs will also indicate when the client connects to the server and any warnings (e.g., about missing certificates) you might need to address.

### Analysis
- **Robust Model Initialization:** The client automatically checks if the model is built and, if not, infers the input shape and number of classes from the training data to build and compile the model. This ensures that the first call to get model parameters does not fail.
- **Updated Client Launch:** By using `start_client` instead of the deprecated `start_numpy_client`, compatibility with future Flower versions is ensured. The client is now wrapped in `FlowerClientWrapper` and converted using `.to_client()` as recommended.
- **System Behavior:** Logs indicate that the server successfully creates and distributes the global model. Clients log messages regarding data loading, model building, and compilation. Any errors or connection issues should be reflected in the logs, allowing for real-time debugging.

## Deployment Considerations
- Make sure that the configuration in `config.yaml` correctly reflects your deployment environment (e.g., server address, port, encryption settings).
- For production, consider setting up proper TLS certificates and updating the paths in the configuration.
- Monitor system resources on clients to ensure they have sufficient capacity to train the models.

## Conclusion
The system is now robust and deployable, with automatic model initialization on the client side, updated Flower client API usage, and detailed logging to monitor and analyze performance during federated learning rounds.

## Current Status and Implementation Plan

The FIDS project is under active development. While the core architecture is in place, several components are still being implemented or enhanced:

### Completed Components
- Core server and client architecture with Flower integration
- Basic model implementations (traditional ML and deep learning)
- Data handling and preprocessing pipeline
- Configuration system and logging infrastructure
- Edge node aggregation framework

### In Progress
- Differential privacy implementation (partially implemented, needs completion)
- Secure aggregation protocols (framework in place, implementation ongoing)
- Client-side device heterogeneity handling (basic implementation in place)
- Comprehensive testing (test structure created, tests being implemented)
- Data augmentation for IoT security (not yet implemented)

### Team Work Division
The implementation work has been divided between team members to ensure efficient development. For details on work assignments, progress tracking, and implementation priorities, please refer to the [research.md](research.md) file.

## Contributing
Contributions are welcome! Please follow the implementation guidelines in the research.md file and ensure all new code has corresponding tests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

