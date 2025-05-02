Okay, here is an updated README file reflecting the proposed **FIDS-PHE** scheme, which incorporates Paillier Partially Homomorphic Encryption for enhanced edge privacy.

```markdown
# FIDS-PHE: Federated Intrusion Detection System with Partially Homomorphic Encryption for IoT

## Table of Contents

1.  [Introduction](#introduction)
2.  [Architecture](#architecture)
3.  [Key Features (FIDS-PHE)](#key-features-fids-phe)
4.  [Codebase Structure](#codebase-structure)
5.  [Setup and Installation](#setup-and-installation)
    * [Prerequisites](#prerequisites)
    * [Installation Steps](#installation-steps)
    * [Data Preparation](#data-preparation)
6.  [Configuration](#configuration-configyaml)
7.  [Running the System](#running-the-system)
    * [Starting the Server](#starting-the-server)
    * [Starting Edge Aggregators](#starting-edge-aggregators)
    * [Starting Clients](#starting-clients)
8.  [Running Tests](#running-tests)
9.  [Security Considerations](#security-considerations)
10. [License](#license)
11. [Acknowledgments](#acknowledgments)

---

## Introduction

The Internet of Things (IoT) presents significant security challenges due to the vast number of interconnected, often resource-constrained devices and the sensitive nature of the data they handle. Traditional centralized Intrusion Detection Systems (IDS) struggle with scalability and raise privacy concerns. Federated Learning (FL) offers a decentralized alternative but faces its own hurdles in heterogeneous IoT, including device variability and security vulnerabilities.

**FIDS (Federated Intrusion Detection System)** was designed as a hierarchical FL framework for IoT security. This repository implements **FIDS-PHE**, an enhanced version specifically designed to address privacy concerns related to potentially untrusted Edge Nodes. FIDS-PHE replaces standard payload encryption with **Partially Homomorphic Encryption (Paillier)**, enabling edges to aggregate encrypted model updates without decrypting them, thus significantly enhancing privacy against intermediate aggregators.

The framework aims to provide effective intrusion detection tailored for heterogeneous IoT environments while preserving user privacy (even from edge nodes), ensuring data integrity, and mitigating common FL vulnerabilities like data poisoning.

---

## Architecture

FIDS-PHE retains the three-tier hierarchical architecture, orchestrated using the [Flower](https://flower.dev/) Federated Learning framework:

1.  **IoT Clients (`client/`)**:
    * End-point IoT devices holding local data.
    * Perform local model training.
    * Employ a `DeviceManager` for resource awareness and potential adaptive training.
    * **Encode** model updates (floats to integers) and **encrypt** them using the server's public Paillier key ($pk_S$).
    * Apply optional Differential Privacy noise *before* encoding/encryption.
    * Compute HMAC over the encrypted payload for integrity.
    * Communicate securely (TLS) with their assigned Edge Node.

2.  **Edge Nodes (`edge/`)**:
    * Intermediate aggregators managing client clusters.
    * Receive encrypted updates from clients.
    * Verify client HMAC tags using shared Client-Edge keys ($K_{CE}$).
    * **Perform Homomorphic Aggregation:** Sum the corresponding encrypted updates directly on the Paillier ciphertexts using only the server's public key ($pk_S$). **Crucially, the Edge cannot decrypt individual or aggregated updates.**
    * Compute HMAC over the aggregated encrypted payload using shared Edge-Server keys ($K_{ES}$).
    * Forward the *encrypted aggregate* to the Central Server via TLS.
    * Implemented in `edge/edge_aggregator.py`.

3.  **Central Server (`server/`)**:
    * Orchestrates the FL process.
    * **Generates and manages the Paillier keypair ($pk_S, sk_S$)**. Securely distributes $pk_S$ to Edges/Clients. Holds $sk_S$ privately.
    * Distributes the global model.
    * Receives encrypted aggregates from Edge Nodes.
    * Verifies edge HMAC tags using shared Edge-Server keys ($K_{ES}$).
    * **Performs Decryption:** Uses its private key ($sk_S$) to decrypt the final aggregated ciphertexts.
    * **Decodes** the resulting integer sums back to floats.
    * **Performs Robust Aggregation:** Applies strategies (e.g., FedAvg, Trimmed Mean, Median) to the *decrypted and averaged* updates to mitigate poisoning.
    * Updates the global model.
    * Implemented in `server/server.py` using a Flower `Strategy` (defined in `server/aggregation.py`).

---

## Key Features (FIDS-PHE)

* **Federated Learning Core**: Utilizes Flower (`flwr`) for efficient and scalable federated learning orchestration. Keeps raw data localized on client devices.
* **Hierarchical Architecture**: Three tiers (Client, Edge, Server) improve scalability and manage communication flow.
* **Multi-Layer Security Stack (Enhanced for Edge Privacy)**:
    * **Transport Layer Security (TLS 1.2+):** Secures all communication channels (Client-Edge, Edge-Server) using certificates (`scripts/generate_certificates.sh`) for confidentiality and integrity of data in transit. Mutual authentication via certificates.
    * **Payload Encryption (Paillier PHE):**
        * Clients encode float updates to integers (fixed-point encoding) and encrypt them using the server's public Paillier key ($pk_S$).
        * Key management is **server-centric**: only the server holds the decryption key ($sk_S$).
        * *Replaces the previous RSA-AES scheme.* Implementation likely in `security/encryption.py` and integrated into client/server logic.
    * **Homomorphic Edge Aggregation:**
        * Edges sum encrypted client updates directly using the additive property of Paillier and $pk_S$.
        * **Provides strong privacy guarantees against edge nodes**, which cannot access individual or aggregated model updates in cleartext.
        * Implemented within `edge/edge_aggregator.py`.
    * **Message Authentication (HMAC-SHA256):**
        * Ensures integrity and origin authenticity between hops.
        * Applied *after* PHE encryption/aggregation to the serialized message containing ciphertexts.
        * Uses separate shared keys for Client-Edge ($K_{CE}$) and Edge-Server ($K_{ES}$) links (requires secure distribution).
        * Integrated into serialization (`models/proto_serialization.py`) and verification steps at edge/server. (Configurable via `config.yaml -> security.authentication`).
    * **Differential Privacy (DP - Optional Layer):**
        * Client-side noise injection (Gaussian, Laplace via `tensorflow-privacy`) can be applied to updates *before* encoding and PHE encryption.
        * Provides formal $\epsilon$-differential privacy against inference attacks, even by the server observing aggregated results.
        * Implemented in `security/differential_privacy.py`. (Configurable via `config.yaml -> security.differential_privacy`).
    * **Robust Aggregation (Server-Side):**
        * Mitigates poisoning attacks by applying strategies (FedAvg, Trimmed Mean, Median) *after* the server decrypts the aggregated updates.
        * Implemented in `security/secure_aggregation.py` and used by the server's Flower `Strategy` (`server/aggregation.py`). (Configurable via `config.yaml -> server.strategy`).
* **Heterogeneity Management**: The `client/device_manager.py` profiles clients and allows adaptive training configurations to handle diverse IoT device capabilities. (Configurable via `config.yaml -> client.device_heterogeneity`).
* **Flexible Model Support**: `models/model_factory.py` supports various ML models (LSTM, CNN, RF, etc. via `models/deep_learning.py`, `models/traditional.py`).
* **Fixed-Point Encoding**: Necessary step before Paillier encryption. Float updates are converted to integers with configurable precision, potentially impacting model accuracy. Utilities likely in `security/encoding.py` (new file).
* **Protocol Buffers**: Uses Protobuf (`models/model_messages.proto`) for structured communication, adapted to carry lists of Paillier ciphertexts.

---

## Codebase Structure

```
FED-main/
├── client/               # Client-side logic (incl. DeviceManager, PHE encryption)
├── config/               # Example configuration files
├── data/                 # Data loading, preprocessing, distribution
├── device_profiles/      # Stores generated client hardware profiles (JSON)
├── edge/                 # Edge node logic (incl. Homomorphic Aggregation)
├── logs/                 # Default directory for log files
├── metrics/              # Stores evaluation/resource metrics
├── models/               # ML models, Protobuf definitions, Serialization
├── scripts/              # Utility scripts (certs, protos, setup)
├── security/             # Security components implementations
│   ├── encryption.py       # Paillier PHE implementation (replaces RSA-AES)
│   ├── encoding.py       # NEW: Fixed-point encoding utilities
│   ├── differential_privacy.py # DP noise addition
│   ├── secure_aggregation.py # Robust aggregation algorithms
│   └── ...                 # Other security modules (HMAC logic likely integrated elsewhere)
├── server/               # Server-side logic (PHE key gen, decryption, robust agg.)
├── tests/                # Unit and integration tests
├── utils/                # General utilities (logging, CLI parsing)
├── main.py               # Main entry point
├── config.yaml           # Main configuration file
├── requirements.txt      # Project dependencies (includes 'phe')
└── README.md             # This file
```
*(Note: `security/encoding.py` is a new proposed file for the PHE implementation)*

---

## Setup and Installation

### Prerequisites

* **Python:** 3.8+
* **pip:** Package installer.
* **OpenSSL:** For generating certificates.
* **(Optional) Environment Management:** `pyenv`, `virtualenv`, `conda`.
* **Paillier Library:** The `phe` library will be installed via `requirements.txt`.

### Installation Steps

1.  **Clone:** `git clone <repository-url>; cd FED-main`
2.  **Environment:** Set up and activate a Python virtual environment (recommended).
3.  **Install Dependencies:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt # Ensure 'phe>=1.5.0' is listed here
    ```
4.  **Generate Certificates:** `bash scripts/generate_certificates.sh`
5.  **Compile Protocol Buffers:** `bash scripts/compile_protos.sh` (Ensure `.proto` files are updated for Paillier ciphertexts - see FIDS-PHE implementation steps).

### Data Preparation

* (Same as previous README: Download N-BaIoT/NSL-KDD, configure paths in `config.yaml`, refer to `data/` loaders).

---

## Configuration (`config.yaml`)

The `config.yaml` file is central to configuring FIDS-PHE.

Key sections to update/verify for FIDS-PHE:

* **`security`**:
    * **`encryption`**:
        * `enabled: true`
        * `type: 'paillier'` (Set this value)
        * `paillier:`
            * `key_size: 2048` (Or 3072 - affects performance/security)
            * `encoding_precision_bits: 64` (Crucial parameter affecting accuracy/range)
    * **`authentication`**:
        * `enabled: true`
        * Consider how shared HMAC keys ($K_{CE}$, $K_{ES}$) will be managed and configured (e.g., paths to key files per link). The single `hmac_key` path might need revision.
    * **`differential_privacy`**: Configure as needed (`enabled`, `noise_multiplier`, etc.).
* **`server`**: Ensure robust aggregation strategy (`TrimmedMean`, `Median`) is selected if needed, as this is the primary defense against poisoning post-decryption. Configure certificate paths.
* **`client` / `edge`**: Configure certificate paths.
* **Other sections** (`model`, `data`, `evaluation`, `logging`) remain conceptually similar.

---

## Running the System

Launching remains the same via `main.py`.

### Starting the Server

```bash
python main.py --mode server --config config.yaml
```
* Server generates Paillier keys, starts listening, distributes $pk_S$ upon secure connection (implementation detail), and waits for edges/clients.

### Starting Edge Aggregators

```bash
python main.py --mode edge --edge_id <unique_edge_id> --config config.yaml
```
* Edge connects to server, receives $pk_S$, waits for clients, performs homomorphic aggregation.

### Starting Clients

```bash
python main.py --mode client --client_id <unique_client_id> --config config.yaml
```
* Client connects to edge/server, receives $pk_S$, trains, encodes, encrypts updates with $pk_S$, adds HMAC.

---

## Running Tests

* (Same as previous README: Use `python tests/run_all_tests.py` or `pytest tests/`). Tests should be updated/added to cover Paillier encryption, homomorphic aggregation, encoding/decoding, and key handling.

---

## Security Considerations

* **Server Trust:** The Central Server is fully trusted as it holds the Paillier secret key ($sk_S$) required for decryption. Compromise of the server compromises all encrypted data.
* **Edge Privacy:** FIDS-PHE significantly enhances privacy against Edge nodes, as they only operate on ciphertexts and cannot learn about individual or aggregated client updates.
* **Key Management:** Secure generation, storage ($sk_S$), and distribution ($pk_S$, HMAC keys $K_{CE}, K_{ES}$) are paramount. TLS protects distribution channels, but endpoint security is vital.
* **HMAC Integrity:** Ensures that ciphertexts are not tampered with between hops, even if their content isn't known. Separate Client-Edge and Edge-Server keys prevent trivial replay across tiers.
* **Encoding Precision:** The fixed-point encoding precision (`encoding_precision_bits`) is a trade-off. Too low precision loses model accuracy; too high might lead to overflow issues with Paillier, depending on the update magnitudes and number of additions.
* **Computational Overhead:** Paillier encryption and homomorphic addition are significantly more computationally expensive than symmetric encryption (like AES) or simple averaging. This impacts client training time, edge aggregation time, and server decryption time, especially with larger key sizes or models.
* **Communication Overhead:** Paillier ciphertexts are much larger than the original float values or AES-encrypted blobs, increasing communication costs.
* **Robustness:** Primary defense against poisoning relies on server-side robust aggregation *after* decryption. Edge nodes cannot inspect encrypted updates for malicious content.

---

## License

(Specify your project's license here)

---

## Acknowledgments

* This framework utilizes the [Flower (flwr)](https://flower.dev/) library.
* Incorporates the Paillier Partially Homomorphic Encryption scheme (e.g., using the [python-paillier `phe`](https://github.com/data61/python-paillier) library).
* Dataset(s) used: N-BaIoT, NSL-KDD (Please cite original sources).
* Built using libraries like TensorFlow, Scikit-learn, Cryptography, Pandas, NumPy.
* Based on the research paper: "Federated Learning-Based Privacy-Preserving Intrusion Detection for IoT Systems" (Include citation or link to `CE_450_G4.pdf`), enhanced with PHE as proposed in FIDS-PHE.

