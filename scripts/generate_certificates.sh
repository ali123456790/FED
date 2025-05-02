#!/bin/bash

# Create certificates directory
mkdir -p certificates

# Generate CA key and certificate
openssl genrsa -out certificates/ca.key 4096
openssl req -new -x509 -key certificates/ca.key -out certificates/ca.crt -days 365 -subj "/CN=FederatedLearningCA"

# Generate server key and CSR
openssl genrsa -out certificates/server.key 2048
openssl req -new -key certificates/server.key -out certificates/server.csr -subj "/CN=FederatedLearningServer"

# Sign server certificate with CA
openssl x509 -req -in certificates/server.csr -CA certificates/ca.crt -CAkey certificates/ca.key -CAcreateserial -out certificates/server.crt -days 365

# Generate client key and CSR
openssl genrsa -out certificates/client.key 2048
openssl req -new -key certificates/client.key -out certificates/client.csr -subj "/CN=FederatedLearningClient"

# Sign client certificate with CA
openssl x509 -req -in certificates/client.csr -CA certificates/ca.crt -CAkey certificates/ca.key -CAcreateserial -out certificates/client.crt -days 365

# Set proper permissions
chmod 600 certificates/*.key
chmod 644 certificates/*.crt

# Clean up CSR files
rm certificates/*.csr

echo "Certificates generated successfully in ./certificates/" 