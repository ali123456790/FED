#!/bin/bash

# Check if protoc is installed
if ! command -v protoc &> /dev/null; then
    echo "protoc not found. Please install Protocol Buffers compiler."
    exit 1
fi

# Compile protocol buffers
protoc --python_out=models/ models/model_messages.proto

echo "Protocol Buffers compiled successfully" 