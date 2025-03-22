#!/bin/bash
# Setup environment for FED project on M2 Mac

# Change to project directory (assuming the script is in the project root)
cd "$(dirname "$0")"

# Create Python virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories if not already present
mkdir -p logs metrics models data

echo "Environment setup complete." 