#!/bin/bash

if [ -n "$VIRTUAL_ENV" ]; then
    echo "Please deactivate your current virtual environment by running 'deactivate' before running this script."
    exit 1
fi

# Initialize pyenv
if command -v pyenv >/dev/null; then
    eval "$(pyenv init --path)"
    eval "$(pyenv init -)"
else
    echo "pyenv is not installed. Please install pyenv via Homebrew: brew install pyenv"
    exit 1
fi

# Check for pyenv-virtualenv
if ! pyenv help virtualenv &> /dev/null; then
    echo "pyenv-virtualenv not found. Installing pyenv-virtualenv via Homebrew..."
    brew install pyenv-virtualenv
    # Reinitialize pyenv-virtualenv
    eval "$(pyenv virtualenv-init -)"
fi

# Setup script for FED project using pyenv with Python 3.8, tensorflow-macos, and tensorflow-metal

# Desired Python version
PYTHON_VERSION="3.8.13"

# Check if Python ${PYTHON_VERSION} is installed; if not, install it
if ! pyenv versions --bare | grep -q "^${PYTHON_VERSION}$"; then
    echo "Installing Python ${PYTHON_VERSION} via pyenv..."
    pyenv install ${PYTHON_VERSION}
fi

# Desired virtual environment name
VENV_NAME="fed_env"

# Check if virtualenv already exists
if ! pyenv virtualenvs --bare | grep -q "^${VENV_NAME}$"; then
    echo "Creating virtual environment ${VENV_NAME} using Python ${PYTHON_VERSION}..."
    pyenv virtualenv ${PYTHON_VERSION} ${VENV_NAME}
fi

# Set local pyenv version to the virtual environment and activate it
pyenv local ${VENV_NAME}
pyenv activate ${VENV_NAME}

# Verify the Python version
echo "Using Python version:" 
python --version

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Modify requirements: remove tensorflow requirement (lines containing 'tensorflow')
grep -v 'tensorflow' requirements.txt > requirements_no_tensorflow.txt

# Install dependencies
echo "Installing dependencies from requirements_no_tensorflow.txt..."
pip install -r requirements_no_tensorflow.txt

# Install Apple-optimized TensorFlow packages
echo "Installing tensorflow-macos and tensorflow-metal..."
pip install tensorflow-macos==2.12.0 tensorflow-metal==0.6.0

# Create necessary directories
mkdir -p logs metrics models data

echo "Environment setup complete using Python ${PYTHON_VERSION} and virtualenv ${VENV_NAME}." 