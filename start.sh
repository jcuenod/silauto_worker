#!/bin/bash

# SILAUTO Worker Startup Script

set -e

# Check if SILAUTO_URL is set
if [ -z "$SILAUTO_URL" ]; then
    echo "Error: SILAUTO_URL environment variable is not set"
    echo "Please set SILAUTO_URL or create a .env file"
    exit 1
fi

echo "Starting SILAUTO Worker..."
echo "API URL: $SILAUTO_URL"

# Check if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,utilization.gpu --format=csv,noheader
else
    echo "Warning: nvidia-smi not found. GPU tasks may not work properly."
fi

# Check Python dependencies
echo "Checking Python dependencies..."
python -c "import requests, torch; print('Dependencies OK')"

# Start the worker
echo "Starting worker process..."
exec python -m app/main.py
