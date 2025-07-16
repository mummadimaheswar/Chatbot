#!/bin/bash

# Start script for DevBot FastAPI application

echo "Starting DevBot API..."

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    # In Docker, use uvicorn directly
    exec uvicorn main:app --host 0.0.0.0 --port 8000
else
    echo "Running locally"
    # Local development
    if [ -f .env ]; then
        echo "Loading environment from .env file"
        export $(cat .env | xargs)
    fi
    
    # Install dependencies if needed
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv venv
        source venv/bin/activate
        pip install -r requirements.txt
    else
        source venv/bin/activate
    fi
    
    # Start the application
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
fi