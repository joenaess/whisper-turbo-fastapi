#!/bin/bash

# Stop and remove the existing container (if it exists)
docker stop whisper-fastapi
docker rm whisper-fastapi

# Rebuild the Docker image
docker build -t whisper-fastapi .

# Run the Docker container
docker run -d -p 8000:8000 whisper-fastapi

# Wait for the container to start (adjust sleep time if needed)
sleep 5

# Test the API endpoint using curl
curl -X POST -F file=@./audio.mp3 http://localhost:8000/transcribe