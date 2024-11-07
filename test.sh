#!/bin/bash

# Stop and remove the existing container (if it exists)
#docker stop whisper-fastapi
#docker rm whisper-fastapi

# Rebuild the Docker image
docker build -t whisper-turbo-api .

# Run the Docker container
docker run -d -p 8000:8000 whisper-turbo-api

# Wait for the container to start (adjust sleep time if needed)
sleep 90

# Test the API endpoint using curl
curl -X POST -F file=@./audio.mp3 http://localhost:8000/transcribe