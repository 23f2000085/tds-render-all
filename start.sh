#!/bin/bash
# Start Redis server in the background
redis-server --daemonize yes

# Start the FastAPI application on the port provided by Render (default 10000)
uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
