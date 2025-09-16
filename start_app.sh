#!/bin/bash

# Omega Portfolio Engine Startup Script
echo "Starting Omega Portfolio Engine web application..."
echo "API will be available at: http://localhost:8000"
echo "Web UI will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop both services"
echo ""

# Activate virtual environment
source venv/bin/activate

# Check if API is already running
if lsof -i :8000 > /dev/null 2>&1; then
    echo "API server is already running on port 8000"
    API_PID=$(lsof -ti :8000)
    echo "Using existing API server (PID: $API_PID)"
else
    # Start API server in background
    echo "Starting API server in background..."
    python -m api.main &
    API_PID=$!
    
    # Wait for API to start
    echo "Waiting for API to start..."
    sleep 5
    
    # Check if API is running
    if ps -p $API_PID > /dev/null; then
        echo "API server started successfully (PID: $API_PID)"
    else
        echo "Failed to start API server"
        exit 1
    fi
fi

# Start Streamlit UI
echo "Starting web UI..."
streamlit run app/ui.py --server.port 8501 --server.address 0.0.0.0

# Cleanup function
cleanup() {
    echo "Shutting down services..."
    # Only kill API if we started it ourselves
    if [ "$API_PID" != "" ] && ps -p $API_PID > /dev/null 2>&1; then
        # Check if this is a process we started (not an existing one)
        if lsof -i :8000 > /dev/null 2>&1; then
            echo "Stopping API server..."
            kill $API_PID 2>/dev/null
        else
            echo "API server was already running, leaving it running"
        fi
    fi
    pkill -f streamlit 2>/dev/null
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait
