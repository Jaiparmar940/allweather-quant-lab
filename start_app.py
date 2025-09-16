#!/usr/bin/env python3
"""
Omega Portfolio Engine Startup Script
Starts both API and UI services with proper error handling
"""

import subprocess
import time
import signal
import sys
import os
import socket
from pathlib import Path

def check_port(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def start_api():
    """Start the API server"""
    if check_port(8000):
        print("API server is already running on port 8000")
        return None
    
    print("Starting API server...")
    try:
        # Start API in background
        process = subprocess.Popen([
            sys.executable, '-m', 'api.main'
        ], cwd=Path(__file__).parent)
        
        # Wait for API to start
        print("Waiting for API to start...")
        for i in range(10):
            if check_port(8000):
                print("API server started successfully")
                return process
            time.sleep(1)
        
        print("Failed to start API server")
        process.terminate()
        return None
        
    except Exception as e:
        print(f"Error starting API server: {e}")
        return None

def start_ui():
    """Start the Streamlit UI"""
    if check_port(8501):
        print("UI is already running on port 8501")
        return None
    
    print("Starting web UI...")
    try:
        # Start Streamlit
        process = subprocess.Popen([
            sys.executable, '-m', 'streamlit', 'run', 'app/ui.py',
            '--server.port', '8501',
            '--server.address', '0.0.0.0',
            '--server.headless', 'true'
        ], cwd=Path(__file__).parent)
        
        # Wait for UI to start
        print("Waiting for UI to start...")
        for i in range(15):
            if check_port(8501):
                print("UI started successfully")
                return process
            time.sleep(1)
        
        print("UI may not have started properly, but continuing...")
        return process
        
    except Exception as e:
        print(f"Error starting UI: {e}")
        return None

def cleanup(api_process, ui_process):
    """Clean up processes"""
    print("\nShutting down services...")
    
    if ui_process:
        print("Stopping UI...")
        ui_process.terminate()
        try:
            ui_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            ui_process.kill()
    
    if api_process:
        print("Stopping API...")
        api_process.terminate()
        try:
            api_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            api_process.kill()

def main():
    """Main startup function"""
    print("Starting Omega Portfolio Engine web application...")
    print("API will be available at: http://localhost:8000")
    print("Web UI will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop both services")
    print("")
    
    api_process = None
    ui_process = None
    
    try:
        # Start API
        api_process = start_api()
        if not api_process and not check_port(8000):
            print("Failed to start API server")
            return 1
        
        # Start UI
        ui_process = start_ui()
        
        print("\n" + "="*50)
        print("Services are running!")
        print("API: http://localhost:8000")
        print("UI:  http://localhost:8501")
        print("Press Ctrl+C to stop")
        print("="*50)
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nReceived interrupt signal...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cleanup(api_process, ui_process)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
