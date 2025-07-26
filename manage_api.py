#!/usr/bin/env python3
"""
API Server Management Script
"""

import os
import sys
import subprocess
import time
import requests
import socket

def check_port(port):
    """Check if a port is in use."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('0.0.0.0', port))
        sock.close()
        return False  # Port is available
    except OSError:
        return True   # Port is in use

def check_api_health(port):
    """Check if the API is responding."""
    try:
        response = requests.get(f'http://localhost:{port}/health', timeout=5)
        return response.status_code == 200
    except:
        return False

def start_server(port=8002):
    """Start the API server."""
    if check_port(port):
        print(f"❌ Port {port} is already in use")
        return False
    
    print(f"🚀 Starting API server on port {port}...")
    os.environ['UKP_PORT'] = str(port)
    
    try:
        subprocess.run([sys.executable, 'start_api.py'], check=True)
    except KeyboardInterrupt:
        print("🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False
    
    return True

def stop_server(port=8002):
    """Stop the API server."""
    print(f"🛑 Stopping API server on port {port}...")
    
    # Find and kill the process using the port
    try:
        result = subprocess.run(['netstat', '-ano'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if f':{port}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    subprocess.run(['taskkill', '/PID', pid, '/F'])
                    print(f"✅ Killed process {pid}")
                    return True
    except Exception as e:
        print(f"❌ Failed to stop server: {e}")
        return False
    
    print(f"⚠️  No process found on port {port}")
    return False

def status(port=8002):
    """Check server status."""
    print(f"📊 API Server Status (Port {port})")
    print("=" * 40)
    
    # Check if port is in use
    port_in_use = check_port(port)
    print(f"Port {port} in use: {'Yes' if port_in_use else 'No'}")
    
    # Check if API is responding
    api_responding = check_api_health(port)
    print(f"API responding: {'Yes' if api_responding else 'No'}")
    
    if api_responding:
        try:
            response = requests.get(f'http://localhost:{port}/health')
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Version: {data['version']}")
        except Exception as e:
            print(f"Error getting health data: {e}")
    
    return api_responding

def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python manage_api.py [start|stop|status|restart] [port]")
        print("  start   - Start the API server")
        print("  stop    - Stop the API server")
        print("  status  - Check server status")
        print("  restart - Restart the API server")
        return
    
    command = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8002
    
    if command == "start":
        start_server(port)
    elif command == "stop":
        stop_server(port)
    elif command == "status":
        status(port)
    elif command == "restart":
        print("🔄 Restarting API server...")
        stop_server(port)
        time.sleep(2)
        start_server(port)
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main() 