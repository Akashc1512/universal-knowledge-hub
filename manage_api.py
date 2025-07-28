#!/usr/bin/env python3
"""
Universal Knowledge Platform API Management Script
Provides utilities for starting, stopping, and monitoring the API server.
"""

import subprocess
import requests
import socket
import sys
import os
import signal
import psutil
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
DEFAULT_PORT = int(os.getenv("UKP_PORT", "8002"))
DEFAULT_HOST = os.getenv("UKP_HOST", "0.0.0.0")
HEALTH_CHECK_TIMEOUT = int(os.getenv("HEALTH_CHECK_TIMEOUT", "5"))
API_BASE_URL = f"http://{os.getenv('UKP_HOST', 'localhost')}:{os.getenv('UKP_PORT', '8002')}"


def check_port_available(host, port):
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        return False


def start_server(port=DEFAULT_PORT):
    """Start the API server."""
    if not check_port_available(DEFAULT_HOST, port):
        print(f"❌ Port {port} is already in use")
        return False

    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
        if response.status_code == 200:
            print(f"✅ Server already running on port {port}")
            return True
    except requests.exceptions.RequestException:
        pass

    print(f"🚀 Starting server on port {port}...")
    try:
        process = subprocess.Popen(
            [sys.executable, "start_api.py"], env={**os.environ, "UKP_PORT": str(port)}
        )

        # Wait for server to start
        time.sleep(3)

        # Check if server started successfully
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
            if response.status_code == 200:
                print(f"✅ Server started successfully on port {port}")
                return True
        except requests.exceptions.RequestException:
            print(f"❌ Server failed to start on port {port}")
            return False

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False


def stop_server(port=DEFAULT_PORT):
    """Stop the API server."""
    print(f"🛑 Stopping server on port {port}...")

    # Find and kill processes using the port
    killed = False
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            connections = proc.connections()
            for conn in connections:
                if conn.laddr.port == port:
                    print(f"🔄 Killing process {proc.info['pid']} using port {port}")
                    proc.terminate()
                    proc.wait(timeout=5)
                    killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            pass

    if killed:
        print(f"✅ Server stopped on port {port}")
    else:
        print(f"ℹ️ No server found running on port {port}")


def status(port=DEFAULT_PORT):
    """Check server status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=HEALTH_CHECK_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server is running on port {port}")
            print(f"   Status: {data.get('status', 'unknown')}")
            print(f"   Version: {data.get('version', 'unknown')}")
            print(f"   Timestamp: {data.get('timestamp', 'unknown')}")
            return True
        else:
            print(f"⚠️ Server responded with status {response.status_code}")
            return False
    except requests.exceptions.RequestException:
        print(f"❌ Server is not running on port {port}")
        return False


def restart_server(port=DEFAULT_PORT):
    """Restart the API server."""
    print(f"🔄 Restarting server on port {port}...")
    stop_server(port)
    time.sleep(2)
    return start_server(port)


def get_logs(port=DEFAULT_PORT, lines=50):
    """Get recent server logs."""
    print(f"📋 Recent logs for server on port {port}:")
    try:
        # This would need to be implemented based on your logging setup
        # For now, we'll just check if the server is running
        if status(port):
            print("✅ Server is running - logs would be available here")
        else:
            print("❌ Server is not running")
    except Exception as e:
        print(f"❌ Error getting logs: {e}")


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python manage_api.py <command> [port]")
        print("Commands: start, stop, restart, status, logs")
        return

    command = sys.argv[1].lower()
    port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT

    if command == "start":
        start_server(port)
    elif command == "stop":
        stop_server(port)
    elif command == "restart":
        restart_server(port)
    elif command == "status":
        status(port)
    elif command == "logs":
        get_logs(port)
    else:
        print(f"❌ Unknown command: {command}")
        print("Available commands: start, stop, restart, status, logs")


if __name__ == "__main__":
    main()
