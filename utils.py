import os
import json
import torch
import numpy as np
import logging
import random
import subprocess
import re
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import socket
import time
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_client_config(client_id: str, server_address: str, data_path: str) -> Dict[str, Any]:
    """
    Create a configuration for a client.
    
    Args:
        client_id: Client ID
        server_address: Server address
        data_path: Path to the data file
        
    Returns:
        Client configuration
    """
    config = {
        "client_id": client_id,
        "server_address": server_address,
        "data_path": data_path,
        "created_at": time.time()
    }
    
    # Create client directory if it doesn't exist
    os.makedirs(f"client_data/{client_id}", exist_ok=True)
    
    # Save configuration
    with open(f"client_data/{client_id}/config.json", "w") as f:
        json.dump(config, f)
    
    return config

def start_server(host: str = "0.0.0.0", port: int = 8080, num_rounds: int = 10) -> subprocess.Popen:
    """
    Start the federated learning server.
    
    Args:
        host: Server host
        port: Server port
        num_rounds: Number of rounds
        
    Returns:
        Server process
    """
    cmd = [
        sys.executable,
        "server.py",
        "--host", host,
        "--port", str(port),
        "--rounds", str(num_rounds)
    ]
    
    server_process = subprocess.Popen(cmd)
    
    # Wait for server to start
    time.sleep(2)
    
    return server_process

def start_client(client_id: str, server_address: str, data_path: str) -> subprocess.Popen:
    """
    Start a federated learning client.
    
    Args:
        client_id: Client ID
        server_address: Server address
        data_path: Path to the data file
        
    Returns:
        Client process
    """
    cmd = [
        sys.executable,
        "client.py",
        "--server-address", server_address,
        "--client-id", client_id,
        "--data-path", data_path
    ]
    
    client_process = subprocess.Popen(cmd)
    
    return client_process

def check_port_open(host: str, port: int) -> bool:
    """
    Check if a port is open.
    
    Args:
        host: Host to check
        port: Port to check
        
    Returns:
        True if the port is open, False otherwise
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        s.shutdown(socket.SHUT_RDWR)
        return True
    except:
        return False
    finally:
        s.close()

def seed_everything(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """
    Get the device to use for PyTorch.
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def setup_directories() -> None:
    """Create necessary directories."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("client_data", exist_ok=True)
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("simulation_data", exist_ok=True)

def load_client_history(client_id: str) -> Dict[str, Any]:
    """
    Load client training history.
    
    Args:
        client_id: Client ID
        
    Returns:
        Client history
    """
    history_path = f"client_data/{client_id}/history.json"
    
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            return json.load(f)
    
    return {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "rounds": []
    }

def get_active_clients() -> List[str]:
    """
    Get a list of active clients.
    
    Returns:
        List of active client IDs
    """
    if not os.path.exists("client_data"):
        return []
    
    clients = []
    
    for client_id in os.listdir("client_data"):
        config_path = f"client_data/{client_id}/config.json"
        
        if os.path.exists(config_path):
            clients.append(client_id)
    
    return clients

def check_dependencies() -> bool:
    """
    Check if all dependencies are installed.
    
    Returns:
        True if all dependencies are installed, False otherwise
    """
    dependencies = {
        "tshark": "tshark --version",
        "mininet": "mn --version",
        "hping3": "which hping3",
        "nmap": "nmap --version",
        "iperf": "iperf --version"
    }
    
    missing = []
    
    for dep, cmd in dependencies.items():
        try:
            subprocess.check_output(cmd, shell=True)
        except:
            missing.append(dep)
    
    if missing:
        logger.warning(f"Missing dependencies: {', '.join(missing)}")
        return False
    
    return True
