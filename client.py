import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from collections import OrderedDict
import argparse
import logging
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json
import os

from model import TinyBERTModel
from data_processor import NetworkTrafficDataset, process_pcap_file
from quantization import quantize_model_update, dequantize_model_update

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkTrafficClient(fl.client.NumPyClient):
    """Flower client implementing network traffic analysis with 1-bit quantization."""
    
    def __init__(
        self,
        model: TinyBERTModel,
        train_dataset: Dataset,
        val_dataset: Dataset,
        client_id: str,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.client_id = client_id
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32)
        
        # Training settings
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Save client training history
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "rounds": []
        }
        os.makedirs(f"client_data/{client_id}", exist_ok=True)
        
    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        round_num = config.get("round_num", 0)
        epochs = config.get("epochs", 1)
        
        # Update model with the latest parameters
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.train()
        
        # Train for the specified number of epochs
        train_loss = 0.0
        correct = 0
        total = 0
        
        for _ in range(epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                epoch_total += targets.size(0)
                epoch_correct += predicted.eq(targets).sum().item()
            
            train_loss += epoch_loss / len(self.train_loader)
            correct += epoch_correct
            total += epoch_total
        
        # Calculate training metrics
        train_loss = train_loss / epochs
        train_accuracy = 100. * correct / total if total > 0 else 0.0
        
        # Save history
        self.history["train_loss"].append(train_loss)
        self.history["train_accuracy"].append(train_accuracy)
        self.history["rounds"].append(round_num)
        
        with open(f"client_data/{self.client_id}/history.json", "w") as f:
            json.dump(self.history, f)
        
        # Get model parameters after training
        parameters_updated = self.get_parameters(config={})
        
        # Apply 1-bit quantization
        quantized_parameters = [quantize_model_update(param) for param in parameters_updated]
        
        # Log metrics
        logger.info(f"Client {self.client_id}: round {round_num}, train loss {train_loss:.4f}, "
                   f"train accuracy {train_accuracy:.2f}%")
        
        return quantized_parameters, len(self.train_loader.dataset), {
            "loss": float(train_loss),
            "accuracy": float(train_accuracy)
        }
    
    def evaluate(self, parameters, config):
        """Evaluate the model on the local validation dataset."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()
        
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate validation metrics
        val_loss = val_loss / len(self.val_loader)
        val_accuracy = 100. * correct / total if total > 0 else 0.0
        
        # Save history
        round_num = config.get("round_num", 0)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)
        
        with open(f"client_data/{self.client_id}/history.json", "w") as f:
            json.dump(self.history, f)
        
        # Log metrics
        logger.info(f"Client {self.client_id}: evaluation, val loss {val_loss:.4f}, "
                   f"val accuracy {val_accuracy:.2f}%")
        
        return float(val_loss), len(self.val_loader.dataset), {
            "loss": float(val_loss),
            "accuracy": float(val_accuracy)
        }

def main(server_address, client_id, data_path):
    """Initialize and start a federated learning client."""
    # Check if PCAP or CSV file
    if data_path.endswith('.pcap'):
        # Process PCAP file and convert to CSV
        csv_path = process_pcap_file(data_path)
        data_path = csv_path
    
    # Load dataset
    df = pd.read_csv(data_path)
    
    # Split into train and validation
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    
    # Create datasets
    train_dataset = NetworkTrafficDataset(train_df)
    val_dataset = NetworkTrafficDataset(val_df)
    
    # Initialize model
    model = TinyBERTModel()
    
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Initialize client
    client = NetworkTrafficClient(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        client_id=client_id,
        device=device
    )
    
    # Start client
    fl.client.start_numpy_client(server_address=server_address, client=client)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    
    parser.add_argument(
        "--server-address",
        type=str,
        default="127.0.0.1:8080",
        help="Server address (default: 127.0.0.1:8080)"
    )
    
    parser.add_argument(
        "--client-id",
        type=str,
        required=True,
        help="Client ID"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to CSV or PCAP file containing network traffic data"
    )
    
    args = parser.parse_args()
    
    main(args.server_address, args.client_id, args.data_path)
