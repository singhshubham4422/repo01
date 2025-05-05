import pandas as pd
import numpy as np
import os
import logging
import json
from typing import List, Dict, Tuple, Optional, Union
import re
import random
import datetime

# Commented out transformer dependencies
# import torch
# from torch.utils.data import Dataset
# from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkTrafficDataset:
    """
    Dataset class for network traffic data.
    Converts network traffic logs into a format suitable for model training.
    (Mock version for demo without transformers dependency)
    """
    
    def __init__(
        self, 
        data: pd.DataFrame,
        tokenizer_name: str = "prajjwal1/bert-tiny",
        max_length: int = 128,
        preprocess_text: bool = True,
        label_col: str = "label"
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame containing network traffic data
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length for tokenization
            preprocess_text: Whether to preprocess text before tokenization
            label_col: Name of the column containing labels
        """
        self.data = data
        self.max_length = max_length
        self.preprocess_text = preprocess_text
        self.label_col = label_col
        
        # Convert labels to numeric if they are strings
        if self.label_col in self.data.columns and self.data[self.label_col].dtype == 'object':
            self.label_map = {label: i for i, label in enumerate(self.data[self.label_col].unique())}
            self.data['label_idx'] = self.data[self.label_col].map(self.label_map)
        else:
            if self.label_col in self.data.columns:
                self.data['label_idx'] = self.data[self.label_col]
                self.label_map = {i: i for i in self.data[self.label_col].unique()}
            else:
                # Add a default label column if it doesn't exist
                self.data['label_idx'] = 0
                self.label_map = {0: 'normal', 1: 'anomaly'}
        
        # Create text representations of network data
        self.prepare_text_features()
        
    def prepare_text_features(self):
        """Convert network traffic data to text representations."""
        # Select relevant columns and convert to text
        text_features = []
        
        for idx, row in self.data.iterrows():
            # Create a text representation of the network traffic
            text = self._row_to_text(row)
            text_features.append(text)
        
        self.data['text_features'] = text_features
    
    def _row_to_text(self, row: pd.Series) -> str:
        """
        Convert a row of network traffic data to text.
        
        Args:
            row: A row of network traffic data
            
        Returns:
            Text representation of the network traffic
        """
        text_parts = []
        
        # Add source and destination IP
        if 'src_ip' in row and 'dst_ip' in row:
            text_parts.append(f"Source IP: {row['src_ip']} Destination IP: {row['dst_ip']}")
        
        # Add protocol information
        if 'protocol' in row:
            text_parts.append(f"Protocol: {row['protocol']}")
        
        # Add port information
        if 'src_port' in row and 'dst_port' in row:
            text_parts.append(f"Source Port: {row['src_port']} Destination Port: {row['dst_port']}")
        
        # Add packet information
        if 'packet_length' in row:
            text_parts.append(f"Packet Length: {row['packet_length']}")
        
        # Add timestamp if available
        if 'timestamp' in row:
            text_parts.append(f"Time: {row['timestamp']}")
        
        # Add flags if available
        if 'tcp_flags' in row:
            text_parts.append(f"TCP Flags: {row['tcp_flags']}")
        
        # Add payload information if available
        if 'payload' in row:
            text_parts.append(f"Payload: {row['payload'][:50]}...")
        
        # Combine all parts
        text = " ".join(text_parts)
        
        if self.preprocess_text:
            text = self._preprocess_text(text)
        
        return text
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for tokenization.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Replace multiple whitespaces with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except spaces and alphanumeric
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Mock version that returns text and label directly, without tokenization.
        In a real implementation, this would return tokenized text.
        """
        text = self.data.iloc[idx]['text_features']
        label = self.data.iloc[idx]['label_idx']
        
        return text, label

def process_pcap_file(pcap_path: str, output_dir: str = "processed_data") -> str:
    """
    Process a PCAP file and convert it to a CSV file.
    Mock implementation for demo without tshark dependency.
    
    Args:
        pcap_path: Path to the PCAP file
        output_dir: Directory to save the CSV file
        
    Returns:
        Path to the generated CSV file
    """
    logger.info(f"Processing PCAP file: {pcap_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    base_name = os.path.basename(pcap_path).split('.')[0]
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    
    # Generate sample data for demo
    num_records = 100
    
    # Generate random timestamps
    now = datetime.datetime.now()
    timestamps = [(now - datetime.timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") for i in range(num_records)]
    
    # Generate random IPs
    src_ips = [f"192.168.1.{random.randint(1, 20)}" for _ in range(num_records)]
    dst_ips = [f"10.0.0.{random.randint(1, 10)}" for _ in range(num_records)]
    
    # Generate random ports
    src_ports = [random.randint(1024, 65535) for _ in range(num_records)]
    dst_ports = [random.choice([80, 443, 22, 53, 8080, 8443]) for _ in range(num_records)]
    
    # Generate random protocols
    protocols = [random.choice(["TCP", "UDP", "ICMP"]) for _ in range(num_records)]
    
    # Generate random packet lengths
    packet_lengths = [random.randint(64, 1500) for _ in range(num_records)]
    
    # Generate random TCP flags
    tcp_flags = ["0x" + format(random.randint(0, 255), '02x') for _ in range(num_records)]
    
    # Generate random payload samples
    payloads = ["".join([chr(random.randint(32, 126)) for _ in range(20)]) for _ in range(num_records)]
    
    # Create a DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'src_ip': src_ips,
        'dst_ip': dst_ips,
        'src_port': src_ports,
        'dst_port': dst_ports,
        'protocol': protocols,
        'packet_length': packet_lengths,
        'tcp_flags': tcp_flags,
        'payload': payloads
    })
    
    # Generate some anomalies
    anomaly_indices = random.sample(range(num_records), int(num_records * 0.1))  # 10% anomalies
    df['label'] = 'normal'
    
    for idx in anomaly_indices:
        anomaly_type = random.choice(['dos', 'port_scan', 'data_exfiltration'])
        df.loc[idx, 'label'] = anomaly_type
        
        # Modify features to make them look like actual anomalies
        if anomaly_type == 'dos':
            df.loc[idx, 'packet_length'] = random.randint(1000, 1500)
        elif anomaly_type == 'port_scan':
            df.loc[idx, 'dst_port'] = random.randint(1, 1023)  # Common service ports
        elif anomaly_type == 'data_exfiltration':
            df.loc[idx, 'packet_length'] = random.randint(1000, 1500)
            df.loc[idx, 'dst_ip'] = f"203.0.113.{random.randint(1, 254)}"  # External IP
    
    # Save the data to CSV
    df.to_csv(csv_path, index=False)
    logger.info(f"Generated mock CSV data to {csv_path}")
    
    return csv_path

def label_anomalies(csv_path: str, rules: List[Dict] = None) -> str:
    """
    Label network traffic anomalies based on rules.
    
    Args:
        csv_path: Path to the CSV file
        rules: List of rules to identify anomalies
        
    Returns:
        Path to the labeled CSV file
    """
    logger.info(f"Labeling anomalies in {csv_path}")
    
    # Default rules if none provided
    if rules is None:
        rules = [
            {
                'name': 'DoS',
                'condition': lambda row: row['src_ip'] == row['src_ip'].mode()[0] and row['packet_length'] > 1000,
                'label': 'dos'
            },
            {
                'name': 'Port Scan',
                'condition': lambda group: len(group['dst_port'].unique()) > 10,
                'group_by': 'src_ip',
                'label': 'port_scan'
            },
            # Add more rules as needed
        ]
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Apply rules
    for rule in rules:
        if 'group_by' in rule:
            # Group-based rules
            grouped = df.groupby(rule['group_by'])
            for name, group in grouped:
                if rule['condition'](group):
                    df.loc[group.index, 'label'] = rule['label']
        else:
            # Row-based rules
            mask = df.apply(rule['condition'], axis=1)
            df.loc[mask, 'label'] = rule['label']
    
    # Save the labeled CSV
    labeled_path = csv_path.replace('.csv', '_labeled.csv')
    df.to_csv(labeled_path, index=False)
    logger.info(f"Saved labeled CSV to {labeled_path}")
    
    return labeled_path

def create_textual_description(csv_path: str) -> List[str]:
    """
    Create textual descriptions of network traffic for LLM processing.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of textual descriptions
    """
    logger.info(f"Creating textual descriptions from {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Create dataset instance to use its text conversion methods
    dataset = NetworkTrafficDataset(df)
    
    # Get text features
    descriptions = df['text_features'].tolist()
    
    return descriptions
