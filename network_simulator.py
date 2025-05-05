import subprocess
import threading
import time
import logging
import os
import pandas as pd
import random
from mininet.net import Mininet
from mininet.node import Controller
from mininet.cli import CLI
from mininet.log import setLogLevel
from typing import List, Dict, Any, Tuple
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetworkSimulator:
    """
    Network simulator using Mininet for generating network traffic data.
    """
    
    def __init__(self, output_dir: str = "simulation_data"):
        """
        Initialize the network simulator.
        
        Args:
            output_dir: Directory to save simulation data
        """
        self.output_dir = output_dir
        self.net = None
        self.is_running = False
        self.capture_process = None
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def create_network(self, topology: Dict[str, Any] = None) -> None:
        """
        Create a network using Mininet.
        
        Args:
            topology: Dictionary describing the network topology
        """
        # Default topology if none provided
        if topology is None:
            topology = {
                'hosts': 3,
                'switches': 1,
                'controllers': 1
            }
        
        setLogLevel('info')
        
        # Create network with a controller
        self.net = Mininet(controller=Controller)
        
        # Add controller
        self.net.addController('c0')
        
        # Add switch
        s1 = self.net.addSwitch('s1')
        
        # Add hosts
        hosts = []
        for i in range(1, topology['hosts'] + 1):
            host = self.net.addHost(f'h{i}')
            hosts.append(host)
            self.net.addLink(host, s1)
        
        logger.info(f"Created network with {len(hosts)} hosts and 1 switch")
        
        # Start network
        self.net.start()
        self.is_running = True
        
        # Return the hosts for reference
        return hosts
    
    def start_packet_capture(self, interface: str = "s1-eth1", duration: int = 60) -> str:
        """
        Start packet capture on a specific interface.
        
        Args:
            interface: Network interface to capture packets from
            duration: Duration of capture in seconds
            
        Returns:
            Path to the PCAP file
        """
        if not self.is_running:
            raise RuntimeError("Network is not running. Call create_network() first.")
        
        # Generate output filename
        timestamp = int(time.time())
        pcap_path = os.path.join(self.output_dir, f"capture_{timestamp}.pcap")
        
        # Start tcpdump in a separate process
        cmd = ["tcpdump", "-i", interface, "-w", pcap_path]
        self.capture_process = subprocess.Popen(cmd)
        
        logger.info(f"Started packet capture on {interface}, saving to {pcap_path}")
        
        # Set a timer to stop the capture after the specified duration
        def stop_capture():
            if self.capture_process:
                logger.info(f"Stopping packet capture after {duration} seconds")
                self.capture_process.terminate()
                self.capture_process = None
        
        timer = threading.Timer(duration, stop_capture)
        timer.start()
        
        return pcap_path
    
    def generate_normal_traffic(self, num_flows: int = 5, duration: int = 30) -> None:
        """
        Generate normal network traffic between hosts.
        
        Args:
            num_flows: Number of traffic flows to generate
            duration: Duration of traffic generation in seconds
        """
        if not self.is_running:
            raise RuntimeError("Network is not running. Call create_network() first.")
        
        hosts = self.net.hosts
        
        # Generate random traffic between hosts
        for _ in range(num_flows):
            # Select random source and destination hosts
            src_host = random.choice(hosts)
            dst_host = random.choice([h for h in hosts if h != src_host])
            
            # Use iperf to generate traffic
            dst_host.cmd(f"iperf -s -p 5001 &")
            
            # Run iperf client in background with random parameters
            bandwidth = random.randint(1, 10)  # Mbps
            src_host.cmd(f"iperf -c {dst_host.IP()} -p 5001 -t {duration} -b {bandwidth}M &")
            
            logger.info(f"Started traffic flow from {src_host.name} to {dst_host.name} "
                       f"with {bandwidth} Mbps for {duration} seconds")
        
        # Wait for traffic to complete
        time.sleep(duration + 5)
    
    def generate_dos_attack(self, target_host_idx: int = 1, duration: int = 30) -> None:
        """
        Generate a DoS attack against a specific host.
        
        Args:
            target_host_idx: Index of the target host (1-based)
            duration: Duration of the attack in seconds
        """
        if not self.is_running:
            raise RuntimeError("Network is not running. Call create_network() first.")
        
        hosts = self.net.hosts
        target_host = hosts[target_host_idx - 1]
        
        # Select all other hosts as attackers
        attacker_hosts = [h for h in hosts if h != target_host]
        
        # Start a simple web server on the target
        target_host.cmd("python -m http.server 80 &")
        logger.info(f"Started web server on {target_host.name}")
        
        # Start DoS attack from each attacker
        for attacker in attacker_hosts:
            # Use hping3 for the DoS attack (SYN flood)
            attacker.cmd(f"hping3 -S -p 80 -i u10 {target_host.IP()} &")
            logger.info(f"Started DoS attack from {attacker.name} to {target_host.name}")
        
        # Wait for attack to complete
        time.sleep(duration)
        
        # Stop the attack
        for attacker in attacker_hosts:
            attacker.cmd("killall hping3")
        
        # Stop the web server
        target_host.cmd("killall python")
        
        logger.info(f"Stopped DoS attack after {duration} seconds")
    
    def generate_port_scan(self, target_host_idx: int = 1, duration: int = 30) -> None:
        """
        Generate a port scan against a specific host.
        
        Args:
            target_host_idx: Index of the target host (1-based)
            duration: Duration of the scan in seconds
        """
        if not self.is_running:
            raise RuntimeError("Network is not running. Call create_network() first.")
        
        hosts = self.net.hosts
        target_host = hosts[target_host_idx - 1]
        
        # Select a random host as the scanner
        scanner_host = random.choice([h for h in hosts if h != target_host])
        
        # Start some services on the target
        for port in [22, 80, 443]:
            target_host.cmd(f"nc -l -p {port} &")
        
        # Start port scan
        scanner_host.cmd(f"nmap -T4 {target_host.IP()} &")
        logger.info(f"Started port scan from {scanner_host.name} to {target_host.name}")
        
        # Wait for scan to complete
        time.sleep(duration)
        
        # Stop services
        target_host.cmd("killall nc")
        
        logger.info(f"Completed port scan after {duration} seconds")
    
    def stop(self) -> None:
        """Stop the network and clean up."""
        if self.is_running:
            if self.capture_process:
                self.capture_process.terminate()
                self.capture_process = None
            
            if self.net:
                self.net.stop()
                self.is_running = False
                logger.info("Network stopped and cleaned up")

def generate_simulation_data(output_dir: str = "simulation_data", scenario: str = "mixed") -> List[str]:
    """
    Generate network simulation data for different scenarios.
    
    Args:
        output_dir: Directory to save simulation data
        scenario: Type of data to generate (normal, dos, port_scan, mixed)
        
    Returns:
        List of paths to generated PCAP files
    """
    pcap_files = []
    simulator = NetworkSimulator(output_dir=output_dir)
    
    try:
        hosts = simulator.create_network()
        
        if scenario == "normal" or scenario == "mixed":
            # Capture normal traffic
            pcap_path = simulator.start_packet_capture(duration=120)
            pcap_files.append(pcap_path)
            
            # Generate normal traffic
            simulator.generate_normal_traffic(num_flows=5, duration=100)
        
        if scenario == "dos" or scenario == "mixed":
            # Capture DoS attack traffic
            pcap_path = simulator.start_packet_capture(duration=120)
            pcap_files.append(pcap_path)
            
            # Generate normal background traffic
            simulator.generate_normal_traffic(num_flows=2, duration=100)
            
            # Generate DoS attack
            simulator.generate_dos_attack(duration=80)
        
        if scenario == "port_scan" or scenario == "mixed":
            # Capture port scan traffic
            pcap_path = simulator.start_packet_capture(duration=120)
            pcap_files.append(pcap_path)
            
            # Generate normal background traffic
            simulator.generate_normal_traffic(num_flows=2, duration=100)
            
            # Generate port scan
            simulator.generate_port_scan(duration=80)
        
    except Exception as e:
        logger.error(f"Error in simulation: {e}")
    finally:
        # Clean up
        simulator.stop()
    
    return pcap_files

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Received interrupt, shutting down...")
        if 'simulator' in locals() and simulator:
            simulator.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Generate simulation data
    pcap_files = generate_simulation_data(scenario="mixed")
    
    print(f"Generated {len(pcap_files)} PCAP files:")
    for pcap_file in pcap_files:
        print(f"- {pcap_file}")
