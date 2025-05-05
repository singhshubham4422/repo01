import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import sys
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import display_network_visualization
from data_processor import process_pcap_file
import auth_helper

st.set_page_config(
    page_title="Traffic Analysis",
    page_icon="🌐",
    layout="wide"
)

# Check if user is authenticated
auth_helper.check_auth()

# Initialize sidebar with logout button
auth_helper.init_sidebar()

st.title("Network Traffic Analysis")
st.markdown("""
This page allows you to analyze network traffic data.
Upload a PCAP or CSV file to visualize the traffic patterns.
""")

# File upload section
uploaded_file = st.file_uploader("Upload Network Traffic Data", type=["pcap", "csv"])

if uploaded_file is not None:
    # Save the uploaded file
    file_path = f"uploaded_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process the file
    if file_path.endswith(".pcap"):
        st.info("Processing PCAP file...")
        csv_path = process_pcap_file(file_path)
        st.success(f"PCAP file processed and converted to CSV: {csv_path}")
    else:
        csv_path = file_path
    
    # Display traffic analysis
    display_network_visualization(csv_path)
else:
    # Display sample traffic analysis
    st.info("Upload a PCAP or CSV file to analyze network traffic.")
    display_network_visualization()

# Add traffic filtering options
st.subheader("Traffic Filtering Options")

with st.expander("Filter Options"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        protocol_filter = st.multiselect(
            "Protocol Filter",
            ["TCP", "UDP", "ICMP", "HTTP", "DNS", "All"],
            default=["All"]
        )
    
    with col2:
        ip_filter = st.text_input("IP Address Filter (comma-separated)")
    
    with col3:
        time_range = st.slider(
            "Time Range (minutes ago)",
            min_value=0,
            max_value=60,
            value=(0, 30)
        )
    
    apply_filters = st.button("Apply Filters")
    
    if apply_filters:
        st.info("Filters applied. (This is a placeholder - actual filtering logic would be implemented here.)")

# Add attack detection section
st.subheader("Anomaly Detection")

with st.expander("Anomaly Detection Settings"):
    col1, col2 = st.columns(2)
    
    with col1:
        detection_method = st.selectbox(
            "Detection Method",
            ["Rule-based", "Machine Learning", "Statistical", "Hybrid"]
        )
    
    with col2:
        sensitivity = st.slider(
            "Detection Sensitivity",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1
        )
    
    run_detection = st.button("Run Anomaly Detection")
    
    if run_detection:
        # This is a placeholder for actual anomaly detection
        st.info("Running anomaly detection...")
        
        # Create sample anomaly data
        anomaly_data = {
            "timestamp": [f"2023-01-01 {i:02d}:00:00" for i in range(1, 6)],
            "src_ip": [f"192.168.1.{random.randint(1, 10)}" for _ in range(5)],
            "dst_ip": [f"10.0.0.{random.randint(1, 10)}" for _ in range(5)],
            "protocol": ["TCP", "UDP", "TCP", "TCP", "ICMP"],
            "anomaly_type": ["DoS", "Port Scan", "Data Exfiltration", "DoS", "Port Scan"],
            "confidence": [random.random() * 0.5 + 0.5 for _ in range(5)]
        }
        
        anomaly_df = pd.DataFrame(anomaly_data)
        
        st.success("Anomaly detection complete!")
        st.dataframe(anomaly_df)
        
        # Plot anomaly distribution
        fig = px.bar(
            anomaly_df,
            x="anomaly_type",
            y="confidence",
            color="protocol",
            title="Detected Anomalies by Type"
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Add traffic statistics section
st.subheader("Traffic Statistics")

tab1, tab2, tab3 = st.tabs(["Volume", "Protocols", "Connections"])

with tab1:
    # Generate sample time-series data for traffic volume
    timestamps = [f"2023-01-01 {i:02d}:00:00" for i in range(24)]
    volume = [random.randint(500, 2000) for _ in range(24)]
    
    # Create DataFrame
    volume_df = pd.DataFrame({
        "timestamp": timestamps,
        "packets": volume
    })
    
    # Create chart
    fig = px.line(
        volume_df,
        x="timestamp",
        y="packets",
        title="Traffic Volume Over Time"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Create sample protocol distribution
    protocols = ["TCP", "UDP", "ICMP", "HTTP", "DNS", "Other"]
    counts = [random.randint(100, 1000) for _ in range(len(protocols))]
    
    # Create DataFrame
    protocol_df = pd.DataFrame({
        "protocol": protocols,
        "count": counts
    })
    
    # Create chart
    fig = px.pie(
        protocol_df,
        values="count",
        names="protocol",
        title="Protocol Distribution"
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    # Create sample connection data
    src_ips = [f"192.168.1.{i}" for i in range(1, 6)]
    dst_ips = [f"10.0.0.{i}" for i in range(1, 6)]
    
    connections = []
    for src in src_ips:
        for dst in dst_ips:
            if random.random() > 0.3:  # 70% chance of a connection
                connections.append({
                    "source": src,
                    "destination": dst,
                    "packets": random.randint(10, 500)
                })
    
    # Create DataFrame
    connection_df = pd.DataFrame(connections)
    
    # Create a connection matrix
    matrix = pd.pivot_table(
        connection_df,
        values="packets",
        index="source",
        columns="destination",
        fill_value=0
    )
    
    # Create heatmap
    fig = px.imshow(
        matrix,
        labels=dict(x="Destination IP", y="Source IP", color="Packets"),
        x=matrix.columns,
        y=matrix.index,
        color_continuous_scale="Blues",
        title="Connection Matrix"
    )
    
    st.plotly_chart(fig, use_container_width=True)
