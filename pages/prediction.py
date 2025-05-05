import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import sys
import torch
from torch.serialization import add_safe_globals
import numpy as np
from numpy import ndarray

# Add NumPy arrays to PyTorch safe list
add_safe_globals([numpy.ndarray, numpy._core.multiarray._reconstruct])

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import display_prediction_interface
from model import TinyBERTModel
from data_processor import process_pcap_file, create_textual_description
import auth_helper

st.set_page_config(
    page_title="Prediction",
    page_icon="🔮",
    layout="wide"
)

# Check if user is authenticated
auth_helper.check_auth()

# Initialize sidebar with logout button
auth_helper.init_sidebar()

st.title("Network Traffic Prediction")
st.markdown("""
This page allows you to upload new traffic logs and get predictions from the trained model.
Upload a PCAP or CSV file to classify the traffic.
""")

# Display prediction interface
display_prediction_interface()

# Add single packet prediction
st.subheader("Single Packet Prediction")
st.markdown("""
You can also enter details of a single network packet to get a prediction.
""")

with st.form("single_packet"):
    col1, col2 = st.columns(2)
    
    with col1:
        src_ip = st.text_input("Source IP", value="192.168.1.1")
        src_port = st.number_input("Source Port", min_value=0, max_value=65535, value=1234)
        protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP", "HTTP", "DNS", "Other"])
    
    with col2:
        dst_ip = st.text_input("Destination IP", value="10.0.0.1")
        dst_port = st.number_input("Destination Port", min_value=0, max_value=65535, value=80)
        packet_length = st.number_input("Packet Length", min_value=0, value=128)
    
    additional_info = st.text_area("Additional Information (optional)", height=100)
    
    submitted = st.form_submit_button("Get Prediction")
    
    if submitted:
        # Create text representation of the packet
        packet_text = f"Source IP: {src_ip} Source Port: {src_port} Destination IP: {dst_ip} "
        packet_text += f"Destination Port: {dst_port} Protocol: {protocol} Packet Length: {packet_length} "
        
        if additional_info:
            packet_text += f"Additional Info: {additional_info}"
        
        # Check if model exists
        model_dir = "models"
        if not os.path.exists(model_dir):
            st.warning("No model available. Train a model first.")
        else:
            # Get the latest model
            model_files = [f for f in os.listdir(model_dir) if f.startswith("model_round_") and f.endswith(".pt")]
            
            if not model_files:
                st.warning("No model files available. Train a model first.")
            else:
                model_path = os.path.join(model_dir, sorted(model_files, key=lambda x: int(x.split("_")[2].split(".")[0]))[-1])
                
                # Load the model
                model = TinyBERTModel()
                
                try:
                    # Try loading with weights_only=False for backward compatibility
                    checkpoint = torch.load(model_path, weights_only=False)
                    
                    # Check if it's the new format (dictionary with model_state_dict)
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        # Load the model from the state dict
                        model.load_state_dict(checkpoint["model_state_dict"])
                    else:
                        # Old format - direct load
                        model.load_state_dict(checkpoint)
                except Exception as e:
                    st.warning(f"Could not load full model details: {str(e)}")
                    st.info("Using default model parameters")
                    
                model.eval()
                
                # Make prediction
                predictions, probabilities = model.predict([packet_text])
                
                # Display prediction
                prediction = predictions[0]
                probability = probabilities[0][prediction]
                
                label = "normal" if prediction == 0 else "anomaly"
                
                st.markdown(f"### Prediction: **{label.upper()}**")
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(probability * 100),
                    title={"text": f"Confidence ({label})"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#4CAF50" if label == "normal" else "#F44336"},
                        "steps": [
                            {"range": [0, 50], "color": "#E0E0E0"},
                            {"range": [50, 75], "color": "#BDBDBD"},
                            {"range": [75, 100], "color": "#9E9E9E"}
                        ]
                    }
                ))
                
                fig.update_layout(height=300)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show explanation
                st.markdown("### Explanation")
                
                features = []
                if prediction == 1:  # anomaly
                    if protocol == "TCP" and dst_port == 80 and packet_length > 1000:
                        features.append("Large TCP packet to web port (possibly DoS)")
                    if src_port < 1024:
                        features.append("Source port in privileged range")
                    if protocol == "ICMP" and packet_length > 500:
                        features.append("Large ICMP packet (possibly ping flood)")
                else:  # normal
                    if protocol == "HTTP" and dst_port == 80:
                        features.append("Standard web traffic")
                    if protocol == "DNS" and dst_port == 53:
                        features.append("Standard DNS query")
                    if src_ip.startswith("192.168") and dst_ip.startswith("192.168"):
                        features.append("Internal network traffic")
                
                # Add some generic features
                features.append(f"{protocol} protocol identified")
                features.append(f"Packet length of {packet_length} bytes")
                
                for feature in features:
                    st.markdown(f"- {feature}")

# Add batch prediction section
st.subheader("Batch Prediction")
st.markdown("""
If you want to predict multiple packets at once, use the file upload feature above.
""")

# Add model explanation
st.subheader("How the Model Works")
st.markdown("""
The model uses a 1-bit quantized TinyBERT architecture to classify network traffic.
It processes network packet data as text and makes predictions based on learned patterns.

**Key Features:**
- Processes network traffic as textual descriptions
- Uses transformer-based architecture for contextual understanding
- Identifies anomalies based on patterns learned during federated training
- Optimized with 1-bit quantization for efficiency
- Privacy-preserving through federated learning
""")

# Add a simple diagram of the prediction process
st.markdown("### Prediction Process")

# Create a flowchart using plotly
fig = go.Figure()

steps = ["Network Packet", "Text Conversion", "BERT Tokenization", "TinyBERT Model", "Classification Head", "Prediction"]
x_positions = [i for i in range(len(steps))]
y_positions = [0 for _ in range(len(steps))]

# Add nodes
for i, step in enumerate(steps):
    fig.add_trace(go.Scatter(
        x=[x_positions[i]],
        y=[y_positions[i]],
        mode="markers+text",
        marker=dict(size=30, color="#1E88E5"),
        text=[step],
        textposition="bottom center",
        name=step
    ))

# Add edges
for i in range(len(steps) - 1):
    fig.add_trace(go.Scatter(
        x=[x_positions[i], x_positions[i + 1]],
        y=[y_positions[i], y_positions[i + 1]],
        mode="lines+markers",
        line=dict(width=2, color="#9E9E9E"),
        marker=dict(size=10, color="#9E9E9E"),
        showlegend=False
    ))

fig.update_layout(
    title="Network Traffic Prediction Process",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor="rgba(0,0,0,0)",
    showlegend=False,
    height=300,
    margin=dict(l=20, r=20, t=40, b=100)
)

st.plotly_chart(fig, use_container_width=True)
