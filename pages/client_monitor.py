import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import display_client_monitoring
from utils import load_client_history, get_active_clients
import auth_helper

st.set_page_config(
    page_title="Client Monitor",
    page_icon="👥",
    layout="wide"
)

# Check if user is authenticated
auth_helper.check_auth()

# Initialize sidebar with logout button
auth_helper.init_sidebar()

st.title("Client Participation Monitor")
st.markdown("""
This page shows which clients are participating in the federated learning process,
their contributions, and individual performance metrics.
""")

# Display client participation
display_client_monitoring()

# Display individual client metrics
st.subheader("Individual Client Performance")

active_clients = get_active_clients()

if not active_clients:
    st.info("No client data available yet.")
else:
    # Create tabs for each client
    tabs = st.tabs([f"Client {client_id}" for client_id in active_clients])
    
    for i, client_id in enumerate(active_clients):
        with tabs[i]:
            # Load client history
            history = load_client_history(client_id)
            
            if not history["rounds"]:
                st.info(f"No training history available for Client {client_id} yet.")
                continue
            
            # Convert to DataFrame
            df = pd.DataFrame({
                "Round": history["rounds"],
                "Training Loss": history["train_loss"],
                "Training Accuracy": history["train_accuracy"]
            })
            
            if history["val_loss"] and history["val_accuracy"]:
                df["Validation Loss"] = history["val_loss"]
                df["Validation Accuracy"] = history["val_accuracy"]
            
            # Display metrics
            col1, col2 = st.columns(2)
            
            with col1:
                # Training metrics
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(
                    x=df["Round"],
                    y=df["Training Loss"],
                    mode="lines+markers",
                    name="Training Loss"
                ))
                
                if "Validation Loss" in df.columns:
                    fig1.add_trace(go.Scatter(
                        x=df["Round"],
                        y=df["Validation Loss"],
                        mode="lines+markers",
                        name="Validation Loss"
                    ))
                
                fig1.update_layout(
                    title=f"Client {client_id} - Loss",
                    xaxis_title="Round",
                    yaxis_title="Loss",
                    height=300
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Accuracy metrics
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=df["Round"],
                    y=df["Training Accuracy"],
                    mode="lines+markers",
                    name="Training Accuracy"
                ))
                
                if "Validation Accuracy" in df.columns:
                    fig2.add_trace(go.Scatter(
                        x=df["Round"],
                        y=df["Validation Accuracy"],
                        mode="lines+markers",
                        name="Validation Accuracy"
                    ))
                
                fig2.update_layout(
                    title=f"Client {client_id} - Accuracy",
                    xaxis_title="Round",
                    yaxis_title="Accuracy",
                    height=300
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Display client config if available
            config_path = f"client_data/{client_id}/config.json"
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                
                st.markdown("### Client Configuration")
                st.json(config)

# Add client registration section
st.subheader("Register New Client")

with st.form("client_registration"):
    client_id = st.text_input("Client ID", value=f"client_{len(active_clients) + 1}")
    server_address = st.text_input("Server Address", value="0.0.0.0:8080")
    
    data_file = st.file_uploader("Upload Network Traffic Data (CSV or PCAP)", type=["csv", "pcap"])
    
    submitted = st.form_submit_button("Register Client")
    
    if submitted and data_file:
        st.info(f"Registering client {client_id}...")
        
        # Save uploaded file
        file_path = f"uploaded_{data_file.name}"
        with open(file_path, "wb") as f:
            f.write(data_file.getbuffer())
        
        # Create client config
        from utils import create_client_config
        config = create_client_config(client_id, server_address, file_path)
        
        st.success(f"Client {client_id} registered successfully!")
        st.json(config)
        
        # Add instructions to start the client
        st.markdown(f"""
        ### Start Client
        
        Run the following command to start the client:
        
        ```bash
        python client.py --server-address {server_address} --client-id {client_id} --data-path {file_path}
        ```
        """)
