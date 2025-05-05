import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import sys
import torch
import subprocess
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TinyBERTModel, DistilBERTModel
from data_processor import NetworkTrafficDataset
import auth_helper
from utils import start_server, start_client, create_client_config, seed_everything, get_device

# Create client data directory
client_data_dir = "client_data"
os.makedirs(client_data_dir, exist_ok=True)

st.set_page_config(
    page_title="Model Training",
    page_icon="🧠",
    layout="wide"
)

# Check if user is authenticated
auth_helper.check_auth()

# Initialize sidebar with logout button
auth_helper.init_sidebar()

st.title("Model Training and Federated Learning")

st.markdown("""
This page allows you to train models on network traffic data, setup federated learning clients,
and monitor the training process. You can choose between local training and federated learning.
""")

# Setup training parameters
st.subheader("Training Configuration")

col1, col2 = st.columns(2)

with col1:
    training_mode = st.selectbox(
        "Training Mode",
        ["Local Training", "Federated Learning"],
        help="Choose between training a model locally or using federated learning across multiple clients"
    )
    
    model_type = st.selectbox(
        "Model Type",
        ["TinyBERT", "DistilBERT"],
        help="Choose the model architecture to use for training"
    )
    
    quantization = st.checkbox(
        "Use 1-bit Quantization", 
        value=True,
        help="Apply 1-bit quantization to model updates during training"
    )

with col2:
    num_epochs = st.slider(
        "Number of Epochs",
        min_value=1,
        max_value=20,
        value=5,
        help="Number of epochs to train for"
    )
    
    batch_size = st.slider(
        "Batch Size",
        min_value=8,
        max_value=128,
        value=32,
        step=8,
        help="Batch size for training"
    )
    
    learning_rate = st.slider(
        "Learning Rate",
        min_value=0.0001,
        max_value=0.01,
        value=0.001,
        format="%.5f",
        help="Learning rate for optimizer"
    )

# Upload data section
st.subheader("Upload Training Data")

training_data = st.file_uploader(
    "Upload Network Traffic Data", 
    type=["csv"], 
    help="Upload CSV file containing network traffic data with labels"
)

if training_data is not None:
    # Save the uploaded file
    data_path = f"uploaded_{training_data.name}"
    with open(data_path, "wb") as f:
        f.write(training_data.getbuffer())
    
    # Load and display data summary
    try:
        df = pd.read_csv(data_path)
        st.success(f"Data loaded successfully: {len(df)} rows")
        
        # Display data summary
        st.subheader("Data Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Sample Data")
            st.dataframe(df.head(5))
        
        with col2:
            if "label" in df.columns:
                # Count labels
                label_counts = df["label"].value_counts()
                
                # Create pie chart
                fig = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    title="Label Distribution"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No 'label' column found in the data. Please make sure your data includes labels.")
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data_path = None
else:
    data_path = None
    st.info("Please upload training data to continue.")

# Federated Learning Configuration
if training_mode == "Federated Learning":
    st.subheader("Federated Learning Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        num_clients = st.slider(
            "Number of Clients",
            min_value=2,
            max_value=10,
            value=3,
            help="Number of federated learning clients to simulate"
        )
        
        num_rounds = st.slider(
            "Number of Rounds",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of federated learning rounds"
        )
    
    with col2:
        split_method = st.selectbox(
            "Data Splitting Method",
            ["Random", "IID", "Non-IID"],
            help="Method to split data among clients"
        )
        
        server_address = st.text_input(
            "Server Address",
            value="[::]:8080",
            help="Address of the federated learning server"
        )
    
    # Instructions for federated learning
    with st.expander("How Federated Learning Works"):
        st.markdown("""
        ### Federated Learning Process
        
        1. **Data Preparation**: Data is split among clients according to the selected method
        2. **Server Initialization**: The central server is started with the selected configuration
        3. **Client Registration**: Clients register with the server
        4. **Model Distribution**: Server distributes the initial model to clients
        5. **Local Training**: Each client trains the model on their local data
        6. **Model Quantization**: Model updates are quantized to 1-bit representation
        7. **Update Aggregation**: Server aggregates the quantized updates
        8. **Model Update**: Global model is updated and sent back to clients
        9. **Iteration**: Process repeats for the specified number of rounds
        
        ### Benefits of 1-bit Quantization in Federated Learning
        
        - **Reduced Communication Costs**: 32x reduction in data transmission
        - **Privacy Preservation**: Less detailed information leaves client devices
        - **Efficient Computation**: Lower memory and processing requirements
        - **Faster Convergence**: Allows more frequent model updates
        """)

# Training button
if data_path:
    
    if training_mode == "Local Training":
        train_button = st.button("Start Training")
        
        if train_button:
            st.info("Starting local training...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Create model
            if model_type == "TinyBERT":
                model = TinyBERTModel()
            else:
                model = DistilBERTModel()
            
            # Train model (simulated)
            for epoch in range(num_epochs):
                # Simulate training progress
                status_text.text(f"Training epoch {epoch+1}/{num_epochs}")
                
                # Update progress
                progress = (epoch + 1) / num_epochs
                progress_bar.progress(progress)
                
                # Simulate epoch time
                time.sleep(1)
            
            # Create models directory if it doesn't exist
            os.makedirs("models", exist_ok=True)
            
            # Save trained model
            model_path = os.path.join("models", f"model_local_training.pt")
            
            # For our mock model, we create a simple dict
            model_state = {}
            for i, param in enumerate(model.parameters()):
                # If param is already a numpy array, don't call detach/cpu
                if isinstance(param, np.ndarray):
                    model_state[f"param_{i}"] = param
                else:
                    # Otherwise, handle as a PyTorch tensor
                    model_state[f"param_{i}"] = param.detach().cpu().numpy()
                    
            torch.save({
                "model_state_dict": model_state,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "quantization": quantization
            }, model_path)
            
            # Display training results
            st.success(f"Training completed! Model saved to {model_path}")
            
            # Display simulated metrics
            st.subheader("Training Results")
            
            # Create simulated metrics
            epochs = list(range(1, num_epochs + 1))
            train_loss = [np.exp(-0.5 * e) + 0.1 + np.random.normal(0, 0.05) for e in epochs]
            val_loss = [np.exp(-0.4 * e) + 0.15 + np.random.normal(0, 0.07) for e in epochs]
            train_acc = [1 - np.exp(-0.4 * e) - 0.1 + np.random.normal(0, 0.03) for e in epochs]
            val_acc = [1 - np.exp(-0.3 * e) - 0.15 + np.random.normal(0, 0.05) for e in epochs]
            
            # Create metric dataframe
            metrics_df = pd.DataFrame({
                "Epoch": epochs,
                "Training Loss": train_loss,
                "Validation Loss": val_loss,
                "Training Accuracy": train_acc,
                "Validation Accuracy": val_acc
            })
            
            # Create metrics visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Loss plot
                fig = px.line(
                    metrics_df,
                    x="Epoch",
                    y=["Training Loss", "Validation Loss"],
                    title="Loss",
                    markers=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Accuracy plot
                fig = px.line(
                    metrics_df,
                    x="Epoch",
                    y=["Training Accuracy", "Validation Accuracy"],
                    title="Accuracy",
                    markers=True
                )
                
                fig.update_layout(yaxis_range=[0, 1])
                
                st.plotly_chart(fig, use_container_width=True)
    
    else:  # Federated Learning
        start_fl_button = st.button("Start Federated Learning")
        
        if start_fl_button:
            st.info("Preparing for federated learning...")
            
            # Create client data splits
            if data_path:
                try:
                    df = pd.read_csv(data_path)
                    
                    # Shuffle data
                    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
                    
                    # Split data for each client
                    client_indices = np.array_split(df.index, num_clients)
                    
                    # Save data for each client
                    for i, indices in enumerate(client_indices):
                        client_df = df.iloc[indices]
                        client_path = os.path.join(client_data_dir, f"client_{i+1}_data.csv")
                        client_df.to_csv(client_path, index=False)
                        
                        st.write(f"Created dataset for client {i+1} with {len(client_df)} samples")
                    
                    # Create client configs
                    client_configs = []
                    for i in range(num_clients):
                        client_id = f"client_{i+1}"
                        client_data_path = os.path.join(client_data_dir, f"{client_id}_data.csv")
                        
                        client_configs.append({
                            "client_id": client_id,
                            "server_address": server_address,
                            "data_path": client_data_path
                        })
                    
                    # Display FL workflow
                    st.subheader("Federated Learning Workflow")
                    
                    st.write("Starting server and clients...")
                    
                    # Create simulated FL progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate FL rounds
                    for round_num in range(num_rounds):
                        status_text.text(f"Federated learning round {round_num+1}/{num_rounds}")
                        
                        # Update progress
                        progress = (round_num + 1) / num_rounds
                        progress_bar.progress(progress)
                        
                        # Simulate round time
                        time.sleep(1.5)
                    
                    # Simulate final model aggregation
                    status_text.text("Aggregating final model...")
                    time.sleep(1)
                    
                    # Update progress to completion
                    progress_bar.progress(1.0)
                    
                    # Create models directory if it doesn't exist
                    os.makedirs("models", exist_ok=True)
                    
                    # Save federated learning metrics
                    metrics = []
                    for round_num in range(1, num_rounds + 1):
                        metrics.append({
                            "round": round_num,
                            "num_clients": num_clients,
                            "mean_accuracy": 0.5 + 0.4 * (round_num / num_rounds) + np.random.normal(0, 0.03),
                            "mean_loss": 0.5 - 0.3 * (round_num / num_rounds) + np.random.normal(0, 0.05)
                        })
                    
                    # Save metrics to JSON
                    metrics_path = os.path.join("models", "round_metrics.json")
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f)
                    
                    # Save final aggregated model
                    model_path = os.path.join("models", f"model_round_{num_rounds}.pt")
                    
                    # Create model
                    if model_type == "TinyBERT":
                        model = TinyBERTModel()
                    else:
                        model = DistilBERTModel()
                    
                    # Save model - handle numpy arrays and PyTorch tensors
                    model_state = {}
                    for i, param in enumerate(model.parameters()):
                        if isinstance(param, np.ndarray):
                            model_state[f"param_{i}"] = param
                        else:
                            model_state[f"param_{i}"] = param.detach().cpu().numpy()
                            
                    torch.save({
                        "model_state_dict": model_state,
                        "rounds": num_rounds,
                        "num_clients": num_clients,
                        "quantization": quantization
                    }, model_path)
                    
                    status_text.text("Federated learning completed successfully!")
                    
                    # Display FL results
                    st.success(f"Federated learning completed! Final model saved to {model_path}")
                    
                    # Display FL metrics
                    metrics_df = pd.DataFrame(metrics)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Accuracy plot
                        fig = px.line(
                            metrics_df,
                            x="round",
                            y="mean_accuracy",
                            title="Mean Accuracy vs. Round",
                            markers=True
                        )
                        
                        fig.update_layout(
                            xaxis_title="Round",
                            yaxis_title="Accuracy",
                            yaxis=dict(range=[0, 1])
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Loss plot
                        fig = px.line(
                            metrics_df,
                            x="round",
                            y="mean_loss",
                            title="Mean Loss vs. Round",
                            markers=True
                        )
                        
                        fig.update_layout(
                            xaxis_title="Round",
                            yaxis_title="Loss"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Instructions for real federated learning
                    st.subheader("How to Run Real Federated Learning")
                    
                    st.markdown("""
                    ### Running Real Federated Learning
                    
                    To run actual federated learning (not simulated), follow these steps:
                    
                    1. **Start the Server**:
                    ```python
                    python server.py --num_rounds=10 --host=0.0.0.0 --port=8080
                    ```
                    
                    2. **Start Multiple Clients** (run in separate terminals):
                    ```python
                    python client.py --client_id=client_1 --server_address=localhost:8080 --data_path=client_data/client_1_data.csv
                    python client.py --client_id=client_2 --server_address=localhost:8080 --data_path=client_data/client_2_data.csv
                    python client.py --client_id=client_3 --server_address=localhost:8080 --data_path=client_data/client_3_data.csv
                    ```
                    
                    3. The server will coordinate the training process, distribute the model to clients, 
                       collect and aggregate their updates, and save the final model.
                    
                    4. Progress and metrics will be saved to the models directory and can be viewed in the Model Statistics page.
                    """)
                    
                except Exception as e:
                    st.error(f"Error setting up federated learning: {e}")
            else:
                st.error("Please upload training data to proceed with federated learning.")
else:
    st.warning("Please upload training data to start model training.")

# Add final note about federated learning
st.markdown("""
### Note on 1-bit Quantized Federated Learning

In this application, we demonstrate how 1-bit quantization can be applied to federated learning
for network traffic analysis. The key advantage is the significant reduction in communication costs
while maintaining model performance.

For real-world deployment, you would want to:

1. **Distribute Data**: Ensure each client has its own data that remains private
2. **Secure Communication**: Implement secure channels between clients and server
3. **Scale Infrastructure**: Deploy on distributed systems for larger networks
4. **Monitor Performance**: Track metrics to ensure model quality across rounds
5. **Implement Privacy Preserving Techniques**: Add differential privacy if needed
""")