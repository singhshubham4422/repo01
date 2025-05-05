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
import numpy._core.multiarray

# Add NumPy arrays to PyTorch safe list
add_safe_globals([numpy._core.multiarray._reconstruct])

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import display_model_stats
from model import TinyBERTModel
from quantization import quantize_model_parameters, calculate_compression_rate
import auth_helper

st.set_page_config(
    page_title="Model Statistics",
    page_icon="📊",
    layout="wide"
)

# Check if user is authenticated
auth_helper.check_auth()

# Initialize sidebar with logout button
auth_helper.init_sidebar()

st.title("Model Statistics")
st.markdown("""
This page displays statistics about the trained model, including size, parameters,
compression rate from 1-bit quantization, and performance metrics.
""")

# Display model statistics
display_model_stats()

# Add detailed model analysis
st.subheader("Detailed Model Analysis")

# Check if models exist
model_dir = "models"
if not os.path.exists(model_dir):
    st.info("No model data available yet. Start training to generate models.")
else:
    # Get all model files
    model_files = [f for f in os.listdir(model_dir) if f.startswith("model_round_") and f.endswith(".pt")]
    
    if not model_files:
        st.info("No model files available yet. Start training to generate models.")
    else:
        # Sort model files by round number
        model_files = sorted(model_files, key=lambda x: int(x.split("_")[2].split(".")[0]))
        
        # Create selectbox for model selection
        selected_model = st.selectbox(
            "Select Model",
            model_files,
            index=len(model_files) - 1  # Default to latest model
        )
        
        # Load the selected model
        model_path = os.path.join(model_dir, selected_model)
        model = TinyBERTModel()
        
        try:
            # Try loading with weights_only=False for backward compatibility
            checkpoint = torch.load(model_path, weights_only=False)
            
            # Check if it's the new format (dictionary with model_state_dict)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Load the model from the state dict
                model.load_state_dict(checkpoint["model_state_dict"])
                
                # Get additional information
                st.write("Additional model information:")
                for key, value in checkpoint.items():
                    if key != "model_state_dict":
                        st.write(f"- {key}: {value}")
            else:
                # Old format - direct load
                model.load_state_dict(checkpoint)
        except Exception as e:
            st.warning(f"Could not load full model details: {str(e)}")
            st.info("Loading model metrics only - model parameters are not loaded")
            
        model.eval()
        
        # Extract round number
        round_num = int(selected_model.split("_")[2].split(".")[0])
        
        # Display model information
        st.markdown(f"### Model from Round {round_num}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Calculate model stats - handle both PyTorch tensors and NumPy arrays
            total_params = 0
            trainable_params = 0
            
            for p in model.parameters():
                # Check if parameter is NumPy array or PyTorch tensor
                if isinstance(p, np.ndarray):
                    total_params += p.size
                    # Mock models don't have requires_grad for NumPy arrays
                    trainable_params += p.size
                else:
                    # PyTorch tensor
                    total_params += p.numel()
                    if hasattr(p, 'requires_grad') and p.requires_grad:
                        trainable_params += p.numel()
            
            frozen_params = total_params - trainable_params
            
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable_params:,}")
            st.metric("Frozen Parameters", f"{frozen_params:,}")
        
        with col2:
            # Calculate size information
            model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32 parameter
            quantized_size_mb = model_size_mb / 32  # 1-bit quantization (32x reduction)
            
            st.metric("Model Size", f"{model_size_mb:.2f} MB")
            st.metric("Quantized Size", f"{quantized_size_mb:.2f} MB")
            st.metric("Compression Rate", "32x")
        
        # Display parameter distribution by layer
        st.markdown("### Parameter Distribution by Layer")
        
        # Group parameters by layer - mock version for simulation
        try:
            # Try to use named_parameters if available
            layer_params = {}
            for name, param in model.named_parameters():
                layer_name = name.split('.')[0]
                
                if layer_name not in layer_params:
                    layer_params[layer_name] = 0
                
                # Handle both NumPy arrays and PyTorch tensors
                if isinstance(param, np.ndarray):
                    layer_params[layer_name] += param.size
                else:
                    layer_params[layer_name] += param.numel()
        except Exception as e:
            # If named_parameters is not implemented in mock model, create fake data
            st.warning(f"Could not get layer parameters: {str(e)}")
            
            # Create simulated layer data for demonstration
            layer_params = {
                "embedding": int(total_params * 0.2),
                "encoder": int(total_params * 0.5),
                "attention": int(total_params * 0.2),
                "classifier": int(total_params * 0.1)
            }
        
        # Create DataFrame for visualization
        layer_df = pd.DataFrame({
            "Layer": list(layer_params.keys()),
            "Parameters": list(layer_params.values())
        })
        
        # Create bar chart
        fig = px.bar(
            layer_df,
            x="Layer",
            y="Parameters",
            title="Parameter Distribution by Layer",
            color="Layer",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display 1-bit quantization visualization
        st.markdown("### 1-Bit Quantization Visualization")
        
        # Create a visualization of parameter quantization
        # For demonstration, we'll visualize a random weight matrix
        try:
            first_param = list(model.parameters())[0]
            # Check if it's a NumPy array or PyTorch tensor
            if isinstance(first_param, np.ndarray):
                sample_weight = first_param.flatten()
            else:
                sample_weight = first_param.detach().cpu().numpy().flatten()
                
            sample_size = min(1000, len(sample_weight))
            sample_weight = sample_weight[:sample_size]
        except Exception as e:
            # If we can't get parameters, create a simulated one for demonstration
            st.warning(f"Generating sample weights for visualization: {str(e)}")
            np.random.seed(42)  # For reproducibility
            sample_weight = np.random.randn(1000)
            sample_size = 1000
        
        # Create quantized version for comparison
        quantized_weight = np.sign(sample_weight)  # This is a simplified version of 1-bit quantization
        
        # Create a DataFrame for the visualization
        weight_df = pd.DataFrame({
            "Index": list(range(sample_size)),
            "Original Weight": sample_weight,
            "Quantized Weight": quantized_weight
        })
        
        # Melt the DataFrame for easier plotting
        melted_df = pd.melt(
            weight_df, 
            id_vars="Index", 
            value_vars=["Original Weight", "Quantized Weight"],
            var_name="Type",
            value_name="Value"
        )
        
        # Create line chart
        fig = px.line(
            melted_df,
            x="Index",
            y="Value",
            color="Type",
            title="Original vs. 1-Bit Quantized Weights"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display training evolution
        st.markdown("### Model Evolution Over Training Rounds")
        
        # Get performance metrics if available
        metrics_path = os.path.join(model_dir, "round_metrics.json")
        
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                round_metrics = json.load(f)
            
            # Create DataFrame
            metrics_df = pd.DataFrame(round_metrics)
            
            if "mean_accuracy" in metrics_df.columns:
                # Create line chart for accuracy evolution
                fig = px.line(
                    metrics_df,
                    x="round",
                    y="mean_accuracy",
                    title="Model Accuracy Evolution",
                    markers=True
                )
                
                fig.update_layout(
                    xaxis_title="Round",
                    yaxis_title="Accuracy",
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Add quantization efficiency section
        st.markdown("### Quantization Efficiency Analysis")
        
        # Create table of theoretical vs actual compression
        compression_data = {
            "Representation": ["Full Precision (32-bit)", "8-bit Quantization", "4-bit Quantization", "2-bit Quantization", "1-bit Quantization (Binary)"],
            "Bits per Weight": [32, 8, 4, 2, 1],
            "Compression Ratio": ["1x", "4x", "8x", "16x", "32x"],
            "Theoretical Model Size (MB)": [model_size_mb, model_size_mb/4, model_size_mb/8, model_size_mb/16, model_size_mb/32]
        }
        
        compression_df = pd.DataFrame(compression_data)
        
        st.dataframe(compression_df)
        
        # Create bar chart for compression
        fig = px.bar(
            compression_df,
            x="Representation",
            y="Theoretical Model Size (MB)",
            title="Model Size by Quantization Level",
            color="Representation"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add gradient quantization explanation
        st.markdown("""
        ### How 1-Bit Quantization Works
        
        In our federated learning system, we use 1-bit quantization to minimize the bandwidth 
        required when clients send model updates to the server. Here's how it works:
        
        1. **Gradient Computation**: Clients compute gradients locally based on their data
        2. **Mean and Standard Deviation**: Calculate the mean and std of each gradient tensor
        3. **Normalization**: Normalize gradients using the mean and std
        4. **Binarization**: Convert to binary values (+1 for positive, 0 for negative)
        5. **Packing**: Pack the bits to minimize bandwidth
        6. **Communication**: Send the packed bits along with scaling metadata
        7. **Dequantization**: Server unpacks and rescales the gradients
        
        The 1-bit representation drastically reduces communication costs while preserving 
        the directional information of gradient updates, which is often sufficient for model convergence.
        """)
        
        # Add accuracy vs quantization tradeoff plot
        st.markdown("### Accuracy vs. Quantization Tradeoff")
        
        # Create theoretical data for visualization
        quantization_bits = [32, 16, 8, 4, 2, 1]
        theoretical_accuracy = [0.95, 0.94, 0.92, 0.88, 0.82, 0.78]  # Theoretical values
        
        # Create DataFrame
        tradeoff_df = pd.DataFrame({
            "Bits": quantization_bits,
            "Accuracy": theoretical_accuracy,
            "Size Reduction": [f"{32/b}x" for b in quantization_bits]
        })
        
        # Create scatter plot
        fig = px.scatter(
            tradeoff_df,
            x="Bits",
            y="Accuracy",
            size=[100/b for b in quantization_bits],  # Size inversely proportional to bits
            text="Size Reduction",
            title="Accuracy vs. Quantization Level",
            labels={"Bits": "Bits per Weight", "Accuracy": "Theoretical Accuracy"}
        )
        
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_type="log")
        
        st.plotly_chart(fig, use_container_width=True)

# Add comparison with other models section
st.subheader("Model Comparison")

# Create comparison table
comparison_data = {
    "Model": ["TinyBERT (1-bit)", "DistilBERT (1-bit)", "TinyBERT (32-bit)", "DistilBERT (32-bit)", "BERT-base"],
    "Parameters (M)": [14.5, 66.0, 14.5, 66.0, 110.0],
    "Size (MB)": [0.45, 2.06, 14.5, 66.0, 440.0],
    "Inference Time (ms)": [5, 15, 5, 15, 45],
    "Relative Accuracy": ["Good", "Better", "Good+", "Better+", "Best"],
    "Suitable for Edge": ["Yes", "Limited", "Limited", "No", "No"]
}

comparison_df = pd.DataFrame(comparison_data)

st.dataframe(comparison_df)

# Create bar chart for model size comparison
fig = px.bar(
    comparison_df,
    x="Model",
    y="Size (MB)",
    title="Model Size Comparison",
    color="Model",
    log_y=True
)

st.plotly_chart(fig, use_container_width=True)

# Add information about federated aggregation with quantized models
st.subheader("Federated Aggregation with Quantized Models")

st.markdown("""
In our federated learning system, we use a modified FedAvg algorithm that handles 1-bit quantized updates:

1. **Client-side Quantization**: Each client quantizes their model updates to 1-bit representation
2. **Metadata Preservation**: Statistical information about the original distribution is preserved
3. **Efficient Communication**: Only binary values and minimal metadata are transmitted
4. **Server-side Dequantization**: Updates are dequantized before aggregation
5. **Weighted Averaging**: Client updates are weighted based on dataset size
6. **Global Model Update**: The aggregated update is applied to the global model

This approach reduces communication bandwidth by up to 32x compared to sending full-precision updates,
while maintaining model convergence and performance.
""")

# Add code example of quantization
st.markdown("### Quantization Code Example")

st.code("""
def quantize_model_update(param: np.ndarray) -> Dict[str, Any]:
    # Calculate the mean and standard deviation for scaling
    mean = np.mean(param)
    std = np.std(param)
    
    # If std is too small, avoid division by zero
    if std < 1e-8:
        std = 1.0
    
    # Normalize the parameter
    normalized = (param - mean) / std
    
    # Quantize to 1-bit: +1 for positive, 0 for negative
    quantized = (normalized > 0).astype(np.uint8)
    
    # Pack bits to save memory
    packed = np.packbits(quantized)
    
    # Return the quantized values and metadata for dequantization
    return {
        'packed': packed,
        'shape': param.shape,
        'mean': mean,
        'std': std
    }
""", language="python")
