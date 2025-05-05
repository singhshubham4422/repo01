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

from visualization import display_learning_progress
import auth_helper

st.set_page_config(
    page_title="Learning Progress",
    page_icon="📊",
    layout="wide"
)

# Check if user is authenticated
auth_helper.check_auth()

# Initialize sidebar with logout button
auth_helper.init_sidebar()

st.title("Federated Learning Progress")
st.markdown("""
This page shows the progress of the federated learning process,
including model performance over multiple rounds.
""")

# Display learning progress
display_learning_progress()

# Add manual tracking section
st.subheader("Manual Performance Tracking")

with st.form("manual_tracking"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        round_num = st.number_input("Round Number", min_value=1, value=1)
    
    with col2:
        accuracy = st.number_input("Accuracy", min_value=0.0, max_value=1.0, value=0.8)
    
    with col3:
        loss = st.number_input("Loss", min_value=0.0, value=0.5)
    
    submitted = st.form_submit_button("Add Metrics")
    
    if submitted:
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Load existing metrics if any
        metrics_path = "models/round_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                metrics = json.load(f)
        else:
            metrics = []
        
        # Add new metrics
        new_metrics = {
            "round": int(round_num),
            "mean_accuracy": float(accuracy),
            "mean_loss": float(loss),
            "num_clients": 1
        }
        
        # Check if round already exists
        for i, m in enumerate(metrics):
            if m["round"] == new_metrics["round"]:
                metrics[i] = new_metrics
                break
        else:
            metrics.append(new_metrics)
        
        # Save metrics
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        
        st.success(f"Added metrics for round {round_num}!")
        st.rerun()

# Display federated learning parameters
st.subheader("Federated Learning Parameters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Current Configuration
    - **Number of Rounds**: 10
    - **Client Fraction**: 0.5 (50% of clients participate in each round)
    - **Minimum Fit Clients**: 2
    - **Minimum Evaluate Clients**: 2
    - **Minimum Available Clients**: 2
    """)

with col2:
    st.markdown("""
    ### Model Configuration
    - **Model Type**: TinyBERT
    - **Quantization**: 1-bit
    - **Optimizer**: Adam
    - **Learning Rate**: 0.001
    - **Local Epochs**: 1
    """)

# Display round details
st.subheader("Round Details")

# Create a form to select a round
round_num = st.selectbox("Select Round", list(range(1, 11)))

# Display dummy round data
st.markdown(f"### Round {round_num} Details")

col1, col2 = st.columns(2)

with col1:
    st.markdown(f"""
    **Participating Clients**: 3
    **Duration**: 245 seconds
    **Training Loss**: {0.5 - 0.03 * round_num:.3f}
    **Training Accuracy**: {0.7 + 0.02 * round_num:.3f}
    """)

with col2:
    st.markdown(f"""
    **Evaluation Loss**: {0.6 - 0.04 * round_num:.3f}
    **Evaluation Accuracy**: {0.65 + 0.025 * round_num:.3f}
    **Model Size**: 5.76 MB
    **Communication Cost**: {5.76 / 32:.2f} MB (1-bit quantized)
    """)

# Create a dummy convergence plot
rounds = list(range(1, round_num + 1))
accuracy = [0.65 + 0.025 * r for r in rounds]
loss = [0.6 - 0.04 * r for r in rounds]

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=rounds,
    y=accuracy,
    mode="lines+markers",
    name="Accuracy"
))

fig.add_trace(go.Scatter(
    x=rounds,
    y=loss,
    mode="lines+markers",
    name="Loss",
    yaxis="y2"
))

fig.update_layout(
    title=f"Convergence Plot (Rounds 1-{round_num})",
    xaxis_title="Round",
    yaxis_title="Accuracy",
    yaxis2=dict(
        title="Loss",
        overlaying="y",
        side="right"
    ),
    height=400,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)
