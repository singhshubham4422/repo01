import streamlit as st
import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add the parent directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from visualization import (
    display_client_monitoring,
    display_learning_progress,
    display_network_visualization,
    display_model_stats
)
import login

st.set_page_config(
    page_title="Federated Learning Network Intelligence",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check if user is logged in
if not login.check_login():
    login.show_login_page()
else:
    # Show main application content if logged in
    st.title(f"Federated Learning Network Intelligence - Welcome {st.session_state.username}!")
    st.markdown("""
    This application demonstrates how 1-bit quantized Language Models (LLMs) can be integrated with 
    Federated Learning (FL) to create a secure, efficient system for collaborative network intelligence sharing.
    """)

# Only show sidebar content if user is logged in
if login.check_login():
    with st.sidebar:
        st.title("Navigation")
        st.markdown("### Main Dashboard")
        
        st.markdown("### Other Pages")
        st.markdown("- [Client Monitor](/client_monitor)")
        st.markdown("- [Learning Progress](/learning_progress)")
        st.markdown("- [Traffic Analysis](/traffic_analysis)")
        st.markdown("- [Prediction](/prediction)")
        st.markdown("- [Model Statistics](/model_statistics)")
        st.markdown("- [Network Visualization](/network_visualization)")
        
        st.markdown("---")
        
        if st.button("Start Federated Learning Server"):
            st.session_state.server_running = True
            st.success("Server started! Clients can now connect.")
        
        if st.button("Stop Federated Learning Server"):
            st.session_state.server_running = False
            st.error("Server stopped!")
            
        # Add logout button
        st.markdown("---")
        login.show_logout_button()

# Only show main dashboard content if user is logged in
if login.check_login():
    # Main dashboard with overview of all components
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Client Participation")
        display_client_monitoring()
        
        st.subheader("Model Statistics")
        display_model_stats()
    
    with col2:
        st.subheader("Learning Progress")
        display_learning_progress()
        
        st.subheader("Traffic Analysis Sample")
        display_network_visualization()
    
    st.subheader("System Architecture")
    st.markdown("""
    ### How the System Works
    
    1. **Network Traffic Simulation**: Using Mininet to generate realistic network traffic
    2. **Data Processing**: PCAP to CSV conversion via tshark
    3. **Federated Learning**: Clients train models locally on their data
    4. **1-Bit Quantization**: Gradients are compressed to 1-bit for efficient communication
    5. **Global Model Aggregation**: Server aggregates model updates from clients
    6. **Model Evaluation**: Performance tracking across federated rounds
    
    ### Key Components:
    - **Federated Learning Framework**: Flower
    - **Model**: TinyBERT/DistilBERT (quantized to 1-bit)
    - **Network Simulation**: Mininet
    - **Data Processing**: tshark, custom tokenization
    - **Visualization**: Streamlit, Plotly
    """)

# Display a diagram of the system architecture (only if user is logged in)
if login.check_login():
    architecture_fig = go.Figure()
    
    # Define the components
    components = [
        {"name": "Client 1", "x": 1, "y": 1, "type": "client"},
        {"name": "Client 2", "x": 1, "y": 2, "type": "client"}, 
        {"name": "Client 3", "x": 1, "y": 3, "type": "client"},
        {"name": "Server", "x": 3, "y": 2, "type": "server"},
        {"name": "Dashboard", "x": 5, "y": 2, "type": "dashboard"}
    ]
    
    # Add nodes
    for comp in components:
        color = "#1E88E5" if comp["type"] == "client" else "#FF5722" if comp["type"] == "server" else "#4CAF50"
        architecture_fig.add_trace(go.Scatter(
            x=[comp["x"]], 
            y=[comp["y"]],
            mode="markers+text",
            marker=dict(size=30, color=color),
            text=[comp["name"]],
            textposition="bottom center",
            name=comp["name"]
        ))
    
    # Add edges (connections)
    for client in [c for c in components if c["type"] == "client"]:
        server = next(c for c in components if c["type"] == "server")
        architecture_fig.add_trace(go.Scatter(
            x=[client["x"], server["x"]],
            y=[client["y"], server["y"]],
            mode="lines",
            line=dict(width=2, color="#9E9E9E"),
            showlegend=False
        ))
    
    server = next(c for c in components if c["type"] == "server")
    dashboard = next(c for c in components if c["type"] == "dashboard")
    architecture_fig.add_trace(go.Scatter(
        x=[server["x"], dashboard["x"]],
        y=[server["y"], dashboard["y"]],
        mode="lines",
        line=dict(width=2, color="#9E9E9E"),
        showlegend=False
    ))
    
    # Update layout
    architecture_fig.update_layout(
        title="System Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(architecture_fig, use_container_width=True)
