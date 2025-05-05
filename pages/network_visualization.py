import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
import sys
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import random

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization import display_network_visualization
from data_processor import process_pcap_file
import auth_helper

st.set_page_config(
    page_title="Network Visualization",
    page_icon="🔍",
    layout="wide"
)

# Check if user is authenticated
auth_helper.check_auth()

# Initialize sidebar with logout button
auth_helper.init_sidebar()

st.title("Network Visualization")
st.markdown("""
This page provides visualizations of network traffic patterns, showing the relationships
between devices, traffic flows, and potential anomalies.
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
    
    # Display network visualization
    display_network_visualization(csv_path)
else:
    # Display sample network visualization
    st.info("Upload a PCAP or CSV file to visualize network traffic. Showing sample visualization for now.")
    display_network_visualization()

# Add visualization customization options
st.subheader("Visualization Options")

with st.expander("Customize Visualization"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        visualization_type = st.selectbox(
            "Visualization Type",
            ["Network Graph", "Connection Matrix", "Traffic Flow", "Hierarchical View"]
        )
    
    with col2:
        color_by = st.selectbox(
            "Color By",
            ["Protocol", "Traffic Volume", "Anomaly Score", "Device Type"]
        )
    
    with col3:
        layout_algo = st.selectbox(
            "Layout Algorithm",
            ["Force-directed", "Circular", "Hierarchical", "Spectral"]
        )
    
    min_edge_weight = st.slider(
        "Minimum Connection Strength",
        min_value=1,
        max_value=50,
        value=1
    )
    
    apply_settings = st.button("Apply Settings")
    
    if apply_settings:
        st.info("Settings applied. Regenerating visualization.")
        
        # This would normally update the visualization based on settings
        st.rerun()

# Add network topology section
st.subheader("Network Topology")

# Create a sample network topology if no file is uploaded
if uploaded_file is None:
    # Create sample network topology data
    nodes = [
        {"id": "router", "type": "router", "ip": "192.168.1.1"},
        {"id": "server1", "type": "server", "ip": "192.168.1.10"},
        {"id": "server2", "type": "server", "ip": "192.168.1.11"},
        {"id": "client1", "type": "client", "ip": "192.168.1.101"},
        {"id": "client2", "type": "client", "ip": "192.168.1.102"},
        {"id": "client3", "type": "client", "ip": "192.168.1.103"},
        {"id": "client4", "type": "client", "ip": "192.168.1.104"},
        {"id": "external", "type": "external", "ip": "10.0.0.1"}
    ]
    
    edges = [
        {"source": "client1", "target": "router", "weight": 15},
        {"source": "client2", "target": "router", "weight": 10},
        {"source": "client3", "target": "router", "weight": 8},
        {"source": "client4", "target": "router", "weight": 12},
        {"source": "router", "target": "server1", "weight": 25},
        {"source": "router", "target": "server2", "weight": 18},
        {"source": "router", "target": "external", "weight": 30},
        {"source": "server1", "target": "server2", "weight": 5}
    ]
    
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes
    for node in nodes:
        G.add_node(node["id"], type=node["type"], ip=node["ip"])
    
    # Add edges
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge["weight"])
    
    # Set node colors based on type
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]["type"] == "router":
            node_colors.append("#1E88E5")  # Blue
        elif G.nodes[node]["type"] == "server":
            node_colors.append("#43A047")  # Green
        elif G.nodes[node]["type"] == "client":
            node_colors.append("#FBC02D")  # Yellow
        else:
            node_colors.append("#E53935")  # Red
    
    # Set node sizes based on degree
    node_sizes = [100 + G.degree(node) * 50 for node in G.nodes()]
    
    # Create position layout
    if layout_algo == "Circular":
        pos = nx.circular_layout(G)
    elif layout_algo == "Hierarchical":
        pos = nx.multipartite_layout(G, subset_key="type")
    elif layout_algo == "Spectral":
        pos = nx.spectral_layout(G)
    else:  # Force-directed
        pos = nx.spring_layout(G, seed=42)
    
    # Create matplotlib figure
    plt.figure(figsize=(10, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, 
        pos, 
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.8
    )
    
    # Draw edges with width based on weight
    edge_widths = [G[u][v]["weight"] / 5 for u, v in G.edges()]
    nx.draw_networkx_edges(
        G, 
        pos, 
        width=edge_widths,
        alpha=0.5,
        edge_color="#555555"
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, 
        pos, 
        font_size=10, 
        font_family="sans-serif"
    )
    
    plt.axis("off")
    plt.tight_layout()
    
    # Display the graph
    st.pyplot(plt)
    
    # Create edge table
    edge_data = []
    for u, v, data in G.edges(data=True):
        edge_data.append({
            "Source": u,
            "Target": v,
            "Source IP": G.nodes[u]["ip"],
            "Target IP": G.nodes[v]["ip"],
            "Traffic Volume": data["weight"]
        })
    
    edge_df = pd.DataFrame(edge_data)
    
    st.dataframe(edge_df)

# Add time-based traffic patterns
st.subheader("Time-Based Traffic Patterns")

# Create sample time-based traffic data
if uploaded_file is None:
    # Generate 24 hours of data
    hours = list(range(24))
    
    # Create traffic volume for different device types
    client_traffic = [10 + 5 * h if h < 12 else 60 - 5 * h if h < 20 else 10 for h in hours]
    server_traffic = [5 + h * 2 if h < 8 else 40 - h if h < 16 else 20 - h / 2 if h < 20 else 5 for h in hours]
    external_traffic = [2 + h for h in hours]
    
    # Create DataFrame
    time_data = pd.DataFrame({
        "Hour": hours,
        "Client Traffic": client_traffic,
        "Server Traffic": server_traffic,
        "External Traffic": external_traffic
    })
    
    # Melt DataFrame for plotting
    melted_time_data = pd.melt(
        time_data,
        id_vars="Hour",
        value_vars=["Client Traffic", "Server Traffic", "External Traffic"],
        var_name="Traffic Type",
        value_name="Volume"
    )
    
    # Create line chart
    fig = px.line(
        melted_time_data,
        x="Hour",
        y="Volume",
        color="Traffic Type",
        title="Traffic Volume Over 24 Hours",
        markers=True
    )
    
    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Traffic Volume (packets/min)",
        xaxis=dict(tickmode="linear", tick0=0, dtick=2)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add anomaly heatmap
st.subheader("Traffic Anomaly Heatmap")

# Create sample anomaly heatmap data
if uploaded_file is None:
    # Create sample source and destination IPs
    src_ips = [f"192.168.1.{i}" for i in range(1, 11)]
    dst_ips = [f"10.0.0.{i}" for i in range(1, 11)]
    
    # Initialize matrix
    matrix = np.zeros((len(src_ips), len(dst_ips)))
    
    # Fill with random values
    for i in range(len(src_ips)):
        for j in range(len(dst_ips)):
            # Higher values for specific connections (simulating anomalies)
            if i == 3 and j in [0, 1, 2, 3, 4]:  # A host scanning multiple destinations
                matrix[i, j] = random.uniform(0.7, 0.9)
            elif i in [0, 1, 2, 3, 4] and j == 7:  # Multiple hosts targeting one destination
                matrix[i, j] = random.uniform(0.6, 0.8)
            else:
                matrix[i, j] = random.uniform(0, 0.3)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=dst_ips,
        y=src_ips,
        colorscale="Viridis",
        showscale=True,
        colorbar=dict(title="Anomaly Score")
    ))
    
    fig.update_layout(
        title="Network Traffic Anomaly Heatmap",
        xaxis_title="Destination IP",
        yaxis_title="Source IP",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add security insights section
st.subheader("Security Insights")

# Create sample security insights
if uploaded_file is None:
    st.markdown("""
    ### Detected Patterns
    
    Based on the network visualization, we can identify several patterns:
    
    1. **Possible Port Scan**: Client at 192.168.1.101 is connecting to multiple ports on Server 1
    2. **Unusual Traffic Volume**: High traffic between Router and External endpoint may indicate data exfiltration
    3. **Isolated Systems**: Server 2 has minimal connections, which is unusual for a server
    
    ### Recommended Actions
    
    1. Investigate client 192.168.1.101 for potential compromise
    2. Monitor external traffic patterns for data exfiltration
    3. Verify that Server 2 is operating correctly
    """)

# Add subnet visualization
st.subheader("Subnet Traffic Analysis")

# Create sample subnet data
if uploaded_file is None:
    # Create subnets
    subnets = ["192.168.1.0/24", "192.168.2.0/24", "10.0.0.0/24", "172.16.0.0/24"]
    
    # Create traffic matrix
    subnet_matrix = np.array([
        [10, 25, 15, 5],
        [20, 5, 30, 10],
        [15, 35, 8, 12],
        [5, 10, 15, 3]
    ])
    
    # Create chord diagram data
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=subnets,
            color=["#1E88E5", "#FBC02D", "#43A047", "#E53935"]
        ),
        link=dict(
            source=[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
            target=[1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
            value=[subnet_matrix[0, 1], subnet_matrix[0, 2], subnet_matrix[0, 3],
                  subnet_matrix[1, 0], subnet_matrix[1, 2], subnet_matrix[1, 3],
                  subnet_matrix[2, 0], subnet_matrix[2, 1], subnet_matrix[2, 3],
                  subnet_matrix[3, 0], subnet_matrix[3, 1], subnet_matrix[3, 2]]
        )
    ))
    
    fig.update_layout(
        title="Traffic Flow Between Subnets",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add protocol distribution section
st.subheader("Protocol Distribution By Host")

# Create sample protocol distribution data
if uploaded_file is None:
    # Create hosts
    hosts = ["192.168.1.101", "192.168.1.102", "192.168.1.103", "192.168.1.104", "192.168.1.10", "192.168.1.11"]
    
    # Create protocols
    protocols = ["TCP", "UDP", "ICMP", "HTTP", "DNS", "HTTPS"]
    
    # Create distribution matrix
    protocol_data = pd.DataFrame({
        "Host": np.repeat(hosts, len(protocols)),
        "Protocol": protocols * len(hosts),
        "Packets": [
            # Client 1
            150, 30, 10, 100, 20, 50,
            # Client 2
            80, 40, 5, 60, 30, 20,
            # Client 3
            120, 25, 15, 70, 40, 50,
            # Client 4
            90, 35, 8, 50, 25, 40,
            # Server 1
            200, 50, 5, 150, 10, 180,
            # Server 2
            180, 30, 3, 100, 120, 60
        ]
    })
    
    # Create grouped bar chart
    fig = px.bar(
        protocol_data,
        x="Host",
        y="Packets",
        color="Protocol",
        title="Protocol Distribution By Host",
        barmode="group"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Add interactive network exploration
st.subheader("Interactive Network Exploration")

st.markdown("""
Use the filters and options below to explore the network in more detail.
Select specific hosts, protocols, or time ranges to focus your analysis.
""")

# Create filters
col1, col2, col3 = st.columns(3)

with col1:
    host_filter = st.multiselect(
        "Filter by Host",
        ["192.168.1.101", "192.168.1.102", "192.168.1.103", "192.168.1.104", "192.168.1.10", "192.168.1.11", "All"],
        default=["All"]
    )

with col2:
    protocol_filter = st.multiselect(
        "Filter by Protocol",
        ["TCP", "UDP", "ICMP", "HTTP", "DNS", "HTTPS", "All"],
        default=["All"]
    )

with col3:
    time_range = st.select_slider(
        "Time Range",
        options=["Last hour", "Last 6 hours", "Last 12 hours", "Last day", "Last week", "All time"],
        value="All time"
    )

apply_filters = st.button("Apply Exploration Filters")

if apply_filters:
    st.info("Filters applied. (This would update the visualizations based on the selected filters)")

# Add network topology evolution
st.subheader("Network Topology Evolution")

st.markdown("""
This section shows how the network topology has evolved over time.
Track changes in connections, traffic patterns, and potential security incidents.
""")

# Create time series slider
time_point = st.slider(
    "Time Point",
    min_value=0,
    max_value=10,
    value=5,
    step=1,
    format="T%d"
)

# Display a placeholder evolution visualization
if uploaded_file is None:
    st.info(f"Showing network topology at time point T{time_point}")
    
    # Create a simple visualization based on time point
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create a grid of points
    n = 10
    m = 8
    positions = {}
    
    for i in range(n):
        for j in range(m):
            positions[(i, j)] = (i, j)
    
    # Create connections based on time point
    connections = []
    
    # More connections as time progresses
    num_connections = 10 + time_point * 5
    
    for _ in range(num_connections):
        i1, j1 = random.randint(0, n-1), random.randint(0, m-1)
        i2, j2 = random.randint(0, n-1), random.randint(0, m-1)
        
        if (i1, j1) != (i2, j2):
            connections.append(((i1, j1), (i2, j2)))
    
    # Draw connections
    for (i1, j1), (i2, j2) in connections:
        ax.plot([i1, i2], [j1, j2], 'k-', alpha=0.2)
    
    # Draw nodes
    for (i, j) in positions:
        # Color some nodes red to simulate anomalies
        if time_point > 5 and random.random() < 0.1:
            color = 'red'
            size = 100
        else:
            color = 'blue'
            size = 50
        
        ax.scatter(i, j, s=size, c=color, alpha=0.7)
    
    ax.set_title(f"Network Topology at Time T{time_point}")
    ax.set_xlim(-1, n)
    ax.set_ylim(-1, m)
    ax.axis('off')
    
    st.pyplot(fig)
    
    # Add explanation based on time point
    if time_point < 3:
        st.markdown("**Normal network activity with few connections.**")
    elif time_point < 7:
        st.markdown("**Increased network activity, more connections forming.**")
    else:
        st.markdown("**Potential anomalies detected (red nodes), indicating possible security incidents.**")

# Add documentation about visualization techniques
st.subheader("Network Visualization Guide")

st.markdown("""
### Understanding Network Visualizations

This dashboard uses several types of visualizations to represent network traffic:

1. **Network Graphs**: Shows connections between devices, with:
   - Nodes representing network devices (clients, servers, routers)
   - Edges representing connections between devices
   - Edge width indicating traffic volume
   - Node color indicating device type or security status

2. **Heatmaps**: Display traffic patterns between IP addresses, with:
   - Darker colors indicating higher traffic volume
   - Sudden patterns (vertical/horizontal lines) potentially indicating scanning or DoS

3. **Time Series**: Show traffic evolution over time, useful for:
   - Identifying unusual spikes in activity
   - Understanding normal usage patterns
   - Detecting after-hours activity

4. **Sankey Diagrams**: Visualize traffic flow between subnets, showing:
   - Overall traffic distribution
   - Imbalances in network communication
   - Potential data exfiltration paths

### Using These Visualizations for Security

- Look for **unexpected connections** in the network graph
- Check for **hotspots** in the heatmap that might indicate scanning or DoS
- Monitor **time-based patterns** for activity outside normal business hours
- Watch for **sudden changes** in topology or traffic volume
""")
