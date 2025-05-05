
# Federated Learning Network Intelligence Application

A comprehensive web application for network traffic analysis using federated learning with 1-bit quantized models.

## Overview

This application demonstrates how 1-bit quantized Language Models (LLMs) can be integrated with Federated Learning (FL) to create a secure, efficient system for collaborative network intelligence sharing. It provides real-time network monitoring, federated model training, and traffic pattern visualization.

## Features

### 1. Authentication System
- Secure login/logout functionality
- User session management
- Role-based access control

### 2. Dashboard
- Real-time network statistics
- Traffic visualization
- Client monitoring
- Model performance tracking

### 3. Model Training
- Support for TinyBERT and DistilBERT models
- 1-bit quantization for efficient training
- Local and federated learning options
- Customizable training parameters

### 4. Network Traffic Analysis
- Real-time traffic monitoring
- Anomaly detection
- Protocol analysis
- Connection tracking
- Traffic pattern visualization

### 5. Model Management
- Model download center
- Performance comparison
- Conversion tools for different platforms
- Deployment guides

### 6. Federated Learning
- Multi-client support
- Secure model aggregation
- Progress tracking
- Client participation monitoring

## Getting Started

### Requirements
```
flwr>=1.18.0
matplotlib>=3.10.1 
networkx>=3.4.2
numpy>=2.2.5
pandas>=2.2.3
plotly>=6.0.1
scikit-learn>=1.6.1
scipy>=1.15.2
streamlit>=1.44.1
torch>=2.7.0
```

### Local Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-directory>
   ```

2. **Install Dependencies**
   ```bash
   # Install required packages
   pip install -r requirements.txt
   ```

3. **Prepare the Environment**
   - Create necessary directories:
   ```bash
   mkdir -p client_data models
   ```
   - Ensure the users.json file exists with initial admin credentials
   - Configure client data in client_data/ directory
   - Place any pre-trained models in models/ directory

4. **Starting the Application**
   ```bash
   # Run the Streamlit server
   streamlit run app.py --server.address 0.0.0.0 --server.port 5000
   ```

5. **Access the Application**
   - Open your web browser
   - Navigate to http://localhost:5000
   - Login with your credentials

### Project Structure
```
├── app.py                  # Main application file
├── auth.py                 # Authentication module
├── client.py              # Federated learning client
├── data_processor.py      # Data processing utilities
├── model.py               # ML model definitions
├── server.py              # Federated learning server
├── visualization.py       # Visualization components
├── client_data/           # Client training data
├── models/                # Saved models directory
└── pages/                 # Streamlit pages
```

3. **Authentication**
   - Access the application through your web browser
   - Log in using provided credentials
   - First-time users can register a new account

## Components

### Server
- Manages federated learning process
- Aggregates model updates
- Coordinates client training
- Saves and distributes models

### Clients
- Process local data
- Train models locally
- Send quantized updates
- Maintain data privacy

### Web Interface
- Real-time monitoring
- Configuration management
- Visualization tools
- Model management

## Key Technologies

- **Frontend**: Streamlit
- **ML Framework**: PyTorch
- **Federated Learning**: Flower
- **Visualization**: Plotly
- **Data Processing**: Pandas, NumPy
- **Network Analysis**: NetworkX

## Usage Guide

### 1. Model Training
- Navigate to "Model Training" page
- Select training mode (Local/Federated)
- Configure parameters
- Upload training data
- Start training

### 2. Traffic Analysis
- Go to "Traffic Analysis" page
- Upload PCAP/CSV files
- View real-time analysis
- Apply filters
- Export results

### 3. Model Download
- Visit "Model Download" page
- Select trained model
- View performance metrics
- Download for deployment

### 4. Client Monitoring
- Check "Client Monitor" page
- View active clients
- Monitor training progress
- Track resource usage

## Architecture

The application follows a distributed architecture:

```
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Client 1 │     │ Client 2 │     │ Client 3 │
└────┬─────┘     └────┬─────┘     └────┬─────┘
     │               │                  │
     └───────────────┼──────────────────┘
                     │
              ┌──────┴───────┐
              │   Server     │
              └──────┬───────┘
                     │
              ┌──────┴───────┐
              │  Dashboard   │
              └──────────────┘
```

## Security Features

- Secure authentication
- Encrypted communications
- Private data handling
- Role-based access

## Performance Optimization

- 1-bit quantization reduces communication costs by 32x
- Efficient model aggregation
- Optimized data processing
- Streamlined visualization

## Troubleshooting

Common issues and solutions:

1. **Connection Issues**
   - Check server address/port
   - Verify network connectivity
   - Ensure clients are properly configured

2. **Training Errors**
   - Verify data format
   - Check model parameters
   - Monitor system resources

3. **Visualization Problems**
   - Clear browser cache
   - Update dependencies
   - Check data format

## Future Enhancements

1. Additional model architectures
2. Enhanced visualization tools
3. Advanced anomaly detection
4. Improved client management
5. Extended protocol support

## Best Practices

1. Regular model updates
2. Consistent data formatting
3. Resource monitoring
4. Security maintenance
5. Performance tracking

## System Requirements

- Python 3.8+
- 4GB RAM minimum
- Modern web browser
- Network connectivity
