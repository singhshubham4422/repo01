import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import sys
import torch
import io
import base64
from torch.serialization import add_safe_globals

# Add NumPy arrays to PyTorch safe list
add_safe_globals([np.ndarray, np.core.multiarray._reconstruct])

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import TinyBERTModel
import auth_helper

st.set_page_config(
    page_title="Model Download",
    page_icon="⬇️",
    layout="wide"
)

# Check if user is authenticated
auth_helper.check_auth()

# Initialize sidebar with logout button
auth_helper.init_sidebar()

st.title("Model Download Center")
st.markdown("""
This page allows you to download trained models for deployment to edge devices.
Select a model to view its details and download it for external use.
""")

# Function to create a download link for a file
def get_download_link(file_path, link_text):
    with open(file_path, "rb") as f:
        file_data = f.read()
    b64_data = base64.b64encode(file_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64_data}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

# Check if models directory exists
model_dir = "models"
if not os.path.exists(model_dir):
    st.warning("No models available yet. Train models using the Model Training page.")
else:
    # Get all model files
    model_files = [f for f in os.listdir(model_dir) if (f.startswith("model_") and f.endswith(".pt"))]

    if not model_files:
        st.warning("No model files found. Train models using the Model Training page.")
    else:
        # Sort models by type and round number
        federated_models = [f for f in model_files if f.startswith("model_round_")]
        local_models = [f for f in model_files if f.startswith("model_local_")]

        sorted_federated = sorted(federated_models, key=lambda x: int(x.split("_")[2].split(".")[0]), reverse=True)

        # Create containers for different model types
        federated_container = st.container()
        local_container = st.container()

        # Add navigation buttons at the top
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Federated Learning Models", use_container_width=True):
                st.session_state.model_section = "federated"
        with col2:
            if st.button("Local Training Models", use_container_width=True):
                st.session_state.model_section = "local"

        # Initialize session state if it doesn't exist
        if "model_section" not in st.session_state:
            st.session_state.model_section = "federated"

        # Display appropriate section based on selection
        if st.session_state.model_section == "federated" and federated_models:
            with federated_container:
                st.subheader("Federated Learning Models")
                st.markdown("These models were trained using federated learning across multiple clients.")

                # Create a table of available models
                model_data = []
                for model_file in sorted_federated:
                    model_path = os.path.join(model_dir, model_file)
                    file_size_kb = os.path.getsize(model_path) / 1024

                    # Try to load model metadata
                    try:
                        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
                        rounds = checkpoint.get("rounds", "N/A") if isinstance(checkpoint, dict) else "N/A"
                        clients = checkpoint.get("num_clients", "N/A") if isinstance(checkpoint, dict) else "N/A"
                        quantized = "Yes" if (isinstance(checkpoint, dict) and checkpoint.get("quantization", False)) else "No"
                    except Exception as e:
                        rounds = model_file.split("_")[2].split(".")[0]
                        clients = "N/A"
                        quantized = "N/A"
                        print(f"Error loading model {model_file}: {str(e)}")

                    model_data.append({
                        "Model": model_file,
                        "Rounds": rounds,
                        "Clients": clients,
                        "Quantized": quantized,
                        "Size (KB)": f"{file_size_kb:.2f}"
                    })

                if model_data:
                    model_df = pd.DataFrame(model_data)
                    st.dataframe(model_df)

                    # Model selection for download
                    selected_model = st.selectbox(
                        "Select a model to download", 
                        sorted_federated,
                        index=0,
                        key="federated_model_select"
                    )

                    model_path = os.path.join(model_dir, selected_model)
                    st.markdown(get_download_link(model_path, "📥 Download Model"), unsafe_allow_html=True)
                else:
                    st.info("No federated models available yet.")

        elif st.session_state.model_section == "local" and local_models:
            with local_container:
                st.subheader("Local Training Models")
                st.markdown("These models were trained locally without federated learning.")

                # Display local models
                for model_file in local_models:
                    model_path = os.path.join(model_dir, model_file)
                    file_size_kb = os.path.getsize(model_path) / 1024
                    st.write(f"**{model_file}** (Size: {file_size_kb:.2f} KB)")
                    st.markdown(get_download_link(model_path, "📥 Download Model"), unsafe_allow_html=True)
        else:
            st.info("No models available in this section. Train some models first.")

        # Divider with active section highlighted
        st.markdown(f"""
        <style>
        .model-nav {{
            display: flex;
            margin-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }}
        .nav-item {{
            padding: 10px 20px;
            margin-right: 10px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
        }}
        .nav-item.active {{
            border-bottom: 3px solid #ff4b4b;
            font-weight: bold;
        }}
        </style>
        <div class="model-nav">
            <div class="nav-item {'active' if st.session_state.model_section == 'federated' else ''}">Federated Learning Models</div>
            <div class="nav-item {'active' if st.session_state.model_section == 'local' else ''}">Local Training Models</div>
        </div>
        """, unsafe_allow_html=True)


# Add deployment instructions
st.markdown("---")
st.subheader("Deployment Instructions")
st.markdown("""
### How to Deploy Models to Edge Devices

1. **Download the model file** (.pt) using the download button above
2. **Download the model info** (.md) for metadata and usage instructions
3. **Copy the model file** to your edge device (e.g., Raspberry Pi, IoT device)
4. **Install the necessary dependencies**:
   ```bash
   pip install torch numpy
   ```
5. **Implement the TinyBERTModel class** or use a compatible model architecture
6. **Load the model** using the code provided in the model info file
7. **Integrate with your application** for real-time traffic analysis
8. **Optimize for your hardware** if needed (quantization parameters can be adjusted)

### Benefits of 1-bit Quantized Models for Edge Deployment

- **Extremely small file size**: 32x smaller than full-precision models
- **Faster inference**: Reduced computation requirements
- **Lower memory footprint**: Can run on devices with limited RAM
- **Reduced power consumption**: Important for battery-powered devices
- **Privacy-preserving**: Trained via federated learning without sharing raw data

For additional assistance, check the documentation or contact support.
""")

# Add section for model conversion 
st.markdown("---")
st.subheader("Model Conversion Tools")
st.markdown("""
Need to convert your model to a different format for specific hardware? 
Select the target platform below to get conversion instructions:
""")

conversion_target = st.selectbox(
    "Target Platform",
    ["TensorFlow Lite", "ONNX", "TorchScript", "CoreML"],
    index=0
)

if conversion_target == "TensorFlow Lite":
    st.markdown("""
    ### Converting to TensorFlow Lite
    
    ```python
    import torch
    import tensorflow as tf
    import numpy as np
    
    # Load PyTorch model
    pytorch_model = YourModel()
    pytorch_model.load_state_dict(torch.load('model_path.pt')['model_state_dict'])
    pytorch_model.eval()
    
    # Create sample input
    sample_input = np.random.rand(1, input_size).astype(np.float32)
    
    # Export to ONNX first
    torch.onnx.export(pytorch_model, 
                     torch.from_numpy(sample_input),
                     "model.onnx",
                     export_params=True,
                     opset_version=12,
                     input_names=['input'],
                     output_names=['output'])
    
    # Convert ONNX to TF
    import tf2onnx
    import onnx
    
    onnx_model = onnx.load("model.onnx")
    tf_rep = tf2onnx.backend.prepare(onnx_model)
    tf_rep.export_graph("tf_model")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model("tf_model")
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
    ```
    """)
elif conversion_target == "ONNX":
    st.markdown("""
    ### Converting to ONNX
    
    ```python
    import torch
    
    # Load PyTorch model
    model = YourModel()
    model.load_state_dict(torch.load('model_path.pt')['model_state_dict'])
    model.eval()
    
    # Create dummy input with the correct shape
    dummy_input = torch.randn(1, input_size)
    
    # Export to ONNX
    torch.onnx.export(model,
                     dummy_input,
                     "model.onnx",
                     export_params=True,
                     opset_version=12,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}})
    
    # Verify the ONNX model
    import onnx
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    ```
    """)
elif conversion_target == "TorchScript":
    st.markdown("""
    ### Converting to TorchScript
    
    ```python
    import torch
    
    # Load PyTorch model
    model = YourModel()
    model.load_state_dict(torch.load('model_path.pt')['model_state_dict'])
    model.eval()
    
    # Create example input
    example_input = torch.rand(1, input_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save the TorchScript model
    traced_model.save("model_torchscript.pt")
    ```
    """)
elif conversion_target == "CoreML":
    st.markdown("""
    ### Converting to CoreML
    
    ```python
    import torch
    import coremltools as ct
    
    # Load PyTorch model
    model = YourModel() 
    model.load_state_dict(torch.load('model_path.pt')['model_state_dict'])
    model.eval()
    
    # Create example input
    example_input = torch.rand(1, input_size)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Convert to CoreML
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=example_input.shape)]
    )
    
    # Save the CoreML model
    mlmodel.save("model.mlmodel")
    ```
    """)