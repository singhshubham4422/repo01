import torch
import numpy as np
from typing import Dict, List, Union, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quantize_model_update(param: np.ndarray) -> Dict[str, Any]:
    """
    Quantize model parameters to 1-bit representation.
    
    Args:
        param: Model parameter array
        
    Returns:
        Dictionary containing the quantized parameters and metadata for dequantization
    """
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

def dequantize_model_update(quantized: Dict[str, Any]) -> np.ndarray:
    """
    Dequantize model parameters from 1-bit representation.
    
    Args:
        quantized: Dictionary containing the quantized parameters and metadata
        
    Returns:
        Dequantized parameter array
    """
    # Extract metadata
    packed = quantized['packed']
    shape = quantized['shape']
    mean = quantized['mean']
    std = quantized['std']
    
    # Unpack bits
    flat_size = np.prod(shape)
    unpacked = np.unpackbits(packed)[:flat_size]
    
    # Convert from {0, 1} to {-1, 1} to represent the sign
    binary = unpacked.astype(np.float32) * 2 - 1
    
    # Scale by standard deviation and add mean
    dequantized = binary * std + mean
    
    # Reshape to original shape
    dequantized = dequantized.reshape(shape)
    
    return dequantized

def quantize_gradients(gradients: List[torch.Tensor]) -> List[Dict[str, Any]]:
    """
    Quantize model gradients to 1-bit representation.
    
    Args:
        gradients: List of gradient tensors
        
    Returns:
        List of dictionaries containing the quantized gradients and metadata
    """
    quantized_gradients = []
    
    for grad in gradients:
        if grad is None:
            # Skip None gradients
            quantized_gradients.append(None)
            continue
        
        # Convert to numpy for quantization
        grad_np = grad.detach().cpu().numpy()
        
        # Quantize the gradient
        quantized = quantize_model_update(grad_np)
        
        # Append to list
        quantized_gradients.append(quantized)
    
    return quantized_gradients

def dequantize_gradients(quantized_gradients: List[Union[Dict[str, Any], None]]) -> List[torch.Tensor]:
    """
    Dequantize model gradients from 1-bit representation.
    
    Args:
        quantized_gradients: List of dictionaries containing quantized gradients
        
    Returns:
        List of dequantized gradient tensors
    """
    gradients = []
    
    for quantized in quantized_gradients:
        if quantized is None:
            # Keep None gradients as None
            gradients.append(None)
            continue
        
        # Dequantize the gradient
        dequantized_np = dequantize_model_update(quantized)
        
        # Convert back to torch tensor
        dequantized = torch.from_numpy(dequantized_np)
        
        # Append to list
        gradients.append(dequantized)
    
    return gradients

def calculate_compression_rate(original_size: int, quantized_size: int) -> float:
    """
    Calculate the compression rate.
    
    Args:
        original_size: Size of the original data in bytes
        quantized_size: Size of the quantized data in bytes
        
    Returns:
        Compression rate (original_size / quantized_size)
    """
    return original_size / max(quantized_size, 1)

def quantize_model_parameters(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Quantize all parameters of a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary containing quantized parameters and metadata
    """
    quantized_params = {}
    original_size = 0
    quantized_size = 0
    
    for name, param in model.named_parameters():
        # Skip parameters that don't require gradients
        if not param.requires_grad:
            continue
        
        # Calculate original size (assuming float32)
        param_size = param.numel() * 4  # 4 bytes per float32
        original_size += param_size
        
        # Convert to numpy for quantization
        param_np = param.detach().cpu().numpy()
        
        # Quantize the parameter
        quantized = quantize_model_update(param_np)
        
        # Calculate quantized size
        q_size = len(quantized['packed']) + 8 + 8  # packed + mean + std
        quantized_size += q_size
        
        # Store in dictionary
        quantized_params[name] = quantized
    
    # Calculate compression rate
    compression_rate = calculate_compression_rate(original_size, quantized_size)
    
    return {
        'parameters': quantized_params,
        'original_size': original_size,
        'quantized_size': quantized_size,
        'compression_rate': compression_rate
    }

def dequantize_model_parameters(model: torch.nn.Module, quantized_params: Dict[str, Any]) -> torch.nn.Module:
    """
    Dequantize parameters and load them into a PyTorch model.
    
    Args:
        model: PyTorch model
        quantized_params: Dictionary containing quantized parameters
        
    Returns:
        Model with dequantized parameters
    """
    # Get the quantized parameters
    parameters = quantized_params['parameters']
    
    # Create a new state dictionary
    state_dict = {}
    
    # Dequantize each parameter
    for name, quantized in parameters.items():
        dequantized_np = dequantize_model_update(quantized)
        dequantized = torch.from_numpy(dequantized_np)
        state_dict[name] = dequantized
    
    # Load missing parameters from the original model
    for name, param in model.named_parameters():
        if name not in state_dict:
            state_dict[name] = param
    
    # Load the state dictionary
    model.load_state_dict(state_dict, strict=False)
    
    return model
