import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten
# from tensorflow.keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten
from keras.models import Model
import time
import numpy as np

#### MANUAL DEFINITION OF UNET FUNCTIONS #####

# Maxpooling
def maxpool2d_manual(input_data, pool_size=(2, 2), stride=2):
    input_height, input_width, num_channels = input_data.shape
    pool_height, pool_width = pool_size
    
    # Calculation of output dimensions
    output_height = (input_height - pool_height) // stride + 1
    output_width = (input_width - pool_width) // stride + 1
    
    output = np.zeros((output_height, output_width, num_channels))
    
    for i in range(0, output_height):
        for j in range(0, output_width):
            for c in range(num_channels):
                region = input_data[i*stride:i*stride+pool_height, j*stride:j*stride+pool_width, c]
                output[i, j, c] = np.max(region)
    return output

# Activation function - ReLu
def relu_manual(input_data):
    return np.maximum(input_data, 0)

# Padding
def add_padding(input_data, kernel_height, kernel_width):
    pad_h = (kernel_height - 1) // 2  # Padding in height
    pad_w = (kernel_width - 1) // 2   # Padding in width
    padded_input = np.pad(input_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    
    return padded_input

# Convolution
def manual_conv2D(input_data, kernel, bias, padding):
    stride = 1
    
    # Batch handling: reduce input to 3 dimensions if it has a batch
    if len(input_data.shape) == 4:
        input_data = input_data.reshape(input_data.shape[1:])
    
    input_height, input_width, input_channels = input_data.shape
    kernel_height, kernel_width, kernel_channels, num_filters = kernel.shape

    # Compatibility check between kernel and input dimensions
    if input_channels != kernel_channels:
        raise ValueError(f"The input channels ({input_channels}) and the kernel channels ({kernel_channels}) must match")
    
    # Add padding if required
    if padding == 'same':
        input_data = add_padding(input_data, kernel_height, kernel_width)
        input_height, input_width, _ = input_data.shape  # Update dimensions after padding

    # Compute output dimensions
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1
    
    # Create output with correct size
    output = np.zeros((output_height, output_width, num_filters))

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            for f in range(num_filters):  # Iterate over filters
                sum_ = 0
                for c in range(input_channels):  # Iterate over channels
                    for di in range(kernel_height):
                        for dj in range(kernel_width):
                            sum_ += input_data[i + di, j + dj, c] * kernel[di, dj, c, f]
                sum_ += bias[f]
                output[i, j, f] = sum_

    return output


# Upsampling
def manual_upsampling(input_data):
    if len(input_data.shape) == 4:  # Batch handling
        input_data = input_data.reshape(input_data.shape[1:])
        input_height, input_width, input_channels = input_data.shape
    elif len(input_data.shape) == 3:  # No batch
        input_height, input_width, input_channels = input_data.shape
    else:
        raise ValueError("The input input_data must have 3 or 4 dimensions")
       
    # Output dimensions
    output_height = input_height * 2
    output_width = input_width * 2
    
    output_data = np.zeros((output_height, output_width, input_channels))
    
    for i in range(0, input_height):
        for j in range(0, input_width):
            for c in range(input_channels):
                output_data[i*2:i*2+2, j*2:j*2+2, c] = input_data[i, j, c]
    return output_data

# Concatenate
def manual_concatenate(tensor1, tensor2, axis=-1):

    # Check that height and width are equal
    if not (tensor1.shape[0] == tensor2.shape[0] and tensor1.shape[1] == tensor2.shape[1]):
        raise ValueError("Error 1 concatenation")

    # Manual concatenation along the channel axis
    if axis == -1 or axis == 3:  # channels
        merged = np.zeros((tensor1.shape[0], tensor1.shape[1], tensor1.shape[2] + tensor2.shape[2])) # Same height, same width, summed number of channels
        merged[:, :, :tensor1.shape[2]] = tensor1
        merged[:, :, tensor1.shape[2]:] = tensor2
        return merged
    else:
        raise ValueError("Error 2 concatenation")

    
# Activation function - Sigmoid
def manual_sigmoid(tensor_input):
    tensor_output = 1 / (1 + np.exp(-tensor_input))
    return tensor_output
    
