# FIRMWARE TESTING SCRIPT
# This script tests a manual firmware implementation against a Keras model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import matplotlib.pyplot as plt
import time

# Tune for rescaling function   
TUNE = 12
# Number of tests for extended statistical test
N = 200

# Manual inputs, weights, bias 
# Set seed for reproducibility
# Input is a 8x8 image with 2 channels, kernel is 3x3 with 2 input channels and 2 output channels, bias is a single integer
np.random.seed(1)
input_float = np.random.randint(-126, 128, size=(1, 8, 8, 2), dtype=np.int32)
np.random.seed(2)
manual_kernel_float_1 = np.random.rand(3, 3, 2, 2).astype(np.int32)
np.random.seed(3)
manual_bias_float_1 =  np.random.randint(-4096, 4097, dtype=np.int32)
np.random.seed(4)
manual_dense_weights = np.random.randint(-10, 10, size=(32, 1)).astype(np.float32)
np.random.seed(5)
manual_dense_bias = np.random.randint(-4096, 4097, dtype=np.int32)


#### MANUAL FUNCTIONS FOR SIMULATED QUANTIZED NETWORK ###

# Manual 2D convolution
def manual_conv2D(input_data, kernel, bias, padding):
    stride = 1
    # Batch index handling: reduce the indeces to 3 if there is batch index (removing it)
    if len(input_data.shape) == 4:
        input_data = input_data.reshape(input_data.shape[1:])
    # Take shape of input and kernel
    input_height, input_width, input_channels = input_data.shape
    kernel_height, kernel_width, kernel_channels, num_filters = kernel.shape
    
    # Check compatibility between input and kernel number of channels
    if input_channels != kernel_channels:
        raise ValueError(f"Input channels ({input_channels}) and kernel channels ({kernel_channels}) are not the same")

    # Add padding if requested and update input shape
    if padding == 'same':
        input_data = add_padding(input_data, kernel_height, kernel_width)
        input_height, input_width, _ = input_data.shape

    # Output shape calculation
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1
    # Create output with correct shape and int 16 bit
    output = np.zeros((output_height, output_width, num_filters), dtype=np.int32)

    # Manual convolution
    for i in range(output_height):
        for j in range(output_width):
            for f in range(num_filters):  
                sum_ = 0
                for c in range(input_channels): 
                    for di in range(kernel_height):
                        for dj in range(kernel_width):
                            sum_ += input_data[i + di, j + dj, c] * kernel[di, dj, c, f]
                sum_ += bias
                output[i, j, f] = sum_
    return output

# Manual 2D maxpooling function without numpy functions to simulate firmware VHDL package
def maxpool2d_manual(input_data, pool_size=(2, 2), stride=2):
    input_height, input_width, num_channels = input_data.shape
    pool_height, pool_width = pool_size
    # Output shape calculation
    output_height = (input_height - pool_height) // stride + 1
    output_width = (input_width - pool_width) // stride + 1
    output = np.zeros((output_height, output_width, num_channels), dtype=np.int32)

    # Manual maxpooling
    for i in range(output_height):
        for j in range(output_width):
            for c in range(num_channels):
                max_v = -32767
                for di in range(i*stride, i*stride + pool_height):
                    for dj in range(j*stride, j*stride + pool_width):
                        val = input_data[di, dj, c]
                        if val > max_v:
                            max_v = val
                output[i, j, c] = max_v
    return output

# Manual padding function that doesn't use the numpy.pad function
def add_padding(input_data, kernel_height, kernel_width):
    has_batch = False
    
    # Check if input data has batch size
    if len(input_data.shape) == 4:
        has_batch = True
        batch, input_data_width, input_data_height, input_data_channels = input_data.shape
    else:
        input_data_width, input_data_height, input_data_channels = input_data.shape
        
    # Padding in height dimension
    pad_h = (kernel_height - 1) // 2
    # Paddding in width dimension
    pad_w = (kernel_width - 1) // 2
    
    if has_batch == True:
        output_data = np.zeros((batch, input_data_height + 2 * pad_h, input_data_width + 2 * pad_w, input_data_channels), dtype=np.float32)
        for b in range(batch):
            for c in range(input_data_channels):
                for i in range(pad_h, input_data_height + pad_h):
                    for j in range(pad_w, input_data_width + pad_w):
                        output_data[b, i, j, c] = input_data[b, i - pad_h, j - pad_w, c]
    else: 
        output_data = np.zeros((input_data_height + 2 * pad_h, input_data_width + 2 * pad_w, input_data_channels), dtype=np.float32)
        for c in range(input_data_channels):
            for i in range(pad_h, input_data_height + pad_h):
                for j in range(pad_w, input_data_width + pad_w):
                    output_data[i, j, c] = input_data[i - pad_h, j - pad_w, c]
    padded_input = output_data
    return padded_input

# Manual ReLu to not use numpy.maximum function
def relu_manual(input_data):
    if input_data > 0:
        return input_data
    else:
        return 0

# Manual function to find max absolute value of input function without using numpy functions
def findmax(input_data):
    max_val = 0
    for x in input_data.flat:
        if x>=0:
            abs_val = x
        else:
            abs_val = -x
        if abs_val > max_val:
            max_val = abs_val
    return max_val

# Manual rescaling function, firmware-like
def rescale(input_float):
    # Find maximum absolute value
    max_input_value = findmax(input_float)
    input_shape = input_float.shape
    # Create rescaled output
    input_rescaled = np.zeros(input_shape,dtype=np.int16)
    # Add 12 bits at the end, equivalent to multiplying by 2^TUNE, here 4096
    f_1 = 127 << TUNE
    # Divide by maximum, find scale factor multiplied by 2^TUNE
    f_s = f_1 // max_input_value 
    input_multiplied = (input_float * f_s).astype(np.int64)
    # Remove extra bits to approximate
    input_rescaled = (input_multiplied >> TUNE).astype(np.int16)
    return input_rescaled, f_s


#### KERAS MODEL ####

# NN simulation: not quantized but rescaled between maxpooling and dense layers 
# Simulates firmware implementation

# First Keras model with convolution, relu, padding, maxpooling
model = Sequential([
    Conv2D(2, (3, 3), activation='relu', input_shape=(8, 8, 2), use_bias=True, padding='same'), 
    MaxPooling2D((2, 2))
])

# Set weights and bias
conv_layer_1 = model.layers[0]
conv_layer_1.set_weights([manual_kernel_float_1, np.array([manual_bias_float_1, manual_bias_float_1])])

# Start timing
start = time.time()

# First output
output_model = model.predict(input_float)
# Rescale 
output_model_rescaled, f_scale = rescale(output_model)

# Second Keras model with dense layer
model2 = Sequential([
    Flatten(input_shape=output_model_rescaled.shape[1:]),                         
    Dense(1, activation='linear')
])
# Set weights and bias for the dense layer
conv_layer_2 = model2.layers[1]  
conv_layer_2.set_weights([manual_dense_weights, np.array([manual_dense_bias])])

# Output of the Keras model
output_model_2 = model2.predict(output_model_rescaled)
# Time elapsed in firmware simulation
end = time.time()
timing = end - start

print("\nOutput Keras:", output_model_2.flatten())
print(f"Generated in {timing:.4f} s")


#### MANUAL FUNCTION-BASED SIMULATION OF FIRMWARE ####

# Start timing
start_manual = time.time()

# Convolution
conv_out = manual_conv2D(rescale(input_float)[0], manual_kernel_float_1, manual_bias_float_1, padding='same')

# Apply ReLU manually
relu_out = np.zeros_like(conv_out)
for i in range(conv_out.shape[0]):
    for j in range(conv_out.shape[1]):
        for c in range(conv_out.shape[2]):
            relu_out[i, j, c] = relu_manual(conv_out[i, j, c])

# Maxpooling manually
pool_out = maxpool2d_manual(relu_out)

# Rescale manually
pool_out_rescaled, scale_dense = rescale(pool_out)

# Flatten for dense layer
flat_input = pool_out_rescaled.flatten()

# Dense
dense_sum = 0
for idx in range(len(flat_input)):
    dense_sum += flat_input[idx] * manual_dense_weights[idx]
dense_sum += manual_dense_bias

# Output of the manual simulation
output_manual = dense_sum
# End timing
end_manual = time.time()
timing_manual = end_manual - start_manual

print("\nOutput Manual:", output_manual)
print(f"Generated in {timing_manual:.4f} s")

#### SIMULATION COMPARISON ####
keras_value = output_model_2.flatten()[0]
manual_value = output_manual

# Compute percentage deviation
percent_deviation = 100 * abs(manual_value - keras_value) / abs(keras_value) if keras_value != 0 else 0

print(f"\nPercentage Deviation: {float(percent_deviation):.2f}%")


#### EXTENDED STATISTICAL TEST ####

# Perform the same test over N random inputs
deviations = []

for test in range(N):
    input_rand = np.random.randint(-126, 128, size=(1, 8, 8, 2), dtype=np.int32)

    # Keras
    output_model = model.predict(input_rand)
    output_model_rescaled, _ = rescale(output_model)
    output_model_2 = model2.predict(output_model_rescaled)
    keras_value = output_model_2.flatten()[0]

    # Manual
    conv_out = manual_conv2D(rescale(input_rand)[0], manual_kernel_float_1, manual_bias_float_1, padding='same')
    relu_out = np.maximum(conv_out, 0)
    pool_out = maxpool2d_manual(relu_out)
    pool_out_rescaled, _ = rescale(pool_out)
    flat_input = pool_out_rescaled.flatten()
    dense_sum = sum(flat_input[i] * manual_dense_weights[i] for i in range(len(flat_input)))
    dense_sum += manual_dense_bias
    manual_value = dense_sum

    # Compute deviation
    if keras_value != 0:
        deviation = 100 * abs(manual_value - keras_value) / abs(keras_value)
    else:
        deviation = 0

    deviations.append(deviation)

# Statistics
deviations = np.array(deviations)
avg_dev = np.mean(deviations)
median_dev = np.median(deviations)
min_dev = np.min(deviations)
max_dev = np.max(deviations)

print("\nDeviation statistics in extended test")
print(f"Average deviation: {avg_dev:.2f}%")
print(f"Median deviation: {median_dev:.2f}%")
print(f"Min deviation: {min_dev:.2f}%")
print(f"Max deviation: {max_dev:.2f}%")

# Plot

plt.figure(figsize=(8, 5))
plt.hist(deviations, bins=30, color='blue', edgecolor='black', alpha=0.7)
plt.title("Deviation (%) distribution\nManual firmware vs Keras")
plt.xlabel("Deviation (%)")
plt.ylabel("Frequency")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

#### DIVISION LIBRARY GENERATION ####
# Generation of library for the division in the firmware (LUT)
# Numbers are represented as hexadecimals strings, all with 8 digits
# 127 * 4096 // (all numbers from 0 to 360000)
# Separate with commas, end with semicolon
# This is a library for the division operation in the firmware, to avoid using division in the VHDL code
with open("division_library.txt", "w") as file:
    for i in range(1, 360000):
        j = (127 * 4096) // i
        k = np.base_repr(j, base=16, padding=0)
        while len(k) < 8: 
            k = '0' + k
        file.write(k)
        if i < 359999:
            file.write(",\n")       
    file.write(";")
print("division_library.txt generated")