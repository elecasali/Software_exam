import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import time
from utils.utils import * # contains funtions manually defined

# Total number of bits to use
B_W = 8  

#### NETWORK WITH MULTIPLE FILTERS, THEN QUANTIZED ####

# Input data
np.random.seed(1)
input_data = np.random.rand(1, 64, 64, 1).astype(np.float32)  # Example input image 64x64
input_float = input_data / np.max(input_data)

# Kernel and biases with random seeds for reproducibility
np.random.seed(1)
manual_kernel_11 = np.random.rand(3, 3, 1, 8).astype(np.float32)
manual_kernel_float_1 = manual_kernel_11 / np.max(manual_kernel_11)
manual_bias_float_1 = np.zeros(8)

np.random.seed(2)
manual_kernel_22 = np.random.rand(3, 3, 8, 16).astype(np.float32)
manual_kernel_float_2 = manual_kernel_22 / np.max(manual_kernel_22)
manual_bias_float_2 = np.zeros(16)

np.random.seed(3)
manual_kernel_33 = np.random.rand(3, 3, 16, 32).astype(np.float32)
manual_kernel_float_3 = manual_kernel_33 / np.max(manual_kernel_33)
manual_bias_float_3 = np.zeros(32)

np.random.seed(4)
manual_kernel_44 = np.random.rand(3, 3, 32, 64).astype(np.float32)
manual_kernel_float_4 = manual_kernel_44 / np.max(manual_kernel_44)
manual_bias_float_4 = np.zeros(64)

np.random.seed(5)
manual_dense_weights1 = np.random.rand(1024, 1).astype(np.float32)
manual_dense_weights_float = manual_dense_weights1 / np.max(manual_dense_weights1)
manual_dense_bias_float = np.array([0.2], dtype=np.float32)


#### NON-QUANTIZED KERAS MODEL ####

model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(64, 64, 1), use_bias=True),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu', use_bias=True),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(1, activation='linear')
])

# Set weights and biases
conv_layer_1 = model.layers[0]
conv_layer_1.set_weights([manual_kernel_float_1, manual_bias_float_1])

conv_layer_2 = model.layers[2]
conv_layer_2.set_weights([manual_kernel_float_2, manual_bias_float_2])

conv_layer_3 = model.layers[4]
conv_layer_3.set_weights([manual_kernel_float_3, manual_bias_float_3])

conv_layer_4 = model.layers[6]
conv_layer_4.set_weights([manual_kernel_float_4, manual_bias_float_4])

dense_layer = model.layers[8]
dense_layer.set_weights([manual_dense_weights_float, manual_dense_bias_float])

# Keras output
start_keras = time.time()
output_model = model.predict(input_float)
stop_keras = time.time()
timing_keras = stop_keras - start_keras

print("\nOutput of the float32 model:")
print(output_model.flatten())
print(f"\nTime for Keras output: {timing_keras:.6f} seconds")


#### QUANTIZATION SETUP #####

# Scaling factor calculation

b_in = B_W - 1 - np.round(np.log2(np.max(np.abs(input_float)))).astype(int)
b_k1 = B_W - 1 - np.round(np.log2(np.max(np.abs(manual_kernel_float_1)))).astype(int)
b_k2 = B_W - 1 - np.round(np.log2(np.max(np.abs(manual_kernel_float_2)))).astype(int)
b_k3 = B_W - 1 - np.round(np.log2(np.max(np.abs(manual_kernel_float_3)))).astype(int)
b_k4 = B_W - 1 - np.round(np.log2(np.max(np.abs(manual_kernel_float_4)))).astype(int)
b_kd = B_W - 1 - np.round(np.log2(np.max(np.abs(manual_dense_weights_float)))).astype(int)

# Scaling factors
scale_input = 2 ** b_in
scale_kernel1 = 2 ** b_k1
scale_kernel2 = 2 ** b_k2
scale_kernel3 = 2 ** b_k3
scale_kernel4 = 2 ** b_k4
scale_dense = 2 ** b_kd


#### QUANTIZATION ####

input_quantized = np.clip((input_float * scale_input).round(), -127, 127).astype(np.int16)

kernel1_quantized = np.clip((manual_kernel_float_1 * scale_kernel1).round(), -127, 127).astype(np.int16)
bias1_quantized = np.clip((manual_bias_float_1 * scale_input * scale_kernel1).round(), -32768, 32767).astype(np.int16)

kernel2_quantized = np.clip((manual_kernel_float_2 * scale_kernel2).round(), -127, 127).astype(np.int16)
kernel3_quantized = np.clip((manual_kernel_float_3 * scale_kernel3).round(), -127, 127).astype(np.int16)
kernel4_quantized = np.clip((manual_kernel_float_4 * scale_kernel4).round(), -127, 127).astype(np.int16)

weights_dense_quantized = np.clip((manual_dense_weights_float * scale_dense).round(), -127, 127).astype(np.int16)


#### MANUAL NETWORK ####

start_manual = time.time()

# First convolution
out1 = manual_conv2D(input_quantized, kernel1_quantized, bias1_quantized, padding=None)
out1 = relu_manual(out1)
out2 = maxpool2d_manual(out1, (2, 2), 2)
max_val = np.max(np.abs(out2))
scale_factor1 = 127 / max_val if max_val > 127 else 1 
out3 = (out2 * scale_factor1).round().astype(np.int16)

# Second convolution
bias2_quantized = np.clip((manual_bias_float_2 * scale_input * scale_kernel2 * scale_kernel1 * scale_factor1).round(), -32768, 32767).astype(np.int16)
out4 = manual_conv2D(out3, kernel2_quantized, bias2_quantized, padding=None)
out4 = relu_manual(out4)
out5 = maxpool2d_manual(out4, (2, 2), 2)
max_val = np.max(np.abs(out5))
scale_factor2 = 127 / max_val if max_val > 127 else 1 
out6 = (out5 * scale_factor2).round().astype(np.int16)

# Third convolution
bias3_quantized = np.clip((manual_bias_float_3 * scale_input * scale_kernel3 * scale_kernel2 * scale_kernel1 * scale_factor1 * scale_factor2).round(), -32768, 32767).astype(np.int16)
out7 = manual_conv2D(out6, kernel3_quantized, bias3_quantized, padding=None)
out7 = relu_manual(out7)
out8 = maxpool2d_manual(out7, (2, 2), 2)
max_val = np.max(np.abs(out8))
scale_factor3 = 127 / max_val if max_val > 127 else 1 
out9 = (out8 * scale_factor3).round().astype(np.int16)

# Fourth convolution
bias4_quantized = np.clip((manual_bias_float_4 * scale_input * scale_kernel4 * scale_kernel3 * scale_kernel2 * scale_kernel1 * scale_factor1 * scale_factor2 * scale_factor3).round(), -32768, 32767).astype(np.int16)
out10 = manual_conv2D(out9, kernel4_quantized, bias4_quantized, padding=None)
out10 = relu_manual(out10)
max_val = np.max(np.abs(out10))
scale_factor4 = 127 / max_val if max_val > 127 else 1 
out11 = (out10 * scale_factor4).round().astype(np.int16)


#### DENSE LAYER ####

bias_dense_quantized = np.clip((manual_dense_bias_float * scale_input * scale_dense * scale_kernel4 * scale_kernel3 * scale_kernel2 * scale_kernel1 * scale_factor1 * scale_factor2 * scale_factor3 * scale_factor4).round(), -32768, 32767).astype(np.int16)

out12 = out11.flatten()

# Overflow handling workaround
# To check steps, decomment the prints statements
result = 0
overflow = 0
for i in range(len(out12)):
    #print("out12", i, " = ", out12[i])
    #print("weights_dense_quantized", i, " = ", weights_dense_quantized[i])    
    product = out12[i] * weights_dense_quantized[i]
    if (result + product < 0):  # Works only if all values are positive
        overflow += 1
        result = result + product + np.int16(32767)
        result = result + 1
        #print("OVERFLOW!!!")
    else:
        result += product
    #print("Result ", i, " = ", result)
    
print("\nTimes it overflows: ", overflow)   
print("\nBias: ", bias_dense_quantized)

stop_manual = time.time()
timing_manual = stop_manual - start_manual

# Output
out13 = result + bias_dense_quantized
print("\nQuantized output: \n", out13)
print(f"\nTime for Keras output: {timing_manual:.6f} seconds")


#### DEQUANTIZATION ####

temp1 = 32768.0
temp2 = temp1 / (scale_input * scale_kernel1 * scale_factor4)
temp3 = temp2 / (scale_kernel2 * scale_factor3)
temp4 = temp3 / (scale_kernel3 * scale_factor2)
temp5 = temp4 / (scale_kernel4 * scale_dense * scale_factor1)

o1 = out13.astype(np.float32) / (scale_input * scale_kernel1 * scale_factor4)
o2 = o1 / (scale_kernel2 * scale_factor3)
o3 = o2 / (scale_kernel3 * scale_factor2)
o4 = o3 / (scale_kernel4 * scale_dense * scale_factor1)
output_dequantized = o4 + overflow * temp5

print("\nDequantized output:")
print(output_dequantized)


#### DIFFERENCE WITH FLOAT MODEL ####

delta = (output_model - output_dequantized)
print(f"\nDifference: {float(delta)}")
delta_perc = delta/output_model * 100
print(f"\nPercentage difference: {float(delta_perc):.4f}%")
timing_difference = - timing_keras + timing_manual
timing_diff_perc = timing_difference / timing_keras * 100
print(f"\nTiming difference: {timing_difference:.6f} seconds")
print(f"\nTiming difference percentage: {timing_diff_perc:.2f}%")