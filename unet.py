import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten
# from tensorflow.keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten
from keras.models import Model
import time
import numpy as np
from utils.utils import *


#### U-NET MODEL DEFINITION ####

def unet_model(input_shape=(64, 64, 1)):
    inputs = Input(shape=input_shape)

    # ENCODER
    # Convolution + MaxPooling
    conv1 = Conv2D(2, (3, 3), activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D((2, 2))(conv1)
    print("Shape after conv1:", conv1.shape)
    print("Shape after pool1:", pool1.shape)

    # Convolution + MaxPooling
    conv2 = Conv2D(2, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D((2, 2))(conv2)
    print("Shape after conv2:", conv2.shape)
    print("Shape after pool2:", pool2.shape)

    # Convolution + MaxPooling
    conv3 = Conv2D(2, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D((2, 2))(conv3)
    print("Shape after conv3:", conv3.shape)
    print("Shape after pool3:", pool3.shape)

    # Bottleneck
    conv4 = Conv2D(2, (3, 3), activation='relu', padding='same')(pool3)
    print("Shape after conv4 (bottleneck):", conv4.shape)

    # DECODER
    # UpSampling + skip connection con conv3
    up1 = UpSampling2D((2, 2))(conv4)
    merge1 = Concatenate()([up1, conv3])
    dec1 = Conv2D(2, (3, 3), activation='relu', padding='same')(merge1)

    # UpSampling + skip connection con conv2
    up2 = UpSampling2D((2, 2))(dec1)
    merge2 = Concatenate()([up2, conv2])
    dec2 = Conv2D(2, (3, 3), activation='relu', padding='same')(merge2)

    # UpSampling + skip connection con conv1
    up3 = UpSampling2D((2, 2))(dec2)
    merge3 = Concatenate()([up3, conv1])
    dec3 = Conv2D(2, (3, 3), activation='relu', padding='same')(merge3)

    # Output layer 
    outputs = Conv2D(1, (1, 1), activation='relu')(dec3)

    
    model = Model(inputs, outputs, name="U-Net")

    return model

# Model creation
unet = unet_model(input_shape=(64, 64, 1))

#### INPUT DATA (FLOAT) ####

# Set the seed for reproducibility and normalize the input
np.random.seed(1)
input_data = np.random.rand(1, 64, 64, 1).astype(np.float32)# Example input tensor (batch, height, width, channels)
input_float = (input_data)/(np.max(input_data))

# Manual initialization and normalization of kernels and biases for Conv2D layers
np.random.seed(1)
manual_kernel_11 = np.random.rand(3, 3, 1, 2).astype(np.float32) # Kernel shape: height=3, width=3, input_channels=1, output_channels=2
manual_kernel_float_1 = (manual_kernel_11)/(np.max(manual_kernel_11))
manual_bias_float_1 = np.zeros(2)

np.random.seed(2)
manual_kernel_22 = np.random.rand(3, 3, 2, 2).astype(np.float32)  # Kernel shape: height=3, width=3, input_channels=2, output_channels=2
manual_kernel_float_2 = (manual_kernel_22)/(np.max(manual_kernel_22))
manual_bias_float_2 = np.zeros(2)

np.random.seed(3)
manual_kernel_33 = np.random.rand(3, 3, 2, 2).astype(np.float32)  # Kernel shape: height=3, width=3, input_channels=2, output_channels=2
manual_kernel_float_3 = (manual_kernel_33)/(np.max(manual_kernel_33))
manual_bias_float_3 = np.zeros(2) 

np.random.seed(4)
manual_kernel_44 = np.random.rand(3, 3, 2, 2).astype(np.float32)  # Kernel shape: height=3, width=3, input_channels=2, output_channels=2
manual_kernel_float_4 = (manual_kernel_44)/(np.max(manual_kernel_44))
manual_bias_float_4 = np.zeros(2) 

np.random.seed(5)
manual_kernel_55 = np.random.rand(3, 3, 4, 2).astype(np.float32)  # Kernel shape: height=3, width=3, input_channels=4, output_channels=2
manual_kernel_float_5 = (manual_kernel_55)/(np.max(manual_kernel_55))
manual_bias_float_5 = np.zeros(2) 

np.random.seed(6)
manual_kernel_66 = np.random.rand(3, 3, 4, 2).astype(np.float32)  # Kernel shape: height=3, width=3, input_channels=4, output_channels=2
manual_kernel_float_6 = (manual_kernel_66)/(np.max(manual_kernel_66))
manual_bias_float_6 = np.zeros(2)

np.random.seed(7)
manual_kernel_77 = np.random.rand(3, 3, 4, 2).astype(np.float32)  # Kernel shape: height=3, width=3, input_channels=4, output_channels=2
manual_kernel_float_7 = (manual_kernel_77)/(np.max(manual_kernel_77))
manual_bias_float_7 = np.zeros(2)

np.random.seed(8)
manual_kernel_88 = np.random.rand(1, 1, 2, 1).astype(np.float32)  # Kernel shape: height=1, width=1, input_channels=2, output_channels=1
manual_kernel_float_8 = (manual_kernel_88)/(np.max(manual_kernel_88))
manual_bias_float_8 = np.zeros(1)  

# Setting weights and biases
unet = unet_model(input_shape=(64, 64, 1))
conv_layer_1 = unet.layers[1]  # Extract the first Conv2D layer
conv_layer_1.set_weights([manual_kernel_float_1, manual_bias_float_1])  # Set kernel and bias
conv_layer_2 = unet.layers[3]  # Extract the second Conv2D layer
conv_layer_2.set_weights([manual_kernel_float_2, manual_bias_float_2])  
conv_layer_3 = unet.layers[5]  # Extract the third Conv2D layer
conv_layer_3.set_weights([manual_kernel_float_3, manual_bias_float_3])  
conv_layer_4 = unet.layers[7]  # Extract the fourth Conv2D layer
conv_layer_4.set_weights([manual_kernel_float_4, manual_bias_float_4])  
conv_layer_5 = unet.layers[10]  # Extract the fifth Conv2D layer
conv_layer_5.set_weights([manual_kernel_float_5, manual_bias_float_5])  
conv_layer_6 = unet.layers[13]  # Extract the sixth Conv2D layer
conv_layer_6.set_weights([manual_kernel_float_6, manual_bias_float_6]) 
conv_layer_7 = unet.layers[16]  # Extract the seventh Conv2D layer
conv_layer_7.set_weights([manual_kernel_float_7, manual_bias_float_7])
conv_layer_8 = unet.layers[17]  # Extract the eighth Conv2D layer
conv_layer_8.set_weights([manual_kernel_float_8, manual_bias_float_8])


##### RUNNING KERAS MODEL #####
start1 = time.time()
output_model = unet.predict(input_float)
end1 = time.time()
timing1 = end1 - start1

print("\nOutput model (float32):")
print(output_model.flatten())
print(f"\nExecution time: {timing1:.4f} seconds")
    
test = np.random.rand(2, 2, 1).astype(np.float32)
test2 = np.random.rand(2, 2, 2).astype(np.float32)
out_test = manual_upsampling(test)
print(out_test.shape)

#### MANUAL QUANTIZED U-NET ####

# Compute scale factor for quantization ----> also  scale factor may not be 8-bit
max_input_value = np.max(np.abs(input_float))
max_kernel1_value = np.max(np.abs(manual_kernel_float_1))
max_kernel2_value = np.max(np.abs(manual_kernel_float_2))
max_kernel3_value = np.max(np.abs(manual_kernel_float_3))
max_kernel4_value = np.max(np.abs(manual_kernel_float_4))

max_kernel5_value = np.max(np.abs(manual_kernel_float_5))
max_kernel6_value = np.max(np.abs(manual_kernel_float_6))
max_kernel7_value = np.max(np.abs(manual_kernel_float_7))
max_kernel8_value = np.max(np.abs(manual_kernel_float_8))


b_w = 8  # Total number of bits
b_in = b_w - 1 - np.round(np.log2(max_input_value)).astype(int) 
b_k1 = b_w - 1 - np.round(np.log2(max_kernel1_value)).astype(int)
b_k2 = b_w - 1 - np.round(np.log2(max_kernel2_value)).astype(int)
b_k3 = b_w - 1 - np.round(np.log2(max_kernel3_value)).astype(int)
b_k4 = b_w - 1 - np.round(np.log2(max_kernel4_value)).astype(int)

b_k5 = b_w - 1 - np.round(np.log2(max_kernel5_value)).astype(int)
b_k6 = b_w - 1 - np.round(np.log2(max_kernel6_value)).astype(int)
b_k7 = b_w - 1 - np.round(np.log2(max_kernel7_value)).astype(int)
b_k8 = b_w - 1 - np.round(np.log2(max_kernel8_value)).astype(int)


# Unrestricted scale factors, no overflow now using 16-bit
scale_input = 2 ** b_in
scale_kernel1 = 2 ** b_k1
scale_kernel2 = 2 ** b_k2
scale_kernel3 = 2 ** b_k3
scale_kernel4 = 2 ** b_k4

scale_kernel5 = 2 ** b_k5
scale_kernel6 = 2 ** b_k6
scale_kernel7 = 2 ** b_k7
scale_kernel8 = 2 ** b_k8


#### QUANTIZATION ####
input_quantized = np.clip((input_float * scale_input).round(), -127, 127).astype(np.int16)

kernel1_quantized = np.clip((manual_kernel_float_1 * scale_kernel1).round(), -127, 127).astype(np.int16)
bias1_quantized = np.clip((manual_bias_float_1 * scale_input * scale_kernel1).round(), -32768, 32767).astype(np.int16)

kernel2_quantized = np.clip((manual_kernel_float_2 * scale_kernel2).round(), -127, 127).astype(np.int16)
kernel3_quantized = np.clip((manual_kernel_float_3 * scale_kernel3).round(), -127, 127).astype(np.int16)
kernel4_quantized = np.clip((manual_kernel_float_4 * scale_kernel4).round(), -127, 127).astype(np.int16)

kernel5_quantized = np.clip((manual_kernel_float_5 * scale_kernel5).round(), -127, 127).astype(np.int16)
kernel6_quantized = np.clip((manual_kernel_float_6 * scale_kernel6).round(), -127, 127).astype(np.int16)
kernel7_quantized = np.clip((manual_kernel_float_7 * scale_kernel7).round(), -127, 127).astype(np.int16)
kernel8_quantized = np.clip((manual_kernel_float_8 * scale_kernel8).round(), -127, 127).astype(np.int16)


start2 = time.time()

#### MANUAL UNET ####

# First convolution
out1 = manual_conv2D(input_quantized, kernel1_quantized, bias1_quantized, padding='same')
out1 = relu_manual(out1)
out2 = maxpool2d_manual(out1, (2,2), 2)
max_val = np.max(np.abs(out2))
scale_factor1 = 127 / max_val if max_val > 127 else 1 
out3 = (out2 * scale_factor1).round().astype(np.int16)

# Second convolution
bias2_quantized = np.clip((manual_bias_float_2 * scale_input * scale_kernel2 * scale_kernel1 * scale_factor1).round(), -32768, 32767).astype(np.int16)
out4 = manual_conv2D(out3, kernel2_quantized, bias2_quantized, padding='same')
out4 = relu_manual(out4)
out5 = maxpool2d_manual(out4, (2,2), 2)
max_val = np.max(np.abs(out5))
scale_factor2 = 127 / max_val if max_val > 127 else 1 
out6 = (out5 * scale_factor2).round().astype(np.int16)

# Third convolution
bias3_quantized = np.clip((manual_bias_float_3 * scale_input * scale_kernel3 * scale_kernel2 * scale_kernel1 *scale_factor1 * scale_factor2).round(), -32768, 32767).astype(np.int16)
out7 = manual_conv2D(out6, kernel3_quantized, bias3_quantized, padding='same')
out7 = relu_manual(out7)
out8 = maxpool2d_manual(out7, (2,2), 2)
max_val = np.max(np.abs(out8))
scale_factor3 = 127 / max_val if max_val > 127 else 1 
out9 = (out8 * scale_factor3).round().astype(np.int16)

# Fourth convolution: bottleneck
bias4_quantized = np.clip((manual_bias_float_4 * scale_input * scale_kernel4 * scale_kernel3 * scale_kernel2 * scale_kernel1 * scale_factor1 * scale_factor2 * scale_factor3).round(), -32768, 32767).astype(np.int16)
out10 = manual_conv2D(out9, kernel4_quantized, bias4_quantized, padding='same')
out10 = relu_manual(out10)
max_val = np.max(np.abs(out10))
scale_factor4 = 127 / max_val if max_val > 127 else 1 
out11 = (out10 * scale_factor4).round().astype(np.int16)

# UpSampling + skip connection with third convolution output
out12 = manual_upsampling(out11)
out13 = (out7 * scale_factor4 * scale_factor3 * scale_kernel4).round().astype(np.int16)
out14 = manual_concatenate(out12, out13, -1)
bias5_quantized = np.clip((manual_bias_float_5 * scale_input * scale_kernel5 * scale_kernel4 * scale_kernel3 * scale_kernel2 * scale_kernel1 * scale_factor1 * scale_factor2 * scale_factor3 * scale_factor4).round(), -32768, 32767).astype(np.int16)
out15 = manual_conv2D(out14, kernel5_quantized, bias5_quantized, padding='same')
out16 = relu_manual(out15)
max_val = np.max(np.abs(out16))
scale_factor5 = 127 / max_val if max_val > 127 else 1 
out17 = (out16 * scale_factor5).round().astype(np.int16)


# UpSampling + skip connection with second convolution output
out18 = manual_upsampling(out17)
out19 = (out4 * scale_factor5 * scale_factor4 * scale_factor3 * scale_factor2 * scale_kernel5 * scale_kernel4 * scale_kernel3).round().astype(np.int16)
out20 = manual_concatenate(out18, out19, -1)
bias6_quantized = np.clip((manual_bias_float_6 * scale_input * scale_kernel6 * scale_kernel5 * scale_kernel4 * scale_kernel3 * scale_kernel2 * scale_kernel1 * scale_factor1 * scale_factor2 * scale_factor3 * scale_factor4 * scale_factor5).round(), -32768, 32767).astype(np.int16)
out21 = manual_conv2D(out20, kernel6_quantized, bias6_quantized, padding='same')
out22 = relu_manual(out21)
max_val = np.max(np.abs(out22))
scale_factor6 = 127 / max_val if max_val > 127 else 1 
out23 = (out22 * scale_factor6).round().astype(np.int16)

# UpSampling + skip connection with first convolution output
out24 = manual_upsampling(out23)
out25 = (out1 * scale_factor6 * scale_factor5 * scale_factor4 * scale_factor3 * scale_factor2 * scale_factor1 * scale_kernel5 * scale_kernel4 * scale_kernel3 * scale_kernel2).round().astype(np.int16)
out26 = manual_concatenate(out24, out25, -1)
bias7_quantized = np.clip((manual_bias_float_7 * scale_input * scale_kernel7 * scale_kernel6 * scale_kernel5 * scale_kernel4 * scale_kernel3 * scale_kernel2 * scale_kernel1 * scale_factor1 * scale_factor2 * scale_factor3 * scale_factor4 * scale_factor5 * scale_factor6).round(), -32768, 32767).astype(np.int16)
out27 = manual_conv2D(out26, kernel7_quantized, bias7_quantized, padding='same')
out28 = relu_manual(out27)
max_val = np.max(np.abs(out28))
scale_factor7 = 127 / max_val if max_val > 127 else 1 
out29 = (out28 * scale_factor7).round().astype(np.int16)

# Output layer (binary segmentation with ReLu activation)
bias8_quantized = np.clip((manual_bias_float_8 * scale_input * scale_kernel8 * scale_kernel7 * scale_kernel6 * scale_kernel5 * scale_kernel4 * scale_kernel3 * scale_kernel2 * scale_kernel1 * scale_factor1 * scale_factor2 * scale_factor3 * scale_factor4 * scale_factor5 * scale_factor6 * scale_factor7).round(), -32768, 32767).astype(np.int16)
out30 = manual_conv2D(out29, kernel8_quantized, bias8_quantized, padding='valid')
print("\nBefore sigmoid/ReLU: \n", out30)
#out31 = manual_sigmoid(out30)
out30 = relu_manual(out30)
#print("\nAfter sigmoid: \n", out31)

# Quantized output
output_quantized_beta = out30  

o1 = output_quantized_beta.astype(np.float32) / (scale_input * scale_kernel1 * scale_factor4)
o2 = o1 / (scale_kernel2 * scale_factor3)
o3 = o2 / (scale_kernel3 * scale_factor2)
o4 = o3 / (scale_kernel4 * scale_factor1)
o5 = o4 / (scale_kernel5 * scale_factor5)
o6 = o5 / (scale_kernel6 * scale_factor6)
o7 = o6 / (scale_kernel7 * scale_factor7 * scale_kernel8)

#o8 = manual_sigmoid(o7)
output_dequantized = o7

end2 = time.time()
timing2 = end2 - start2

print("\nDequantized output:")
print(output_dequantized)
print(f"\nExecution time: {timing2:.4f} seconds")

print("\nOutput from non-quantized network:")
print(output_model)

delta = np.abs((output_model - output_dequantized)/(output_model)) * 100
print("\nPercentage difference: \n", delta, "%")
max_diff = np.max(delta)
print("\nMaximum percentage difference: ", max_diff, "%")
mean_diff = np.mean(delta)
print("\nMean percentage difference: ", mean_diff, "%")