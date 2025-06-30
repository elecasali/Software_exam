# Documentation for Neural Networks Project

**Authors**: Elena Casali, Lucrezia Fendillo 

---

## Table of Contents

- [Introduction](#introduction)
- [Contributions](#contributions)
- [Repository structure](#repository-structure)
  - [`cnn_quantized.py` -- L.F.](#cnn_quantizedpy)
  - [`unet.py` -- E.C.](#unetpy)
  - [`test_firmware.py` -- L.F.](#test_firmwarepy)
  - [`utils/` -- E.C., L.F.](#utils)
    - [`__init__.py`](#__init__py)
    - [`utils.py`](#utilspy)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [References](#references)

---

## Introduction

Neural Networks (NNs) are computational models inspired by the structure of biological neural networks, designed to identify patterns, weight options, and reach decisions through layers and nodes of neurons [[1]](#references). Each neuron receives weighted inputs, applies a bias, and passes its output to following neurons through an activation function. NNs are the basis of deep learning, with applications in medical imaging and segmentation tasks, but often require significant computational resources due to their need of floating-point operations and multi-layer structures [[2]](#references)[[3]](#references). In particular, CNN architectures use **convolution** as the operation applied in their neurons.

The **U-Net** is an architecture based on the CNN and often employed for image segmentation. It has an ecoder-decoder structure, where the **encoder** starts by contracting the spatial dimensions of the input and extracting abstract features; the point of maximum information compression is the **bottleneck**, which is often a fully-connected dense layer. Until this point, the functions are the same as a regular CNN. The up-going branch is the **decoder** which reconstructs the image back to its original resolution. It expands the feature maps and uses **skip connections** to link corresponding layers in the encoder in order to transfer and preserve spatial details. Upsampling increases the spatial resolution and concatenation merge feature maps from different layers.

The goal of this project is to **quantize** NN architectures following the quantization method presented in *Xiong et al.* [[2]](#references), and to evaluate the computational feasibility and numerical accuracy. The quantization reduces input and weights to signed 8 bit integers while memorizing them on 16 bit to avoid overflow, followed by rescaling to transfer values through the network. The primary goal is to verify that quantization can be successfully applied to a CNN and a U-Net from a computational perspective, comparing the outputs of the float model and the dequantized model to verify numerical correspondence. Furthermore, we require the code to provide the execution time so that, once implemented in VHDL firmware, it can be evaluated in comparison.  

It is important to note that **the networks have not been trained**, and the **biases, weights, and input matrices are randomly generated numbers**. This approach is intentional, as the project focuses on verifying the implementation of quantization and its compatibility with hardware-level simulation, rather than evaluating segmentation accuracy. The purpose of the project is to verify the successful implementation of quantization so that it can later be expanded and applied to more complex networks. This would enable their implementation on FPGA with **significantly reduced execution times while maintaining a good level of precision**, providing a scalable and efficient foundation for deploying larger networks in hardware environments. 

These codes are complementary to the FPGA implementation in VHDL, not provided in this repository. 


## Contributions 

This project is a joint effort between E. Casali and L. Fendillo (shortened hereafter as E.C. and L.F.).  

Respective contributions are as follows: when not specifically highlighted, both authors contributed equally; the parts authored by one contributor are specified in the corresponding section of the README document. The author of the code is the same as the author of the README section.  

In short, `utils.py `, as the basis of the project, was written by both E.C. and L.F., while the other codes are single-authored. For further clarification, see the following sections. 

## Repository structure

```
├── cnn_quantized.py   # Builds float32 CNN, performs quantization to int16, and dequantization
├── unet.py            # Builds float32 U-Net, performs quantization to int16, and dequantization
├── test_firmware.py   # Tests a manual firmware implementation against a Keras model
├── utils/             # Folder with helper functions for manual layer-wise quantization
│   ├── __init__.py    # marks the folder as a Python package
│   └── utils.py       # manual implementations of U-Net layer operations
├── env.yml            # Conda environment definition with all dependencies
└── README.md          # Project overview, instructions, and design rationale
```
Before commenting on the structure of the individual scripts, we first explain the quantization procedure followed in both the CNN and the U-Net.

### Purpose of Quantization:

Quantization is used to turn floating-point weights, inputs, and activations into lower bit-width integers to save memory, reduce bandwidth, and enable faster computations on FPGA hardware that uses integer arithmetic. The goal is to achieve these benefits with as little numerical accuracy loss as possible.

Here, we apply 8-bit signed quantization while storing values in int16 to prevent overflow when multiplying and accumulating values during layer operations.

#### Quantization Formula

Given a floating-point value: $x_{float}$, we first compute a scale factor:

$$
S = 2^{b} \quad \text{where} \quad b = b_w - 1 - \lfloor \log_2 (\max(|x_{float}|)) \rfloor
$$

The quantized value is computed as:

$$
\boxed{ x_{quant} = \mathrm{clip}(\mathrm{round}(x_{float} \times S), -127, 127) } .
$$

as described in *Xiong et al.* [[2]](#references).
#### Dequantization Formula

To get back to approximate floating-point values, we use:

$$
\boxed{ x_{dequant} = \frac{x_{quant}}{S} }
$$

Since layers are connected in sequence, we divide by the product of all scale factors and kernel scales used throughout the forward pass.

Using int16 is essential here because when multiple quantized values are multiplied together, they can exceed the 8-bit range and risk overflow. Storing in int16 ensures computations remain accurate before scaling down for the next layer.

#### Process Summary

1. Compute scale factors based on the maximum absolute values in weights and inputs to fully use the 8-bit range while preventing overflow.
2. Quantize inputs, weights, and biases using these scale factors and store them as int16.
3. Perform the forward pass manually using functions stored in `utils`, staying within the quantized domain.
4. Dequantize progressively by dividing by the combined scale factors.
5. Compare the dequantized output with the float32 model to ensure they align closely.

---

### cnn_quantized.py -- L.F.
#### Library Imports

For building and testing the CNN, the following libraries are used:

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D
import time
from utils.utils import * 
```

* **TensorFlow Keras** is used to define and run the CNN float32 model for reference.
* **NumPy** is used for generating random input data and manually creating and processing kernels and biases.
* **time** is used to measure execution times for both Keras and manual quantized models.

---

#### CNN Model Definition

The CNN is a simple feed-forward convolutional network composed of:

* 4 convolutional layers, each followed by ReLU activation.
* 3 MaxPooling layers that progressively reduce spatial dimensions.
* A final Flatten layer followed by a fully connected Dense output layer.

##### Model structure:

```python
model = Sequential([
    Conv2D(8, (3, 3), activation='relu', input_shape=(64, 64, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(16, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(1, activation='linear')
])
```

---

#### Keras CNN Model Execution

1. **Input generation:**

```python
np.random.seed(1)
input_data = np.random.rand(1, 64, 64, 1).astype(np.float32)
input_float = input_data / np.max(input_data)
```

2. **Kernel initialization:**

```python
np.random.seed(1)
manual_kernel = np.random.rand(3, 3, 1, 8).astype(np.float32)
manual_kernel_float = manual_kernel / np.max(manual_kernel)
manual_bias_float = np.zeros(8)
```

3. **Weight assignment:**

```python
conv_layer_1.set_weights([manual_kernel_float_1, manual_bias_float_1])
```

4. **Model execution:**

```python
start_keras = time.time()
output_model = model.predict(input_float)
stop_keras = time.time()
timing_keras = stop_keras - start_keras
```

---

#### Quantization

##### Scale Factor Calculation

```python
b_in = B_W - 1 - np.round(np.log2(np.max(np.abs(input_float)))).astype(int)
scale_input = 2 ** b_in
```

##### Quantization Formula

```math
x_{quant} = \mathrm{clip}(\mathrm{round}(x_{float} \cdot S), -127, 127)
```

##### Dequantization Formula

```math
x_{dequant} = \frac{x_{quant}}{S}
```

##### Overflow Handling

* Manual overflow handling resets the sum to -32767.
* An overflow counter is incremented each time.
* Final result corrected with:

```python
result += overflow_counter * 32767
```

---

#### Process Summary

0. Hardcode bit width (8 bits signed, int16 storage).
1. Generate random inputs and weights.
2. Build float Keras model.
3. Calculate scale factors.
4. Quantize inputs, weights, biases.
5. Manual forward pass using firmware-like functions.
6. Handle overflow events.
7. Dequantize output.
8. Compare results to float32 model.

---

#### Results

* Validates quantization before moving to U-Net.
* Overflow handling behaves correctly.
* Numerical error is low and acceptable.
* Results demonstrate viability for FPGA/ASIC deployment.

---

#### Summary Conclusion

This CNN quantization pipeline confirms the accuracy and robustness of the integer-based approach, including overflow handling and rescaling. It provides a reliable foundation for scaling to larger networks or hardware implementations like FPGAs.


--- 

### unet.py -- E.C.

#### Library imports

For building the network, the following libraries were used:

```python
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Flatten
from keras.models import Model
import time
import numpy as np
from utils.utils import *  # contains funtions manually defined
```
where **TensorFlow and Keras** are required for constructing and executing the U-Net, **time** is used to measure execution time, **numpy** is used for generating and manipulating input data and manually initializing kernels and biases, and **utils.utils** contains manually implemented layer operations for manual quantization.

#### U-NET model definition

The command:
```python
def unet_model(input_shape=(64, 64, 1)):
```
defines a function that builds and returns a U-Net model, which takes as input `input_shape=(64, 64, 1)`, a tensor with dimension 64x64 and 1 channel (grayscale image).

In the *Encoder* section, an input layer with the specified shape is created for the model, followed by the application of Convolution and MaxPooling operations three times in sequence.

```python
inputs = Input(shape=input_shape)

conv1 = Conv2D(2, (3, 3), activation='relu', padding='same')(inputs)
pool1 = MaxPooling2D((2, 2))(conv1)
```
where `Conv2D(2, (3, 3), activation='relu', padding='same')` applies 2 convolution filters of size 3x3, uses ReLU as activation function and with `padding='same'` keeps the output size the same as input. The `MaxPooling2D((2, 2))` instead, applies 2x2 max pooling, reducing spatial dimensions by 2.
The same pattern repeats for `conv2`, `pool2`, `conv3`, `pool3`.

The bottleneck layer:

```python
conv4 = Conv2D(2, (3, 3), activation='relu', padding='same')(pool3)
```
is a convolution after the final downsampling, used to extract high-level features at the lowest resolution.
The following prints confirm that the encoder is progressively reducing spatial dimensions while maintaining the expected channel structure.
```
Shape after conv1: (None, 64, 64, 2)
Shape after pool1: (None, 32, 32, 2)
Shape after conv2: (None, 32, 32, 2)
Shape after pool2: (None, 16, 16, 2)
Shape after conv3: (None, 16, 16, 2)
Shape after pool3: (None, 8, 8, 2)
Shape after conv4 (bottleneck): (None, 8, 8, 2)
```

In the *Decoder* section the **UpSampling2D** is used to double spatial dimensions (height and width) of the previous layer output using nearest-neighbor upsampling. Hence **Concatenate** is applied to concatenate the upsampled feature map with encoder outputs (skip connections), resulting in a intermediate stage with the number of channels doubled. 
Then the **Conv2D** layers applies a 2-filter convolution with a 3x3 kernel on the concatenated feature map halving the number of channels.

```python
    up1 = UpSampling2D((2, 2))(conv4)
    merge1 = Concatenate()([up1, conv3])
    dec1 = Conv2D(2, (3, 3), activation='relu', padding='same')(merge1)
```

The scheme followed in this branch is:

`up1` upsample + concatenate with `conv3` → `dec1` conv.

`up2` upsample + concatenate with `conv2` → `dec2` conv.

`up3` upsample + concatenate with `conv1` → `dec3` conv.

These layers progressively reconstruct the image size while preserving spatial details from encoder layers. 

The final output layer

```python
outputs = Conv2D(1, (1, 1), activation='relu')(dec3)
```
applies a final 1x1 convolution to reduce the channels to 1 for output.

Finally, we build and return the U-Net model, connecting the `inputs` and `outputs`.

```python
    model = Model(inputs, outputs, name="U-Net")
    return model

unet = unet_model(input_shape=(64, 64, 1))
```

#### Keras U-Net Model Execution

A fixed random seed is set for reproducibility, and a random 64x64 grayscale image (batch=1, height=64, width=64, channels=1) is generated. The input is then normalized to the [0, 1] range.

```python
np.random.seed(1)
input_data = np.random.rand(1, 64, 64, 1).astype(np.float32)  
input_float = input_data / np.max(input_data) 
```

The following block manually initializes and normalizes kernels and biases for each Conv2D layer in the U-Net using different fixed seeds. Kernels are generated with random values, normalized to [0, 1] to avoid overflow, while biases are initialized to zero.

```python
# Manual initialization and normalization of kernels and biases for Conv2D layers
np.random.seed(1)
manual_kernel_11 = np.random.rand(3, 3, 1, 2).astype(np.float32)
manual_kernel_float_1 = manual_kernel_11 / np.max(manual_kernel_11)
manual_bias_float_1 = np.zeros(2)
...
np.random.seed(8)
manual_kernel_88 = np.random.rand(1, 1, 2, 1).astype(np.float32)
manual_kernel_float_8 = manual_kernel_88 / np.max(manual_kernel_88)
manual_bias_float_8 = np.zeros(1)
```
Across the U-Net layers, the kernel dimensions evolve consistently while following architectural needs.

The weights are then assigned by extracting each Conv2D layer by its index and assigning the manually created and normalized kernels and biases to that layer using `set_weights()`.

```python
# Setting weights
unet = unet_model(input_shape=(64, 64, 1))
conv_layer_1 = unet.layers[1]
conv_layer_1.set_weights([manual_kernel_float_1, manual_bias_float_1])
 ...
conv_layer_8 = unet.layers[17]
conv_layer_8.set_weights([manual_kernel_float_8, manual_bias_float_8])
```

Now, the Keras model is executed while measuring the execution time using `time.time()`, and the output is printed for inspection.

```python
##### RUNNING KERAS MODEL #####
start1 = time.time()
output_model = unet.predict(input_float)
end1 = time.time()
timing1 = end1 - start1

print("\nOutput model (float32):")
print(output_model.flatten())
print(f"\nExecution time: {timing1:.4f} seconds")
```

#### Quantization

In the following block, the scale factors are computed as previously explained:

```python
#### MANUAL QUANTIZED U-NET ####

# Compute scale factors for quantization
max_input_value = np.max(np.abs(input_float))
...
max_kernel8_value = np.max(np.abs(manual_kernel_float_8))

b_w = 8  # Total number of bits
b_in = b_w - 1 - np.round(np.log2(max_input_value)).astype(int)
...
b_k8 = b_w - 1 - np.round(np.log2(max_kernel8_value)).astype(int)

# Compute scale factors to use for quantization
scale_input = 2 ** b_in
...
scale_kernel8 = 2 ** b_k8
```

Whhile in the `#### QUANTIZATION ####` step, the input, kernels, and biases are quantized.

```python
#### QUANTIZATION ####
input_quantized = np.clip((input_float * scale_input).round(), -127, 127).astype(np.int16)

kernel1_quantized = np.clip((manual_kernel_float_1 * scale_kernel1).round(), -127, 127).astype(np.int16)
bias1_quantized = np.clip((manual_bias_float_1 * scale_input * scale_kernel1).round(), -32768, 32767).astype(np.int16)

...
kernel8_quantized = np.clip((manual_kernel_float_8 * scale_kernel8).round(), -127, 127).astype(np.int16)
```
In the final block:

```python
#### MANUAL UNET ####
```

The manual convolution, ReLU, MaxPooling, Upsampling, and concatenation functions used are implemented in the `utils` module. After completing the layer-by-layer implementation of the manually quantized U-Net, the network is dequantized to restore a float representation. This allows us to verify the agreement between the dequantized output and the original float32 output from the Keras model, ensuring the accuracy of the manual quantization before moving to FPGA testing.

#### Results
The results confirm that manual quantization and dequantization preserve the integrity of the Keras U-Net output within a low percentage error, validating the correctness of the pipeline for FPGA preparation.

---

### test_firmware.py -- L.F.

#### Description

This script simulates the behavior of a CNN as it would be implemented in VHDL firmware. It uses slightly modified functions compared to those in `utils.py` to more closely replicate the VHDL logic and assist in hardware coding and verification.

---

#### Process Summary

0. **Fixed Parameters:**

   * `TUNE = 12`: A constant used in the rescaling process to mimic hardware shifts instead of divisions. Equivalent to multiplying by \$2^{12} = 4096\$.
   * `N = 200`: Number of random input samples for the extended statistical test.

1. **Integer Data Representation:**

   * Inputs, weights, and biases are stored as **8-bit signed integers (`int8`)**.
   * Intermediate computations use **16-bit signed integers (`int16`)** to prevent overflow.

2. **Keras Model:**

   * A CNN with:

     * 1 convolution layer (with padding), followed by ReLU activation.
     * MaxPooling layer.
     * Dense (fully connected) layer.
   * A rescaling step is inserted between MaxPooling and Dense to match hardware behavior.
   * The CNN is split into two Keras models to incorporate this rescaling.

3. **Rescaling Mechanism:**

   * Instead of division, the code applies **bit-shifting** based rescaling.
   * Formula: multiply by a scaling factor (e.g., `127 * 4096`), then apply a right shift (`>> TUNE`) to normalize the value.

4. **Manual Forward Propagation:**

   * All operations are performed with **manual integer-based functions**, avoiding NumPy built-ins that use floating-point.

5. **Single Run Deviation Analysis:**

   * The manual integer pipeline output is compared to the Keras float output.
   * Computes absolute and percentage deviation.

6. **Extended Statistical Testing:**

   * Runs N = 200 random input tests.
   * Collects:

     * Average deviation.
     * Median deviation.
     * Minimum deviation.
     * Maximum deviation.
   * Generates a histogram to visualize the deviation distribution.

7. **Division Lookup Table Generation:**

   * Generates `division_library.txt`.
   * This file stores precomputed integer division results in hexadecimal format.
   * Used to replace division operations with LUT access in FPGA hardware.

---

#### Results After Executing `test_firmware.py`

* **Keras Output:**

  * A scalar value.
  * Execution time displayed in seconds.

* **Manual Output:**

  * A scalar integer value.
  * Execution time displayed in seconds.

* **Deviation Statistics:**

  * **Average deviation**
  * **Median deviation**
  * **Minimum deviation**
  * **Maximum deviation**

* **Plot:**

  * A histogram visualizing the deviations from the extended statistical test.
  * If the firmware model is accurate, the histogram should show a single bin at 0% deviation.

* **Division Lookup Table:**

  * `division_library.txt` contains precomputed scaled division results in hexadecimal.
  * Used to remove floating-point operations from hardware implementations.

---

#### Observations

* The firmware test script shows **excellent numerical stability** using only integer arithmetic.
* The lookup-table-based division approach works flawlessly and is optimized for FPGA deployment.
* The extended statistical test typically shows **0% deviation**, demonstrating the quantization and rescaling strategies are accurate and reliable.
* This script acts as a **crucial validation step**, ensuring that the firmware logic in Python matches the expected FPGA hardware behavior.

---

#### Summary Conclusion

This firmware simulation pipeline provides a reliable bridge between Python-based algorithm development and hardware implementation constraints. The integer arithmetic, overflow handling, and rescaling logic have been fully validated, offering a practical framework for translating neural network models into FPGA-optimized designs.


### utils/ -- E.C., L.F.

#### `__init__.py`
The `__init__.py` file, even if empty, serves as a marker to Python that the directory should be treated as a package, not just a plain folder.

---

#### utils.py

This module provides manual NumPy-based implementations of core layer operations needed to simulate U-Net and CNN forward passes under quantization constraints, without relying on Keras internals.

##### `maxpool2d_manual`

It takes `input_data` with shape (height, width, channels) and performs 2D max pooling using a 2x2 window that moves with a stride of 2, taking the maximum value in each window to reduce spatial dimensions. Then returns a downsampled tensor.

```python
def maxpool2d_manual(input_data, pool_size=(2, 2), stride=2):
    input_height, input_width, num_channels = input_data.shape
    pool_height, pool_width = pool_size
    output_height = (input_height - pool_height) // stride + 1
    output_width = (input_width - pool_width) // stride + 1
    output = np.zeros((output_height, output_width, num_channels))
    for i in range(output_height):
        for j in range(output_width):
            for c in range(num_channels):
                region = input_data[i*stride:i*stride+pool_height, j*stride:j*stride+pool_width, c]
                output[i, j, c] = np.max(region)
    return output
```

##### `relu_manual`

The ReLU activation function replaces all negative values with zero and leaves positive values unchanged while maintaining the same input dimensions.

```python
def relu_manual(input_data):
    return np.maximum(input_data, 0)
```

##### `add_padding`

This function adds zero-padding to the input tensor in order to have an output with the same input dimensions after a convolution with 'same' padding.
Te padding size (`pad_h`, `pad_w`) is computed based on kernel size.

```python
def add_padding(input_data, kernel_height, kernel_width):
    pad_h = (kernel_height - 1) // 2
    pad_w = (kernel_width - 1) // 2
    padded_input = np.pad(input_data, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant', constant_values=0)
    return padded_input
```

##### `manual_conv2D`

This function manually performs a 2D convolution on an input tensor using a specified kernel and bias, with optional 'same' padding. It checks if the input has a batch dimension (4D) and reshapes it to 3D for computation. 
It extracts the dimensions of both the input and the kernel, ensuring that the input channels match the kernel channels. 
If 'same' padding is requested, it uses `add_padding` to maintain output size consistency. It calculates the output dimensions based on the input and kernel sizes with a stride of 1, initializes an output tensor, and uses nested loops over height, width, and filters. 
For each position, it computes the sum of the element-wise product of the kernel and the corresponding region in the input across all channels, adds the filter bias, and stores the result.

```python
def manual_conv2D(input_data, kernel, bias, padding):
    stride = 1
    if len(input_data.shape) == 4:
        input_data = input_data.reshape(input_data.shape[1:])
    input_height, input_width, input_channels = input_data.shape
    kernel_height, kernel_width, kernel_channels, num_filters = kernel.shape
    if input_channels != kernel_channels:
        raise ValueError(f"The input channels ({input_channels}) and the kernel channels ({kernel_channels}) must match")
    if padding == 'same':
        input_data = add_padding(input_data, kernel_height, kernel_width)
        input_height, input_width, _ = input_data.shape
    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width, num_filters))
    for i in range(output_height):
        for j in range(output_width):
            for f in range(num_filters):
                sum_ = 0
                for c in range(input_channels):
                    for di in range(kernel_height):
                        for dj in range(kernel_width):
                            sum_ += input_data[i + di, j + dj, c] * kernel[di, dj, c, f]
                sum_ += bias[f]
                output[i, j, f] = sum_
    return output
```

The convolution operation computed is:

$$
output[i, j, f] = 
\sum_{c=0}^{C-1} \sum_{di=0}^{K_h-1} \sum_{dj=0}^{K_w-1} input[i+di, j+dj, c] \times kernel[di, dj, c, f] + bias[f]
$$

where \$C\$ is the number of channels, \$K\_h\$ and \$K\_w\$ are the kernel height and width, and \$f\$ indexes the output filter.

##### `manual_upsampling`

It doubles the height and width of the input tensor using nearest-neighbor upsampling by copying each pixel into a 2x2 block while keeping channels unchanged.

```python
def manual_upsampling(input_data):
    if len(input_data.shape) == 4:
        input_data = input_data.reshape(input_data.shape[1:])
    elif len(input_data.shape) != 3:
        raise ValueError("The input input_data must have 3 or 4 dimensions")
    input_height, input_width, input_channels = input_data.shape
    output_height = input_height * 2
    output_width = input_width * 2
    output_data = np.zeros((output_height, output_width, input_channels))
    for i in range(input_height):
        for j in range(input_width):
            for c in range(input_channels):
                output_data[i*2:i*2+2, j*2:j*2+2, c] = input_data[i, j, c]
    return output_data
```

##### `manual_concatenate`

This mimics `Concatenate` in Keras, hence concatenates two tensors along the channel axis while ensuring they have the same height and width.
After checking shape compatibility, it creates a new tensor with combined channels, filling it with the values from the two tensors.

```python
def manual_concatenate(tensor1, tensor2, axis=-1):
    if tensor1.shape[0] != tensor2.shape[0] or tensor1.shape[1] != tensor2.shape[1]:
        raise ValueError("Error 1 concatenation: tensors must have the same height and width")
    if axis == -1 or axis == 3:
        merged = np.zeros((tensor1.shape[0], tensor1.shape[1], tensor1.shape[2] + tensor2.shape[2]))
        merged[:, :, :tensor1.shape[2]] = tensor1
        merged[:, :, tensor1.shape[2]:] = tensor2
        return merged
    else:
        raise ValueError("Error 2 concatenation: invalid axis")
```

##### `manual_sigmoid`

It applies the sigmoid activation element-wise to squash values between 0 and 1.

```python
def manual_sigmoid(tensor_input):
    return 1 / (1 + np.exp(-tensor_input))
```

---

## Installation

To install and run the project locally, download the repository using your browser (Download ZIP) or with Git in your preferred terminal (Windows CMD, Git Bash, or Anaconda Prompt):

```bash
git clone https://github.com/elecasali/Software_exam.git
cd Software_exam
```
Open the Anaconda Prompt (or Miniconda Prompt), then create and activate the environment using the provided `env.yml`:

```bash
conda env create -f env.yml -n Casali_Fendillo_exam
conda activate Casali_Fendillo_exam
```

This will install all required dependencies for the project. After that, your prompt will change to:

```
(Casali_Fendillo_exam) C:\Users\YourName>
```

indicating that your environment is active and ready for execution.

## Usage

Once the environment is activated, you can run the Python scripts directly:

```bash
python unet.py
```

or

```bash
python cnn_quantized.py
```

or :

```bash
python test_firmware.py
```

> **Note:** Ensure you are inside the cloned project folder in your terminal when running the scripts.

## Dependencies

The dependencies required for this project are included in the `env.yml` file, which contains:

* Python 3.12
* TensorFlow 2.18.1
* NumPy
* Keras
* and additional necessary libraries.

These will be installed automatically during the `conda env create` step.

If new packages need to be installed or changes need to be made to the environment, the `env.yml` file should be updated using the following command:

```bash
conda env export > env.yml
```

To deactivate the environment when finished:

```bash
conda deactivate
```

## References

[1] I. Goodfellow, Yoshua Bengio, and A. Courville, *Deep Learning*. Frechen: Mitp, 2018.

[2] S. Xiong et al., “MRI-based brain tumor segmentation using FPGA-accelerated neural network,” *BMC Bioinformatics*, vol. 22, no. 1, Sep. 2021. [https://doi.org/10.1186/s12859-021-04347-6](https://doi.org/10.1186/s12859-021-04347-6)

[3] Cristiana Fiscone et al., “Generalizing the Enhanced-Deep-Super-Resolution neural network to brain MR images: a retrospective study on the Cam-CAN dataset,” *eNeuro*, vol. 11, no. 5, pp. ENEURO.0458-22.2023, May 2024. [https://doi.org/10.1523/eneuro.0458-22.2023](https://doi.org/10.1523/eneuro.0458-22.2023)

[4] K. Team, “Keras documentation: Developer guides,” [https://keras.io/guides/](https://keras.io/guides/)
