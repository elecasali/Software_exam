# Documentation for Neural Networks Project

**Authors**: Elena Casali, Lucrezia Fendillo 

---

## Table of Contents

- [Introduction](#introduction)
- [Contributions](#contributions)
- [Repository structure](#repository-structure)
  - [`cnn_quantized.py`](#cnn_quantizedpy)
  - [`unet.py`](#unetpy)
  - [`test_firmware.py`](#test_firmwarepy)
  - [`utils/`](#utils)
    - [__init__.py](#__init__py)
    - [utils.py](#utilspy)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [References](#references)

---

## Introduction

Neural Networks (NNs) are computational models inspired by the structure of biological neural networks, designed to identify patterns, weight options, and reach decisions through interconnected layers of neurons [[1]](#references). Each neuron receives weighted inputs, applies a bias, and passes its output to subsequent neurons through an activation function, enabling complex hierarchical representations. NNs form the foundation of deep learning, finding extensive application in medical imaging and segmentation tasks but often require significant computational resources due to the reliance on floating-point operations and multi-layer structures [[2]](#references)[[3]](#references). In particular, CNN architectures use **convolution** as the operation applied in its neurons.

A specialized architecture for image segmentation is the **U-Net**, which adopts an encoder-decoder structure based on the CNN. The **encoder** progressively contracts the spatial dimensions of the input while extracting abstract features, followed by a **bottleneck** that represents the point of maximum compression of information. The **decoder** then reconstructs the image to the original input resolution by expanding the feature maps, while **skip connections** link corresponding layers in the encoder and decoder, preserving spatial details and enhancing segmentation precision. Upsampling within the decoder increases spatial resolution, and concatenation operations merge feature maps from different layers to retain fine-grained spatial information. 

The goal of this project is to **quantize** NN architectures following the quantization method presented in *Xiong et al.* [[2]](#references), and to evaluate its computational feasibility and numerical correctness. The quantization reduces input and weights to 8-bit signed integers while maintaining them on 16-bit to avoid overflow, followed by layer-wise rescaling to propagate values through the network. The primary objective is to verify that quantization can be successfully applied to a CNN and a U-Net from a computational perspective, comparing the outputs of the float model and the dequantized model to assess numerical alignment. Furthermore, we provide the execution time so that, once implemented in VHDL firmware, it can be evaluated in comparison.  

It is important to note that **the networks have not been trained**, and the **biases, weights, and input matrices are randomly generated numbers**. This approach is intentional, as the project focuses on verifying the implementation of quantization and its compatibility with hardware-level simulation, rather than evaluating segmentation accuracy. The purpose of the project is to verify the successful implementation of quantization so that it can later be expanded and applied to more complex networks. This would enable their implementation on FPGA with **significantly reduced execution times while maintaining a good level of precision**, providing a scalable and efficient foundation for deploying larger networks in hardware environments. 

These codes are complementary to the FPGA implementation in VHDL, not provided in this repository. 

## Contributions 

This project is a joint effort between E. Casali and L. Fendillo (shortened hereafter as E.C. and L.F.).  

Respective contributions are as follows: when not specifically highlighted, both authors contributed equally; the parts authored by one contributor are specified in the corresponding section of the README document. The author of the code is the same as the author of the README section.  

In short, `utils.py `, as the basis of the project, was written by both E.C. and L.F., while the other codes are single-authored. For further clarification, see the following sections. 

## Repository structure

```
├── unet.py            # Builds float32 U-Net, performs quantization to int16, and dequantization
├── cnn_quantized.py   # Builds float32 CNN, performs quantization to int16, and dequantization
├── test_firmware.py   # Tests a manual firmware implementation against a Keras model
├── utils/             # Folder with helper functions for manual layer-wise quantization
│   ├── __init__.py    # marks the folder as a Python package
│   └── utils.py       # manual implementations of U-Net layer operations
├── env.yml            # Conda environment definition with all dependencies
└── README.md          # Project overview, instructions, and design rationale
```
Before commenting on the structure of the individual scripts, let's first explain the quantization procedure followed in both the CNN and the U-Net.

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

### cnn_quantized.py

#### Process Summary
0. Hardcoding: number of bits the user wants the quantization to be based on. for signed integers between -127 and 127, 8 bits are necessary.

1. Random input data, weights and biases are initialized, with seeds to ensure repsoducibility.

2. The non-quantized, Keras-based model is generated with 4 convolutional layers with ReLu, 3 MaxPolling layers, and a final, fully-connected dense layer.

3. Scale factors are calculated for the input and all kernels as outlined in *Xiong et al.* [[2]](#references).

4. Inputs, weights, and biases are quantized to signed `int8` but stored as `int16` to avoid overflow issues.

5. Manual forward propagation is executed layer-by-layer and based on the manual functions from the `utils.py` script.

6. Overflow detection and correction mechanisms are included during dense computations to mirror hardware-like constraints. In hardware, it was opted to store the data as `int32` or `int64`. The current mechanisms relies on the overflow resetting the summed variable to -32767, hence keeps a tally of the number of times the overflow happens and adds 32767 to bring the count back to 0, then adds the overflow after dequantizaion.

7. Dequantization reconstructs the floating-point output.

8. Results are directly compared to the float32 Keras CNN output for output difference, percentage difference, and timing difference. These values are employed in the evaluation of the firmware acceleration.

#### Results After Executing cnn_quantized.py
* **Output of the float32 model**
  A single scalar value.
* **Time for Keras output**
  Execution time of the Keras model, in seconds.
* **Times it overflows**
  Number of times the overflow limit is crossed in the manual network, used to re-add the overflow after dequantization
* **Quantized manual output:**
  Manual network output, before dequantization. Can be used to verify correct working of overflow handling mechanism.
* **Time for manual output**
  Execution time of the manual model, in seconds.
* **Dequantized manual output:**
  A single scalar value, closely matching the float32 output with minor numerical differences.
* **Difference**
  Difference between Keras output and manual output.
* **Percentage difference**
  Difference between Keras output and manual output as percentage of the Keras output.
* **Timing difference**
  Difference of execution times between Keras and manual networks.
* **Timing difference percentage**
  Difference of execution times between Keras and manual networks as percentage of the Keras network timing.



#### Observations:
The CNN serves as validation of the correctness of quantization for convolutional pipelines. It also provides comparison terms for the hardware implementation, in order to quantify the acceleration as well as the numerical accuracy in future complex implementations.

This step is essential before applying the quantization method to more complex architectures like the U-Net.

Results show a low numerical error with correct scaling and overflow handling, demonstrating compatibility with FPGA integer-based computation.

--- 

### unet.py

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
The same pattern repeats for `conv2`, `pool2`, `conv3`, `pool3`:

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

the manual convolution, ReLU, max pooling, upsampling, and concatenation functions used are implemented in the `utils` module. After completing the forward pass with the manually quantized U-Net, the network is then dequantized to recover a float representation. This allows you to verify the agreement between the dequantized output and the original float32 output from the Keras model, ensuring the correctness of the manual quantization workflow before moving to FPGA testing.

#### Results
The following results confirm that manual quantization and dequantization preserve the integrity of the U-Net output within a low percentage error, validating the correctness of the pipeline for FPGA preparation.

**Execution times:**

  * Float32 Keras: \~1.17 seconds
  * Manual quantized pipeline: \~1.06 seconds

**Percentage difference (float vs dequantized):**

  * **Maximum:** 8.33 %
  * **Mean:** 0.99 %

---

### test_firmware.py --
This code accurately simulates the VHDL implementation. It is based on slightly modified functions with respect to those outlined in `utils.py` to faithfully reproduce the VHDL version and aid in the VHDL coding.

#### Process Summary
0. Hardcoding: tune for the rescaling function mimicking the rescaling in the hardware implementation, and number of statistical tests the user wants to perform.

1. Inputs, weights, and biases are all stored as 8  bit signed integers, but we use 16 bit integers for all intermediate calculations to avoid overflow.

2. The Keras-based CNN consists of a single 2D convolution layer with padding, ReLU activation, MaxPooling, a fully connected dense layer. Rescaling is applied between the MaxPooling and the dense layers, resulting in the CNN having to be split in two Keras sub-models.

3. A shift-based rescaling method is applied with a configurable parameter TUNE = 12 (scaling factor equivalent to multiplying by $2^{12} = 4096$). This simulates bit-shifting instead of division, in order to approximate the result of the division without using float values in hardware.

4. Manual forward propagation through manual, numpy-exempt functions.

5. Deviation analysis computing output difference for a single run and percentage deviation relative to Keras output.

6. Extended statistical testing performs the same computation on N = 200 randomly generated inputs, producing average deviation, minimum, maximum, and median deviation, a histogram of deviations for statistical analysis.

7. Division library generation generates a file division_library.txt containing precomputed division results in hexadecimal format, designed to be used in FPGA LUT-based division, eliminating the need for hardware division circuits.

#### Results After Executing test_firmware.py
* **Output Keras: ... Generated in: ...**
  A single scalar, output of the Keras model, with the execution time in seconds.
* **Output manual: ... Generated in: ...**
  A single scalar, output of the manual model, with the execution time in seconds.
* **Deviation statistics in extended test**
  Results from N statistical tests with random inputs. 
* **Average deviation, Median eviation, Min deviation, Max deviation**
  Deviations of the distribution of results from the extended statistical test.
* **Plot**
  Histogram of the deviations distribution, which, if everything works correctly in the simulation, should show a single bin at 0% deviation.
* **division_library.txt**
  Library storing the division between value and scaling factor, so that floating point operations are not necessary in the hardware implementation.

#### Observations:
The firmware test script demonstrates robust numerical stability under integer-only computation.

The library-based division strategy is validated and ready for hardware deployment.

Deviation from float32 is consistently 0%, proving that the quantization and rescaling approach is viable for FPGA implementation.

This script bridges the gap between Python modeling and FPGA hardware constraints, offering a ready-to-use validation step for hardware accelerators.


### utils/

#### __init__.py
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

If you install additional packages or modify the environment, update `env.yml` with:

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
