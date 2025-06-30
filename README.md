# Documentation for Neural Networks Project

**Authors**: Elena Casali, Lucrezia Fendillo 

---

## Table of Contents

- [Introduction](#introduction)
- [Contributions](#contributions)
- [Repository structure](#repository-structure)
  - [`unet.py`](#unetpy)
  - [`cnn_quantized.py`](#cnn_quantizedpy)
  - [`test_firmware.py`](#test_firmwarepy)
  - [`utils/`](#utils)
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


### unet.py

#### Purpose of Quantization:

Quantization reduces **floating-point weights, inputs, and activations to lower bit-width integers** to reduce memory and bandwidth, to enable compatibility with FPGA integer arithmetic and to achieve faster computation with minimal numerical loss.

Here, **8-bit signed quantization** is applied, but **int16 storage is used** to **avoid overflow during intermediate multiplications and accumulations** within layers.

---

#### Quantization Formula

Given:

* Floating-point value: $x_{float}$
* Scale factor: $S = 2^{b}$ where $b = b_w - 1 - \lfloor \log_2 (\max(|x_{float}|)) \rfloor$

The quantized value is computed as:

$$
\boxed{ x_{quant} = \mathrm{clip}(\mathrm{round}(x_{float} \times S), -127, 127) }
$$

Here, the clip range is $[-127, 127]$ (8-bit signed), while **storage and computation are performed in int16** to avoid overflow in intermediate layers.

---

#### Dequantization Formula

To recover approximate floating-point values, the dequantized output is computed as:

$$
\boxed{ x_{dequant} = \frac{x_{quant}}{S} }
$$

Since multiple layers are cascaded, dequantization is applied progressively by dividing by the product of all scale factors and kernel scales used during the forward pass.

---

* The quantization uses 8-bit scaling, but during convolutions and accumulations in deeper layers, the product of multiple quantized values can **exceed the 8-bit range**, risking overflow.
* Using `int16` ensures safe computation without loss of information before re-scaling and clipping for the next layers.
* This is essential for **manual quantization and FPGA implementation** where overflow can lead to incorrect computations.

---

### Process Summary

1. **Scale factors are computed** based on the maximum absolute value of weights and inputs to utilize the 8-bit range fully while avoiding overflow.
2. **Inputs, weights, and biases are quantized using these scale factors** to int16.
3. Forward propagation is manually executed using `manual_conv2D`, `manual_maxpool2D`, `manual_upsampling`, and `manual_concatenate`, maintaining the quantized domain.
4. **Dequantization is performed progressively** by dividing by the product of all scale factors and kernel scales.
5. Results are compared with the float32 model to assess numerical alignment.

---

### Results After Executing `unet.py`

**Shape logs during U-Net construction:**

```
Shape after conv1: (None, 64, 64, 2)
Shape after pool1: (None, 32, 32, 2)
Shape after conv2: (None, 32, 32, 2)
Shape after pool2: (None, 16, 16, 2)
Shape after conv3: (None, 16, 16, 2)
Shape after pool3: (None, 8, 8, 2)
Shape after conv4 (bottleneck): (None, 8, 8, 2)
```

**Execution results:**

* **Output model (float32):**
  Values in the range of \~130k to 740k.
* **Dequantized output:**
  Closely matches the float32 output with minor numerical differences.
* **Execution times:**

  * Float32 Keras: \~1.17 seconds
  * Manual quantized pipeline: \~1.06 seconds
* **Percentage difference (float vs dequantized):**

  * **Maximum:** 8.33 %
  * **Mean:** 0.99 %

These results confirm that **manual quantization and dequantization preserve the integrity of the U-Net output within a low percentage error**, validating the correctness of the pipeline for FPGA preparation.

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

#### utils.py

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
