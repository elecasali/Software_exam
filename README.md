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

Neural Networks (NNs) are computational models inspired by the structure of biological neural networks, designed to identify patterns, weight options, and reach decisions through interconnected layers of neurons [[1]](#references). Each neuron receives weighted inputs, applies a bias, and passes its output to subsequent neurons through an activation function, enabling complex hierarchical representations. NNs form the foundation of deep learning, finding extensive application in medical imaging and segmentation tasks but often require significant computational resources due to the reliance on floating-point operations and multi-layer structures [[2]](#references)[[3]](#references). 

A specialized architecture for image segmentation is the **U-Net**, which adopts an encoder-decoder structure. The **encoder** progressively contracts the spatial dimensions of the input while extracting abstract features, followed by a **bottleneck** that represents the point of maximum compression of information. The **decoder** then reconstructs the image to the original input resolution by expanding the feature maps, while **skip connections** link corresponding layers in the encoder and decoder, preserving spatial details and enhancing segmentation precision. Upsampling within the decoder increases spatial resolution, and concatenation operations merge feature maps from different layers to retain fine-grained spatial information. 

The goal of this project is to **quantize** NN architectures following the quantization method presented in *Xiong et al.* [[2]](#references), and to evaluate its computational feasibility and numerical correctness. The quantization reduces input and weights to 8-bit signed integers while maintaining them on 16-bit to avoid overflow, followed by layer-wise rescaling to propagate values through the network. The primary objective is to verify that quantization can be successfully applied to a CNN and a U-Net from a computational perspective, comparing the outputs of the float model and the dequantized model to assess numerical alignment. Furthermore, we provide the execution time so that, once implemented in VHDL firmware, it can be evaluated in comparison.  

It is important to note that **the network has not been trained**, and the **biases, weights, and input matrices are randomly generated numbers**. This approach is intentional, as the project focuses on verifying the implementation of quantization and its compatibility with hardware-level simulation, rather than evaluating segmentation accuracy. The purpose of the project is to verify the successful implementation of quantization so that it can later be expanded and applied to more complex networks. This would enable their implementation on FPGA with **significantly reduced execution times while maintaining a good level of precision**, providing a scalable and efficient foundation for deploying larger networks in hardware environments. 

These codes are complementary to the FPGA implementation in VHDL, not provided in this repository. 

## Contributions 

This project is a joint effort between E. Casali and L. Fendillo (shortened E.C. and L.F.).  

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

### test_firmware.py

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
