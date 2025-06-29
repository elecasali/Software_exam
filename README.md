# Documentation for U-net Project

**Author**: Elena Casali

---

## Table of Contents

- [Introduction](#introduction)
  - [Contributions](#contributions)
- [Repository structure](#repository-structure)
  - [`unet.py`](#unetpy)
  - [`utils/`](#utils)
    - [utils.py](#utilspy)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Introduction

Neural Networks (NNs) are computational models inspired by the structure of biological neural networks, designed to identify patterns, weight options, and reach decisions through interconnected layers of neurons. Each neuron receives weighted inputs, applies a bias, and passes its output to subsequent neurons through an activation function, enabling complex hierarchical representations. NNs form the foundation of deep learning, finding extensive application in medical imaging and segmentation tasks but often require significant computational resources due to the reliance on floating-point operations and multi-layer structures.

A specialized architecture for image segmentation is the **U-Net**, which adopts an encoder-decoder structure. The **encoder** progressively contracts the spatial dimensions of the input while extracting abstract features, followed by a **bottleneck** that represents the point of maximum compression of information. The **decoder** then reconstructs the image to the original input resolution by expanding the feature maps, while **skip connections** link corresponding layers in the encoder and decoder, preserving spatial details and enhancing segmentation precision. Upsampling within the decoder increases spatial resolution, and concatenation operations merge feature maps from different layers to retain fine-grained spatial information.

The goal of this project is to **quantize the U-Net** architecture following the quantization method presented in *Xiong et al., BMC Bioinformatics (2021) 22:421*, and to evaluate its computational feasibility and numerical correctness. The quantization reduces input and weights to 8-bit signed integers while maintaining them on 16-bit to avoid overflow, followed by layer-wise rescaling to propagate values through the network. The primary objective is to verify that quantization can be successfully applied to the U-Net from a computational perspective, comparing the outputs of the float model and the dequantized model to assess numerical alignment.

It is important to note that **the network has not been trained**, and the **biases, weights, and input matrices are randomly generated numbers**. This approach is intentional, as the project focuses on verifying the implementation of quantization and its compatibility with hardware-level simulation, rather than evaluating segmentation accuracy. as the purpose of the project is to verify the successful implementation of quantization so that it can later be expanded and applied to more complex networks. This would enable their implementation on FPGA with **significantly reduced execution times while maintaining a good level of precision**, providing a scalable and efficient foundation for deploying larger networks in hardware environments.

## Repository structure

```
├── unet.py            # Main script: builds float32 U-Net, performs quantization to int16, and dequantization
├── utils/             # Folder with helper functions for manual layer-wise quantization
│   ├── __init__.py    # marks the folder as a Python package
│   └── utils.py       # manual implementations of U-Net layer operations
├── env.yml            # Conda environment definition with all dependencies
└── README.md          # Project overview, instructions, and design rationale
```


### unet.py

### utils/

#### utils.py

## Installation

## Usage

## Dependencies

Te dependencies for te project are contained in te file 'env.yml'. To run te project te followin commands can be used 

```
conda env create -f env.yml -n Casali_Fendillo_exam
conda activate Casali_Fendillo_exam
pyton unet.py
```
