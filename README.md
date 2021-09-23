# Thesis Project: Thermal Image Super Resolution

This project aims to enhance the resolution of inexpensive thermal camera via deep learning. Deep learning models are supposed to be deployed on embedded platforms such as Raspberry Pi or Jetson dev kits. Some advanced optmization techniques have been applied to speed up inference on above platforms. So far, models have been successfully deployed on Jetson Xavier NX via TensorRT.

## Directory

**Pytorch-Lightning**
* Build, train and evaluate models
* Convert Pytorch models to onnx models
  
**TensorRT**
* Build, calibrate and train quantized models
* Convert quantzed models to onnx models

## Prerequisites
To run codes successfully, you will first need to install following software packages.

**Nvidia CUDA & TensorRT Packages**
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Katana: cuda-11.3.1 + cuDNN-8.2
  * Jetson: cuda-10.2 + cuDNN-8.2
* [TensorRT GA Build](https://developer.nvidia.com/nvidia-tensorrt-download) v8.0.1.6

Note: CUDA & TensorRT have been pre-installed in Jetson [JetPack SDK](https://developer.nvidia.com/embedded/jetpack). The JetPack SDK version should be >= 4.6 (TensorRT >= 8) in order to parse quantized layers.

**Python Packages**
* Pytorch-Lightning >= 1.4.0
* Pytorch >= 1.8.0
* pytorch_quantization >= 2.1.1
* onnx 1.8.0
* onnx-runtime 1.7.0
* scikit-learn
* opencv
* piq
* tensorrt (will be installed during installing the TensorRT package)
* torch2trt
* h5py
* flirpy

## Usage

Will be introduced in each directory

## Results

Models are tested by the FLIR 14-bit dataset

Model | PSNR(dB) | SSIM | FPS(RTX3080) | FPS(Jetson)
------|----------|------|--------------|------------
Bicubic | 61.042 | 0.99923
EDSR | 61.856 | 0.99939 | 10 | OOM
EDSR-TRT | 61.88 | 0.99939 | 13 | 0.6
EDSR_Quant_TRT | 61.738 | 0.99938 | 34 | 1.5
RRDBNet | NA | NA | OOM | OOM
RRDBNet-TRT
RRDBNet_Quant_TRT
