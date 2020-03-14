### Representation Learning with Contrastive Predictive Coding for images

This repository contains a Keras implementation of contrastive-predictive-coding for **images**, an algorithm fully described in:
* [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748).
* [Data-Efficient Image Recognition with Contrastive Predictive Coding](https://arxiv.org/abs/1905.09272).

The goal of unsupervised representation learning is to capture semantic information about the world, recognizing regular patterns in the data without using annotations. This paper presents a new method called Contrastive Predictive Coding (CPC) that can do so across multiple applications. This implementation covers the case of CPC applied to images (vision).

In a nutshell, an input image is divided into a grid of overlapping patches, and each patch is embedded using an encoding network. From these embeddings, the model computes a context vector at each position using masked convolutions (do not have access to future pixels). These context vectors propagate and integrate spatial information. Given a row in the grid of vectors, the model predicts entire rows below at different offsets (2 rows below, 3 rows below, etc.). These predictions should match the embedded vectors computed at the very beginning. The model is optimized to find the correct embedding when vectors from other patches are considered.

My code is optimized for readability and it is meant to be used as a resource to understand how CPC works. Therefore, I would like to explain a few concepts that I found challenging to understand in the papers.


<p align="center">
<img src="/resources/context.png" alt="CPC algorithm - context" height="150">
</p>

In this figure, horizontal lines in the left represent embedded input image rows (7 embedding vectors), and triangles correspond to masked convolutions. We can see how all the orange rows (0 to 4) contribute to the context vector of row 3 (center). Although masked convolutions prevent information from lower rows to flow to the context, notice how input row 4 makes its way to the end. This is because patches are extracted with overlapping, that is, patches encoded in row 3 (left) contain pixels from row 4 below (in yellow). For this reason, we should never optimize the CPC model to predict just one row below, since the information would flow directly from the input preventing it from learning anything useful.

Once the context vectors are computed, we can proceed with the actual row predictions. We cannot predict the row below, but we can predict the rest. Depending on how far we would like to predict (offset), we will use different prediction networks. In total, there are 5 prediction networks.


<p align="center">
<img src="/resources/offsets.png" alt="CPC algorithm - offset" height="300">
</p>

Given a set of context vectors, we apply each prediction network to all rows, however, not all predictions will be used. These predictions are aligned with the embedded input image rows taking into account the row offset used during the prediction (see bottom of the figure for a detailed mapping). This operation is performed in ```cpc_model.py > CPCLayer > align_embeddings()```.

To train the CPC algorithm, I have created a toy dataset based on 64x64 color MNIST.

<p align="center">
<img src="/resources/samples.png" alt="CPC algorithm - samples" height="100">
</p>

Disclaimer: this code is provided *as is*, if you encounter a bug please report it as an issue. Your help will be much welcomed!

### Usage

- Execute ```python train_cpc.py``` to train the CPC model.
- Execute ```python train_classifier.py``` to train a classifier on top of the CPC encoder.

### Requisites

- [Anaconda Python 3.5.3](https://www.continuum.io/downloads)
- [Keras 2.0.6](https://keras.io/)
- [Tensorflow 1.4.0](https://www.tensorflow.org/)
- GPU for fast training.
