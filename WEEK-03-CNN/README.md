# Week 3 – MNIST Digit Classification Using CNN

##  Project Overview

This project focuses on building a **Convolutional Neural Network (CNN)** using TensorFlow and Keras to classify handwritten digits from the MNIST dataset.

The MNIST dataset contains grayscale images of handwritten digits ranging from 0 to 9. The CNN model learns patterns and features from these images to accurately identify the corresponding digits.

##  Objectives

- Understand the working of Convolutional Neural Networks (CNN).
- Load and preprocess the MNIST dataset.
- Perform image reshaping and normalization.
- Build a CNN model using TensorFlow and Keras.
- Train the model and evaluate its performance.
- Visualize training and validation accuracy and loss.

##  Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook / Google Colab

##  Dataset Description

**Dataset:** MNIST Handwritten Digit Dataset

The MNIST dataset consists of:

- 60,000 training images.
- 10,000 testing images.
- Image dimensions: 28 × 28 pixels.
- Image type: Grayscale.
- Number of classes: 10 (digits 0–9).

The dataset is loaded directly using TensorFlow Keras:

```python
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
```

##  Project Implementation

### 1. Importing Libraries

Imported TensorFlow, Keras layers and models, and Matplotlib for visualization.

### 2. Loading the Dataset

Loaded the MNIST dataset using the built-in TensorFlow Keras datasets module.

### 3. Data Preprocessing

- Reshaped images to include a channel dimension.
- Converted image dimensions to 28 × 28 × 1.
- Normalized pixel values from the range 0–255 to 0–1.

### 4. CNN Model Architecture

Developed a Sequential CNN model consisting of:

| Layer | Configuration |
|---|---|
| Conv2D | 32 filters, 3 × 3 kernel, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3 kernel, ReLU |
| MaxPooling2D | 2 × 2 |
| Conv2D | 64 filters, 3 × 3 kernel, ReLU |
| Flatten | Converts feature maps into a vector |
| Dense | 64 neurons, ReLU |
| Output Dense | 10 neurons, Softmax |

### 5. Model Compilation

The model was compiled using:

- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Evaluation Metric: Accuracy

### 6. Model Training

The CNN model was trained using:

- Training dataset: 60,000 images
- Validation dataset: 10,000 images (test dataset used for validation)
- Number of epochs: 5

### 7. Model Evaluation

Evaluated the trained model using the test dataset to measure its classification accuracy and loss.

### 8. Visualization

Plotted the following graphs using Matplotlib:

- Training Accuracy vs Validation Accuracy
- Training Loss vs Validation Loss

These visualizations help understand the model's learning performance across epochs.

##  Results

The CNN model achieved the following performance:

| Metric | Result |
|---|---:|
| Training Accuracy (Epoch 5) | 99.41% |
| Validation Accuracy (Epoch 5) | 98.89% |
| Test Accuracy | **98.89%** |
| Test Loss | 0.0386 |

The model successfully classified handwritten digits with approximately **98.89% test accuracy**.

##  Conclusion

This project demonstrates how Convolutional Neural Networks can be used for handwritten digit recognition.

By applying convolution, pooling, flattening, and dense layers, the model learns important image features and achieves high classification accuracy on the MNIST dataset.

The project provides practical experience in image preprocessing, CNN architecture design, model training, evaluation, and visualization.
