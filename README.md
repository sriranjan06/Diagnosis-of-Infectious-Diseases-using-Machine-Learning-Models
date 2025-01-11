# Malaria Diagnosis Using Machine Learning Models
This project leverages machine learning models to classify blood cell images as either **infected** or **uninfected** with the malarial parasite. It employs various models, including SVM, KNN, ANN, and CNN, with a detailed methodology, preprocessing steps, and performance evaluation.

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Project Workflow](#project-workflow)
    - [Data Loading](#data-loading)
    - [Data Augmentation](#data-augmentation)
    - [Data Preprocessing](#data-preprocessing)
4. [Model Architectures](#model-architectures)
    - [Support Vector Machine (SVM)](#support-vector-machine-svm)
    - [K-Nearest Neighbors (KNN)](#k-nearest-neighbors-knn)
    - [Artificial Neural Network (ANN)](#artificial-neural-network-ann)
    - [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
5. [Model Evaluation](#model-evaluation)
6. [Results and Insights](#results-and-insights)
7. [References](#references)

## Introduction
Malaria is a life-threatening disease caused by the protozoan parasite *Plasmodium vivax*. This project aims to automate the diagnostic process using machine learning models trained on labeled cell images. The ultimate goal is to aid healthcare professionals by providing fast and accurate predictions.

Key highlights:
- **Dataset**: Blood cell images (infected and uninfected).
- **Approach**: Image classification using SVM, KNN, ANN, and CNN models.
- **Best Model**: Optimized CNN with 96.7% accuracy.

## Dataset Description
The dataset contains **19,291 labeled cell images**:
- **Infected cells**: Images of blood cells containing the malarial parasite.
- **Uninfected cells**: Images of normal blood cells.

### Dataset Source:
The dataset was sourced from the **National Library of Medicine (NLM)**. Each image is a stained blood smear of 3 color channels (RGB), with pixel values ranging from 0 to 255.

## Project Workflow

### 1. Data Loading
The dataset is loaded from two directories:
- `cell_images/Parasitized/`: Contains infected cell images.
- `cell_images/Uninfected/`: Contains uninfected cell images.

Each image is converted to its **RGB pixel matrix** and resized to **50x50 pixels** to ensure uniformity.

### 2. Data Augmentation
To increase the dataset size and diversity:
- **Infected Cells**:
  - Images are rotated by **45°** and **75°**.
  - Images are blurred to avoid overfitting on small blemishes.
- **Uninfected Cells**:
  - Images are rotated by **45°** and **75°**.

### 3. Data Preprocessing
- **Shuffling**: Ensures the dataset is unbiased by mixing infected and uninfected images.
- **Normalization**: Pixel values (0–255) are scaled to (0–1) using **Min-Max normalization**.
- **Train-Test Split**: The dataset is split into 80% training and 20% testing subsets.
- **Categorical Conversion**: Labels are converted to categorical format for multi-class classification.

## Model Architectures

### 1. Support Vector Machine (SVM)
- **Feature Extraction**: A CNN extracts features from input images.
- **Kernel**: Radial Basis Function (RBF) for handling non-linear data.
- **Performance**: Accuracy of 57%.

### 2. K-Nearest Neighbors (KNN)
- **Feature Extraction**: Similar to SVM, features are extracted using a CNN.
- **Hyperparameters**: 12 neighbors for classification.
- **Performance**: Accuracy of 57%.

### 3. Artificial Neural Network (ANN)
- **Architecture**:
  - Dense layers with neurons increasing from 16 to 64.
  - Dropout layers to prevent overfitting.
- **Activation Functions**:
  - **ReLU**: For intermediate layers.
  - **Softmax**: For the output layer to predict probabilities.
- **Performance**: Accuracy of 70%.

### 4. Convolutional Neural Networks (CNNs)
CNNs use convolutional layers to extract spatial features from images. Three configurations were tested:
1. **CNN-1**: Filters of 50, 100, 250; filter size 3x3.
2. **CNN-2**: Filters of 50, 100, 250; filter size 5x5.
3. **CNN-3**: Filters of 64, 128, 256; filter size 3x3 (optimized).

#### Key Features:
- **Pooling**: Max pooling reduces dimensionality while retaining important features.
- **Dropout**: Prevents overfitting by randomly dropping neurons.

## Model Evaluation

### Metrics Used:
1. **Confusion Matrix**: Visualizes true positives, false positives, etc.
2. **Accuracy**: Overall correctness of predictions.
3. **Precision and Recall**: Evaluate the relevance of predictions.
4. **F1-Score**: Balances precision and recall.

| Model  | Precision | Recall | F1-Score | Accuracy |
|--------|-----------|--------|----------|----------|
| SVM    | 0.44      | 0.57   | 0.42     | 57.09%   |
| KNN    | 0.38      | 0.57   | 0.42     | 57.10%   |
| ANN    | 0.78      | 0.70   | 0.66     | 70.06%   |
| CNN-1  | 0.96      | 0.96   | 0.96     | 96.41%   |
| CNN-2  | 0.96      | 0.96   | 0.96     | 96.41%   |
| CNN-3  | **0.97**  | **0.97**| **0.97** | **96.7%** |

## Results and Insights

1. **CNNs outperform SVM, KNN, and ANN** models by a significant margin.
2. **Optimized CNN-3** achieves the best accuracy (96.7%).
3. **Data Augmentation** is essential for improving model performance.
4. **Smaller filter sizes (3x3)** with a larger number of filters work better.

## References
- Dataset: [National Library of Medicine](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
- Publication: [Diagnosis of Malaria using Machine Learning Models](https://ieeexplore.ieee.org/document/9972568)