# Brain Tumor Classification with Deep Learning

This repository contains a deep learning model for classifying brain tumors using MRI images. The model is trained on a dataset of brain MRI scans and can classify tumors into four categories: glioma, meningioma, pituitary, and no tumor.

## Dataset

The dataset used in this project is a collection of brain MRI scans obtained from Kaggle (https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). It consists of images belonging to four different classes: glioma, meningioma, pituitary, and no tumor. The images are preprocessed and converted into NumPy arrays for efficient loading and training. Each picture is getting transformed into grayscale, reshaped into a one-dimensional array, and normalized.

## Model

The model is a feedforward neural network implemented using PyTorch. It consists of three fully connected layers with batch normalization and regularization methods such as dropout layers, Kaiming, and weight-decay. The model is trained using the Adam optimizer and cross-entropy loss function.

## SVM Classifier

Originally, I developed a pipeline with scikit learn that utilized a SVM classifier. Parameters were optimized through gridsearch and I applied cross validation.

