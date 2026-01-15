# ALMS_assignment25_26
UCL ELEC0134 AMLS Assignment 25/26
Student Number:25046280  
Dataset: BreastMNIST (Binary Classification: Benign vs. Malignant)

## 1. Project Overview
This project benchmarks two different machine learning approaches for classifying breast ultrasound images using the BreastMNIST dataset. The goal is to analyze how model capacity, data augmentation, and feature engineering impact performance.

Model A (Classical ML): Support Vector Machine (SVM). Compares performance between raw pixel features and Histogram of Oriented Gradients (HOG) features, optimized via GridSearch.
Model B (Deep Learning): Convolutional Neural Network (CNN). Investigates the impact of advanced data augmentation and learning rate scheduling on a custom CNN architecture.

2. Project Structure
The project is organized into modules as follows:


AMLS_25-26_25046280/
├── Code/
│   ├── Model_A/
│   │   └── svm_classifier.py   # Implementation of SVM (Raw vs HOG)
│   └── Model_B/
│   │   └── cnn_classifier.py   # Implementation of CNN (Training & Eval)
├── Datasets/                   # Empty folder (Data downloads automatically)
├── Results/                    # Generated plots and training logs
├── main.py                     # Main entry point to run the entire project
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project documentation