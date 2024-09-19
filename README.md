## Overview
This project aims to develop a machine learning model for signature verification. It can distinguish between genuine and forged signatures by extracting features from images and training a classification model.

## Directory Structure

Signature Verification/
│
├── genuine_signatures/          # Folder containing genuine signature images 
│
├── test_signatures/             # Folder containing test signature images (potentially forged) 
│
├── features/                    # Folder to store extracted feature CSV files 
│   ├── genuine_features.csv      # Features extracted from genuine signatures 
│   └── test_features.csv         # Features extracted from test signatures 
│
├── main.py                      # Main script for the project 
└── requirements.txt             # File listing project dependencies 



## Getting Started

### Prerequisites
Make sure you have the following installed:
- Python 3.x
- Jupyter Notebook
- pip (Python package installer)

### Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
Navigate to the project directory:
bash

cd Signature Verification
**Install the required packages:**
bash

pip install -r requirements.txt
**Usage**
Feature Extraction and Model Training
Open the main.ipynb notebook in Jupyter Notebook and follow the instructions to extract features from genuine and test signatures, and train the model.

**Signature Comparison**
After training, you can compare a new signature by providing its path when prompted in the notebook.
