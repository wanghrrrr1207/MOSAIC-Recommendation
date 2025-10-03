# meta-review-mmoe
The source code for paper **Beyond a Single Story: Meta-Reviewing Sparse and Incomplete User-generated Contents for Recommendation** submitted to WWW'26.

This repository contains Python scripts for data preprocessing, model definition, metric computation, and end-to-end training and evaluation.

## 📁 Repository Structure

├── data/ # Dataset directory 
├── models/ # Model definitions and architecture components
├── main.py # Entry script for running experiments
├── train_eval.py # Core training and evaluation pipeline
├── metrics.py # Metric computation (RMSE, MAE)
├── requirements.txt # Dependency list for reproducibility
└── README.md # Documentation

## 🧩 File Descriptions

### `data/data_prepare.py`
- Provides dataset preprocessing and **dataloader** construction.
### `models/mmoe_attn.py`
- Implements the primary model architecture, a **Mixture-of-Experts (MMoE)** backbone augmented with an **attention mechanism**.
### `main.py`
- Acts as the entry point for the project. It initializes configurations, loads data, builds models, and triggers the training or evaluation process.  
### `train_eval.py`
- Contains the complete pipeline for **training and evaluating models**.
### `metrics.py`
- Provides implementations of evaluation metrics for recommendations.

### 1️⃣ Environment Setup
pip install -r requirements.txt
### 2️⃣ Running the Code
python main.py --use_cuda



