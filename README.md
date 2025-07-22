# Credit Card Fraud Detection System 🔒

A comprehensive machine learning project for detecting fraudulent credit card transactions using multiple algorithms including Logistic Regression, Shallow Neural Networks, and Convolutional Neural Networks (CNNs).

## 📊 Project Overview

This project implements and compares three different machine learning approaches to detect credit card fraud:

1. **Logistic Regression** - Classical statistical approach
2. **Shallow Neural Network** - Deep learning with fully connected layers
3. **Convolutional Neural Network (CNN)** - Advanced deep learning approach

## 🎯 Features

- **Data Preprocessing**: Robust scaling and normalization
- **Multiple Models**: Comparison of different ML approaches
- **Visualization**: Training history plots and data distribution analysis
- **Performance Metrics**: Accuracy, F1-score, and detailed classification reports
- **Model Persistence**: Automated model saving with checkpoints

## 📁 Project Structure

```
fraud-detection/
│
├── frauddetection.py          # Main implementation file
├── requirements.txt           # Python dependencies
├── README_FRAUD_DETECTION.md  # Project documentation
│
├── models/                    # Saved trained models
│   ├── shallow_nn.keras
│   └── cnn_model.keras
│
├── plots/                     # Generated visualizations
│   ├── data_distribution.png
│   ├── shallow_neural_network_training_history.png
│   └── cnn_training_history.png
│
├── data/                      # Data directory
│   └── creditcard.csv         # Dataset (download required)
│
└── notebooks/                 # Jupyter notebooks
    └── fraud_detection_analysis.ipynb
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Kaggle API credentials (for dataset download)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/fraud-detection.git
   cd fraud-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kaggle API** (Required for dataset download)
   ```bash
   # Download kaggle.json from your Kaggle account settings
   # Place it in ~/.kaggle/ directory
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Download the dataset**
   ```bash
   kaggle datasets download -d mlg-ulb/creditcardfraud
   unzip creditcardfraud.zip
   ```

### Running the Project

**Option 1: Run the complete analysis**
```bash
python frauddetection.py
```

**Option 2: Use as a module**
```python
from frauddetection import FraudDetectionSystem

# Initialize the system
detector = FraudDetectionSystem()

# Load and process data
detector.load_and_prepare_data()
detector.explore_data()
detector.preprocess_data()
detector.split_data()

# Train models
X_train, y_train, X_test, y_test, X_val, y_val = detector.prepare_features()
lr_model = detector.train_logistic_regression(X_train, y_train, X_val, y_val)
```

## 📈 Dataset Information

- **Source**: [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 30 (28 anonymized features + Time + Amount)
- **Target**: Binary classification (0: Normal, 1: Fraud)
- **Imbalance**: ~0.17% fraudulent transactions

### Dataset Features

| Feature | Description |
|---------|-------------|
| Time | Seconds elapsed between transactions |
| V1-V28 | Anonymized features from PCA transformation |
| Amount | Transaction amount |
| Class | Target variable (0: Normal, 1: Fraud) |

## 🔬 Methodology

### Data Preprocessing
1. **Duplicate Removal**: Remove duplicate transactions
2. **Feature Scaling**: RobustScaler for 'Amount' feature
3. **Time Normalization**: Min-max normalization for 'Time' feature
4. **Data Split**: 70% Train, 15% Test, 15% Validation

### Model Architectures

#### 1. Logistic Regression
- Simple linear classification model
- Good baseline for comparison
- Fast training and interpretation

#### 2. Shallow Neural Network
- **Architecture**:
  - Input Layer: 30 features
  - Hidden Layer: 2 neurons + ReLU + BatchNormalization
  - Output Layer: 1 neuron + Sigmoid
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy

#### 3. Convolutional Neural Network
- **Architecture**:
  - Conv1D layers: 32 and 64 filters
  - MaxPooling1D layers for dimension reduction
  - Dropout layers (0.5) for regularization
  - Dense layers: 128 → 1 neurons
- **Input Shape**: (30, 1) - treating features as 1D sequence
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Logistic Regression | ~99.9% | ~0.85 | Fast |
| Shallow Neural Network | ~99.9% | ~0.87 | Medium |
| CNN | ~99.9% | ~0.88 | Slow |

### Key Insights
- All models achieve high accuracy due to dataset imbalance
- F1-score is more meaningful for this imbalanced dataset
- CNN slightly outperforms other models in F1-score
- Logistic Regression provides fastest training with competitive results

## 📋 Requirements

See `requirements.txt` for complete list:

```
pandas>=1.3.3
numpy>=1.21.0
scikit-learn>=1.0.0
tensorflow>=2.7.0
matplotlib>=3.4.3
seaborn>=0.11.2
kaggle>=1.5.12
```

## 🎮 Usage Examples

### Basic Usage
```python
# Import the system
from frauddetection import FraudDetectionSystem

# Initialize
detector = FraudDetectionSystem()

# Complete pipeline
detector.load_and_prepare_data('creditcard.csv')
detector.explore_data()
detector.preprocess_data()
detector.split_data()

# Get features
X_train, y_train, X_test, y_test, X_val, y_val = detector.prepare_features()

# Train a model
model = detector.train_logistic_regression(X_train, y_train, X_val, y_val)

# Evaluate
accuracy, f1, predictions = detector.evaluate_model(
    model, X_test, y_test, 'Logistic Regression', is_neural_net=False
)
```
---

⭐ **Star this repository if you found it helpful!** ⭐
