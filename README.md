# 💳 Credit Card Fraud Detection using Machine Learning

A machine learning-based system for detecting potentially fraudulent credit card transactions.

The project uses data preprocessing, feature scaling, SMOTE for handling class imbalance, and machine learning classification models. A Streamlit web application is also provided for making predictions.

---

## 📌 Project Overview

Credit card fraud is a major problem in digital financial transactions. Since fraudulent transactions are much fewer than legitimate transactions, detecting them accurately is a challenging machine learning problem.

This project develops a machine learning pipeline that:

- Cleans and preprocesses transaction data
- Performs exploratory data analysis
- Handles class imbalance using SMOTE
- Scales numerical features using StandardScaler
- Trains multiple classification models
- Evaluates model performance
- Uses Random Forest for the final prediction system
- Provides an interactive Streamlit web application

---

## 🎯 Objectives

- Detect fraudulent credit card transactions.
- Handle highly imbalanced transaction data.
- Compare different machine learning algorithms.
- Evaluate the models using suitable classification metrics.
- Build a simple web interface for fraud prediction.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Joblib
- Streamlit
- Jupyter Notebook

---

## 🤖 Machine Learning Algorithms

### Logistic Regression

Used as a baseline classification model for comparing performance.

### Random Forest

Used as the main classification model for the final fraud detection application.

### SMOTE

Synthetic Minority Oversampling Technique (SMOTE) is used to balance the highly imbalanced training data by generating synthetic samples for the minority class.

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset**.

The original dataset contains:

- 284,807 transactions
- 30 input features
- 1 target variable (`Class`)

### Target Variable

| Class | Meaning |
|---|---|
| 0 | Normal Transaction |
| 1 | Fraudulent Transaction |

After removing duplicate records:

- 283,726 transactions
- 473 fraudulent transactions
- 283,253 normal transactions

The dataset is highly imbalanced, which makes fraud detection challenging.

### Features

The dataset contains:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

The `V1`–`V28` features are anonymized numerical features obtained from a transformed version of the original transaction data.

---

## 🔄 Project Workflow

```text
Credit Card Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Train/Test Split
        ↓
Feature Scaling
        ↓
SMOTE Balancing
        ↓
┌───────────────────────┐
│                       │
Logistic Regression   Random Forest
│                       │
└───────────┬───────────┘
            ↓
     Model Evaluation
            ↓
      Random Forest
            ↓
     Streamlit Web App
            ↓
 Fraud / Normal + Probability
