# Customer Churn Predictor

A machine learning project that predicts telecom customer churn using supervised learning techniques. The project covers data preprocessing, feature engineering, class imbalance handling, dimensionality reduction, model comparison, hyperparameter tuning, ensemble learning, model serialization, and prediction on new customer data.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- imbalanced-learn (SMOTE)
- Google Colab / Jupyter Notebook

---

## Repository Structure

```
Customer-Churn-Predictor/
│── Tele-Commerce-Customer-Churn-Prediction.ipynb
│── WA_Fn-UseC_-Telco-Customer-Churn.csv
│── README.md
```

---

## Dataset

**Dataset File**

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

The dataset contains **7,043** telecom customer records with customer demographics, subscribed services, account information, and churn status.

**Target Variable**

- **0** → No Churn
- **1** → Churn

---

# Project Workflow

The notebook performs the following steps:

### 1. Data Loading

- Loads the telecom customer churn dataset.
- Explores the dataset structure and data types.

### 2. Data Preprocessing

- Removes unnecessary columns.
- Handles missing values.
- Encodes categorical features using Label Encoding.
- Standardizes numerical features using StandardScaler.
- Handles class imbalance using SMOTE.
- Applies Principal Component Analysis (PCA) to reduce dimensionality.

### 3. Model Training

The following machine learning models are trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

### 4. Hyperparameter Tuning

Model performance is optimized using:

- GridSearchCV
- RandomizedSearchCV

### 5. Model Evaluation

Models are evaluated and compared using:

- Accuracy
- Precision
- Recall
- F1-Score

Performance is compared before and after hyperparameter tuning.

### 6. Ensemble Learning

The notebook evaluates the following ensemble techniques:

- Bagging
- Boosting (XGBoost)
- Stacking
- Voting

Each ensemble model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

### 7. Model Serialization

The trained model and preprocessing components are saved using Pickle.

Saved files include:

- `xgboost_pca_model.pkl`
- `scaler.pkl`
- `pca.pkl`
- `encoders.pkl`

Equivalent `.sav` files are also generated.

### 8. Prediction

The notebook demonstrates how to:

- Load the saved model.
- Load the scaler, PCA transformer, and label encoders.
- Preprocess new customer data.
- Predict customer churn.
- Display prediction probabilities.

---

# Running the Project

## Option 1: Google Colab

### Step 1

Download or clone this repository.

### Step 2

Open **Tele-Commerce-Customer-Churn-Prediction.ipynb** in Google Colab.

### Step 3

Provide the dataset using one of the following methods.

### Method A: Upload the Dataset

Upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to the Colab session.

If the notebook contains:

```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

no changes are required after uploading the dataset.

### Method B: Load the Dataset from Google Drive

Upload the dataset to Google Drive.

Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Update the dataset path:

```python
import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

Replace `path/to/` with the actual location of the dataset in your Google Drive.

### Step 4

Install any missing libraries if prompted.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### Step 5

Run every notebook cell sequentially from top to bottom.

---

## Option 2: Jupyter Notebook

### Step 1

Place the notebook and dataset in the same directory.

```
Tele-Commerce-Customer-Churn-Prediction.ipynb
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

### Step 2

Open the notebook using Jupyter Notebook or JupyterLab.

### Step 3

If your dataset is stored in another location, update the path accordingly.

Example:

```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

or

```python
df = pd.read_csv("path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

### Step 4

Install any missing libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### Step 5

Run all notebook cells sequentially.

---

# Saving the Trained Model

The notebook demonstrates how to save the trained model and preprocessing components using Pickle.

Saved files include:

```
xgboost_pca_model.pkl
scaler.pkl
pca.pkl
encoders.pkl
```

The notebook also demonstrates saving the same objects as `.sav` files:

```
xgboost_pca_model.sav
scaler.sav
pca.sav
encoders.sav
```

These files can be reused later without retraining the model.

---

# Using the Saved Model

The notebook includes an example showing how to:

1. Load the saved model.
2. Load the StandardScaler, PCA transformer, and label encoders.
3. Prepare new customer data.
4. Apply the same preprocessing steps used during training.
5. Predict whether the customer will churn.
6. Obtain the prediction probability.

This enables predictions on new customer data without retraining the model.

---

# Features

- Data preprocessing
- Label encoding
- Feature scaling
- Class imbalance handling using SMOTE
- Principal Component Analysis (PCA)
- Comparison of multiple machine learning models
- Hyperparameter tuning
- Ensemble learning
- Confusion matrix visualization
- Model serialization using Pickle
- Prediction on new customer data

---

# Author

**Logeshwari**

- GitHub: https://github.com/Logeshwari-2910
- LinkedIn: https://www.linkedin.com/in/logeshwari-l-6a3b80296
