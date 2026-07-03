# Customer Churn Predictor

A machine learning project that predicts telecom customer churn using supervised learning techniques. The project covers data preprocessing, feature engineering, dimensionality reduction, model comparison, hyperparameter tuning, ensemble learning, model serialization, and prediction on new customer data.

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

The dataset contains **7,043 telecom customer records** with customer demographics, subscribed services, account information, and churn status.

Target Variable:

- **0** → No Churn
- **1** → Churn

---

# Project Workflow

The notebook performs the following steps:

### 1. Data Loading

- Loads the telecom customer churn dataset.
- Explores dataset structure and data types.

### 2. Data Preprocessing

- Handles missing values.
- Removes unnecessary columns.
- Encodes categorical features using Label Encoding.
- Standardizes numerical features using StandardScaler.
- Handles class imbalance using SMOTE.
- Applies Principal Component Analysis (PCA) for dimensionality reduction.

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

Models are compared using:

- Accuracy
- Precision
- Recall
- F1-Score

Performance is evaluated before and after hyperparameter tuning.

### 6. Ensemble Learning

The notebook evaluates the following ensemble methods:

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

The notebook saves the trained model and preprocessing components using Pickle.

Saved files include:

- `xgboost_pca_model.pkl`
- `scaler.pkl`
- `pca.pkl`
- `encoders.pkl`

The notebook also demonstrates saving the same objects as `.sav` files.

### 8. Prediction

The notebook demonstrates how to:

- Load the saved model.
- Load the scaler, PCA transformer, and encoders.
- Preprocess new customer data.
- Predict customer churn.
- Display prediction probabilities.

---

# How to Run

## Using Google Colab

### Step 1

Download or clone this repository.

### Step 2

Open

```
Tele-Commerce-Customer-Churn-Prediction.ipynb
```

using Google Colab.

### Step 3

Upload

```
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

to the Colab session.

### Step 4

Install any missing Python packages if prompted.

### Step 5

Run every notebook cell from top to bottom.

The notebook will automatically:

- preprocess the data
- train multiple machine learning models
- perform hyperparameter tuning
- compare model performance
- evaluate ensemble methods
- save the trained model
- predict churn for new customer data

---

## Using Jupyter Notebook

1. Place the notebook and dataset in the same folder.

```
Tele-Commerce-Customer-Churn-Prediction.ipynb
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

2. Open the notebook.

3. Run all cells sequentially.

---

# Saving the Trained Model

The notebook saves the trained XGBoost model together with the preprocessing objects using Pickle.

Generated files include:

```
xgboost_pca_model.pkl
scaler.pkl
pca.pkl
encoders.pkl
```

Equivalent `.sav` files are also generated.

These files can be reused later without retraining the model.

---

# Using the Saved Model

The notebook includes an example demonstrating how to:

- Load the saved model.
- Load the scaler, PCA transformer, and label encoders.
- Prepare new customer data.
- Apply the same preprocessing steps.
- Predict customer churn.
- Obtain prediction probabilities.

This allows new customer records to be classified without retraining the model.

---

# Features

- Data preprocessing
- Feature engineering
- Label encoding
- Feature scaling
- SMOTE
- Principal Component Analysis (PCA)
- Hyperparameter tuning
- Model comparison
- Ensemble learning
- Confusion matrix visualization
- Model serialization using Pickle
- Prediction on new customer data

---

# Author

**Logeshwari**

GitHub: https://github.com/Logeshwari-2910

LinkedIn: https://www.linkedin.com/in/logeshwari-l-6a3b80296
