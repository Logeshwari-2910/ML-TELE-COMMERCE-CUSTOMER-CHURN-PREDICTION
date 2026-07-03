# Customer Churn Predictor

A machine learning project for predicting telecom customer churn using supervised learning algorithms. The project includes data preprocessing, class imbalance handling, dimensionality reduction, model comparison, hyperparameter tuning, ensemble learning, and prediction on new customer data.

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
- Google Colab

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

**Dataset:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

The dataset contains **7,043** telecom customer records with demographic details, subscribed services, account information, and churn status.

Target Variable:
- **Churn**
  - 0 – No Churn
  - 1 – Churn

---

## Project Workflow

### 1. Data Preprocessing

- Removed unnecessary features
- Handled missing values
- Label encoded categorical variables
- Standardized numerical features using StandardScaler
- Balanced the dataset using SMOTE
- Applied Principal Component Analysis (PCA) for dimensionality reduction

---

### 2. Model Training

The following machine learning models were trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

Models were compared before and after hyperparameter tuning using:

- Accuracy
- Precision
- Recall
- F1-Score

---

### 3. Hyperparameter Tuning

Model performance was optimized using:

- GridSearchCV
- RandomizedSearchCV

Best parameters were selected for each model before final evaluation.

---

### 4. Ensemble Learning

The optimized models were further evaluated using ensemble techniques:

- Bagging
- Boosting (XGBoost)
- Stacking
- Soft Voting

Performance was compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

Boosting (XGBoost) achieved the best overall performance among the ensemble methods.

---

### 5. Model Serialization

The trained model and preprocessing components were saved using **Pickle**, including:

- XGBoost model
- StandardScaler
- PCA transformer
- Label Encoders

---

### 6. Prediction

The saved model was loaded to predict churn for new customer data after applying the same preprocessing steps:

- Label Encoding
- Feature Scaling
- PCA Transformation

The model outputs:

- Churn Prediction
- Prediction Probability

---

## Features

- Data preprocessing
- Feature engineering
- Class imbalance handling using SMOTE
- PCA for dimensionality reduction
- Comparison of five machine learning models
- Hyperparameter tuning
- Ensemble learning
- Confusion matrix visualization
- Model serialization using Pickle
- Prediction on new customer data

---

## How to Run

Clone the repository.

```bash
git clone https://github.com/Logeshwari-2910/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor
```

Install dependencies.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

Open **Tele-Commerce-Customer-Churn-Prediction.ipynb** using Google Colab or Jupyter Notebook and run all cells.

---

## Author

**Logeshwari**

- GitHub: https://github.com/Logeshwari-2910
- LinkedIn: https://www.linkedin.com/in/logeshwari-l-6a3b80296
