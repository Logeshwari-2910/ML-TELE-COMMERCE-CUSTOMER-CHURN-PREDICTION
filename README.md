# Customer Churn Predictor

A machine learning project that predicts telecom customer churn using supervised learning techniques. The project includes data preprocessing, model comparison, hyperparameter tuning, and churn prediction using multiple classification algorithms.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Google Colab

---

## Dataset

This project uses the **IBM Telco Customer Churn** dataset containing **7,043 customer records**. The dataset includes customer demographics, account information, subscribed services, and churn status.

---

## Project Workflow

### Data Preprocessing
- Removed unnecessary columns.
- Handled missing values.
- Applied label encoding to categorical features.
- Performed feature scaling using `StandardScaler`.
- Balanced the dataset using **SMOTE**.
- Reduced dimensionality using **Principal Component Analysis (PCA)**.

### Model Training
The following classification algorithms were trained and evaluated:

- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- XGBoost

Model performance was compared before and after applying SMOTE.

### Hyperparameter Tuning
Optimized model performance using:

- GridSearchCV
- RandomizedSearchCV

---

## Project Structure

```
Tele-Commerce-Customer-Churn-Prediction.ipynb
README.md
dataset.csv
```

---

## Getting Started

### Clone the repository

```bash
git clone https://github.com/Logeshwari-2910/Customer-Churn-Predictor.git
cd Customer-Churn-Predictor
```

### Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn
```

---

## Running the Project

Open the notebook in **Google Colab** or **Jupyter Notebook**.

If using Google Colab, upload the dataset:

```python
from google.colab import files
uploaded = files.upload()
```

Then load the dataset:

```python
import pandas as pd

df = pd.read_csv("your_dataset.csv")
```

Alternatively, mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/path/to/dataset.csv')
```

---

## Features

- Data preprocessing and visualization
- Class imbalance handling using SMOTE
- Feature scaling and dimensionality reduction
- Comparison of multiple machine learning models
- Hyperparameter tuning
- Customer churn prediction

---

## Future Improvements

- Deploy the trained model as a web application using FastAPI or Flask.
- Build an interactive dashboard for churn prediction.
- Perform feature importance analysis and model explainability using SHAP.

---

## Author

**Logeshwari**

- GitHub: https://github.com/Logeshwari-2910
- LinkedIn: https://www.linkedin.com/in/logeshwari-l-6a3b80296
