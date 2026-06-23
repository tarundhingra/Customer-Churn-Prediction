# Customer-Churn-Prediction


This project builds a machine learning model to predict whether a telecom customer is likely to churn.  
The notebook performs data loading, exploratory data analysis, preprocessing, class imbalance handling using SMOTE, model training, evaluation, and prediction using a saved model.

## Project Files

| File | Description |
|---|---|
| `CODE FILE.ipynb` | Main Jupyter Notebook containing the complete churn prediction workflow |
| `WA_Fn-UseC_-Telco-Customer-Churn.csv` | Telco customer churn dataset |
| `customer_churn_model.pkl` | Saved trained Random Forest model generated after running the notebook |
| `encoders.pkl` | Saved label encoders generated after preprocessing categorical features |

## Dataset Overview

The dataset contains customer-level telecom information such as:

- Demographic details: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- Account details: `tenure`, `Contract`, `PaymentMethod`, `PaperlessBilling`
- Service details: `PhoneService`, `InternetService`, `OnlineSecurity`, `TechSupport`, etc.
- Billing details: `MonthlyCharges`, `TotalCharges`
- Target column: `Churn`

Dataset size:

- Rows: 7,043
- Columns: 21
- Target classes:
  - `No`: 5,174 customers
  - `Yes`: 1,869 customers

## Objective

The goal is to predict customer churn so that a telecom company can identify customers who are likely to leave and take preventive retention actions.

## Workflow

### 1. Import Libraries

The notebook uses common data science and machine learning libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn
- XGBoost
- Pickle

### 2. Data Loading and Understanding

The CSV file is loaded into a Pandas DataFrame.  
Initial checks include:

- Dataset shape
- First few rows
- Column information
- Unique values in categorical columns
- Missing value check
- Target class distribution

The `customerID` column is dropped because it is only an identifier and does not help in model training.

### 3. Data Cleaning

The `TotalCharges` column contains blank values as strings.  
These blank values are replaced with `0.0`, and the column is converted to float.

### 4. Exploratory Data Analysis

EDA is performed using:

- Histograms for numerical features
- Box plots for numerical features
- Correlation heatmap
- Count plots for categorical features

This helps understand feature distributions, relationships, and class imbalance.

### 5. Data Preprocessing

Preprocessing steps include:

- Encoding target column:
  - `Yes` → `1`
  - `No` → `0`
- Label encoding categorical columns
- Saving encoders in `encoders.pkl`
- Splitting data into training and testing sets

### 6. Handling Class Imbalance

The dataset is imbalanced because non-churn customers are much higher than churn customers.  
SMOTE is applied only on the training data to balance the target classes.

### 7. Model Training

The following models are trained using 5-fold cross-validation:

- Decision Tree Classifier
- Random Forest Classifier
- XGBoost Classifier

Random Forest gives the best default performance in the notebook and is selected as the final model.

### 8. Model Evaluation

The final Random Forest model is evaluated on the test set using:

- Accuracy score
- Confusion matrix
- Classification report

### 9. Model Saving

The trained Random Forest model and feature names are saved using Pickle in:

```text
customer_churn_model.pkl
```

The categorical encoders are saved in:

```text
encoders.pkl
```

### 10. Predictive System

The notebook also includes a sample prediction system where a new customer record is passed as input.  
The saved encoders transform categorical values, and the trained model predicts whether the customer will churn.

## How to Run the Project

### Step 1: Install Required Libraries

Run the following command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Step 2: Place Dataset in the Correct Location

Keep the dataset file in the same folder as the notebook:

```text
WA_Fn-UseC_-Telco-Customer-Churn.csv
```

If you are running the notebook locally, update this line:

```python
df = pd.read_csv("/content/WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

to:

```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
```

### Step 3: Run the Notebook

Open the notebook and run all cells in order:

```bash
jupyter notebook "CODE FILE.ipynb"
```

or upload the notebook to Google Colab and run it there.

## Sample Prediction Input

```python
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}
```

The output will show:

```text
Prediction: Churn / No Churn
Prediction Probability: [...]
```

## Key Insights

- Customers with month-to-month contracts are more likely to churn.
- Customers with shorter tenure are more likely to churn.
- Higher monthly charges may be associated with churn.
- The dataset has class imbalance, so SMOTE is useful before training.
- Random Forest performs best among the tested default models.

## Future Improvements

- Perform hyperparameter tuning
- Try Stratified K-Fold Cross Validation
- Compare more models such as Logistic Regression, LightGBM, and CatBoost
- Use feature scaling where required
- Try downsampling or other imbalance handling methods
- Improve overfitting control
- Deploy the model using Flask, FastAPI, or Streamlit

## Conclusion

This project demonstrates an end-to-end machine learning pipeline for customer churn prediction.  
It includes data cleaning, visualization, preprocessing, class balancing, model training, evaluation, model saving, and prediction on new customer data.
