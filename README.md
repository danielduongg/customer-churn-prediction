# End-to-End Customer Churn Prediction System

## Overview

This project focuses on building an end-to-end machine learning system to predict customer churn for a telecommunications company. The goal is to identify customers at risk of churning so that targeted retention strategies can be implemented, thereby reducing customer attrition and increasing customer lifetime value.

## Skills Demonstrated

* **Languages:** Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
* **Data Tools:** Jupyter Notebooks, Git (via GitHub Desktop)
* **ML Platforms:** Scikit-learn
* **Methodologies:** Data Wrangling, Supervised Machine Learning, Model Evaluation, Feature Engineering, Basic Model Interpretation

## Dataset

The project utilizes the [Telco Customer Churn dataset from Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
This dataset contains information about a telecommunications company's customers, including their services, account information, and whether they churned. It includes 21 features and 7043 rows.

## Methodology & Workflow

The project followed a standard machine learning workflow:

### 1. Data Acquisition & Initial Loading
* Data was downloaded from Kaggle and loaded using Pandas.
* Initial inspection of data types and missing values was performed (`.info()`, `.head()`, `.describe()`).

### 2. Data Cleaning & Preprocessing
* The `TotalCharges` column was identified as an object type due to empty strings; it was converted to numeric, and missing values (representing new customers) were imputed with 0.
* The `customerID` column was dropped as it's a unique identifier and not useful for modeling.
* Inconsistent categorical values like "No internet service" and "No phone service" were standardized to "No" across relevant columns for consistency.
* The target variable `Churn` ('Yes'/'No') was converted to numerical (1/0).

### 3. Exploratory Data Analysis (EDA)
* **Target Variable Distribution:** Analyzed the balance of churn vs. non-churn customers. (e.g., Note if it was imbalanced - which it is).
* **Categorical Feature Analysis:** Used count plots to visualize churn rates across different categories (e.g., contract type, internet service, payment method).
    * **Key Insight:** Customers on **month-to-month contracts**, those with **fiber optic internet**, and those using **electronic checks** showed significantly higher churn rates.
* **Numerical Feature Analysis:** Used histograms and box plots to examine distributions of `tenure`, `MonthlyCharges`, and `TotalCharges` by churn status.
    * **Key Insight:** Customers with **shorter tenure** and **higher monthly charges** were more prone to churn.
* **Correlation Analysis:** A heatmap showed correlations between numerical features and churn.

### 4. Feature Engineering
* **One-Hot Encoding:** Applied `pd.get_dummies()` to all remaining categorical features to convert them into a numerical format suitable for machine learning models, preventing multicollinearity by dropping the first category.
* **New Feature Creation:**
    * `TotalServices`: A new feature counting the total number of services a customer subscribed to.
    * `MonthlyCharges_Tenure`: An interaction feature (MonthlyCharges * tenure) to capture the combined effect of these two variables.
* **Feature Binning:** The continuous `tenure` feature was binned into categorical groups (e.g., '0-12 M', '13-24 M') and then one-hot encoded, to potentially capture non-linear relationships.

### 5. Model Development & Evaluation
* **Data Splitting:** The dataset was split into 80% training and 20% testing sets using `train_test_split` with `stratify=y` to maintain the churn proportion.
* **Feature Scaling:** `StandardScaler` was applied to numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`, `TotalServices`, `MonthlyCharges_Tenure`) to normalize their ranges. The scaler was fit only on the training data to prevent data leakage.
* **Model Training:** Three classification models were trained:
    * Logistic Regression
    * Random Forest Classifier
    * Gradient Boosting Classifier
* **Model Evaluation:** Performance was assessed using:
    * Accuracy
    * Confusion Matrix (True Positives, True Negatives, False Positives, False Negatives)
    * Classification Report (Precision, Recall, F1-Score)
    * ROC AUC Score (Area Under the Receiver Operating Characteristic Curve) - *Chosen as the primary metric for model selection due to class imbalance.*
    * ROC Curves were plotted for visual comparison.

### 6. Model Interpretation & Basic Deployment Simulation
* **Best Model Selection:** Based on ROC AUC score, the Random Forest Classifier was identified as the best performing model.
* **Feature Importance:** The top 15 most important features for predicting churn were extracted from the Random Forest Classifier and visualized.
    * **Key Interpretations:**
        * `Contract_Month-to-month` was the most significant predictor of churn.
        * Lack of services like `OnlineSecurity_No`, `TechSupport_No` (their 'No' categories) were strong indicators of churn.
        * `tenure` (lower tenure being higher risk) and `MonthlyCharges` (higher charges being higher risk) were also crucial.
        * `InternetService_Fiber optic` also correlated with higher churn.
* **Business Recommendations:**
    * Focus retention efforts on month-to-month contract customers and new customers.
    * Promote value-added services (online security, tech support) to reduce churn.
    * Investigate reasons for high churn among fiber optic users and electronic check payment users.
    * Consider offering incentives for longer-term contracts to high-risk customers.
* **Model Saving & Simulation:** The trained Random Forest Classifier model and the `StandardScaler` were saved using `joblib` to demonstrate how they would be persisted for future use or deployment. A basic simulation showed how a loaded model could make predictions on new, preprocessed data.

## How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/customer-churn-prediction.git](https://github.com/YourUsername/customer-churn-prediction.git)
    cd customer-churn-prediction
    ```
2.  **Set up Virtual Environment (if not already done):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
3.  **Install Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter joblib
    ```
4.  **Download Dataset:**
    * Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
    * Place the downloaded `.csv` file directly into the `customer-churn-prediction` directory.
5.  **Run Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
    Open `churn_prediction_eda_modeling.ipynb` and run all cells.

## Future Enhancements

* **Hyperparameter Tuning:** Systematically optimize model parameters for better performance.
* **Advanced Feature Engineering:** Explore more complex feature interactions or external data sources.
* **Imbalanced Learning Techniques:** Implement techniques like SMOTE (Synthetic Minority Over-sampling Technique) to specifically address the class imbalance.
* **Model Deployment:** Create a simple web API (e.g., using Flask/Streamlit) to serve predictions.
* **A/B Testing:** Design and simulate A/B tests for proposed retention strategies based on model insights.

## Connect with Me

https://www.linkedin.com/in/danielmduong/
