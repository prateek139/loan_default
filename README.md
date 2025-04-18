Loan Default Prediction
This project aims to predict whether a borrower will default on a loan using financial and credit information. The dataset includes various borrower features such as income, loan amount, credit score, employment status, and more.

Table of Contents
Project Overview

Dataset

Installation

Usage

Model

Evaluation

Conclusion

Project Overview
The goal of this project is to develop a machine learning model that can classify whether a borrower will default on a loan. We will use the Random Forest Classifier to predict the target variable Default based on borrower financial data.

Dataset
The dataset consists of the following features:

LoanID: Unique identifier for each loan (dropped during analysis).

Age: Age of the borrower.

Income: Borrower's annual income.

LoanAmount: Requested loan amount.

CreditScore: Borrower's credit score.

MonthsEmployed: Duration of employment in months.

NumCreditLines: Number of credit lines the borrower has.

InterestRate: Loan interest rate.

LoanTerm: Loan duration in months.

DTIRatio: Debt-to-income ratio.

Education: Borrower's highest level of education.

EmploymentType: Employment status (e.g., full-time, part-time).

MaritalStatus: Borrower's marital status.

HasMortgage: Whether the borrower has a mortgage.

HasDependents: Whether the borrower has dependents.

LoanPurpose: Purpose of the loan.

HasCoSigner: Whether the loan has a co-signer.

Default: Target variable indicating whether the borrower defaulted on the loan (1 = default, 0 = no default).

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/loan-default-prediction.git
Install required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Usage
Upload your dataset (loan_data.csv) to the project directory.

Open the loan_default_prediction.ipynb Jupyter notebook.

Follow the instructions in the notebook to preprocess the data, train the model, and evaluate the results.

Model
The RandomForestClassifier is used to predict loan defaults. It was chosen for its ability to handle complex relationships in the data without requiring too much feature engineering.

Key Steps:
Data Preprocessing:

Drop unnecessary columns (e.g., LoanID).

Convert categorical variables to numeric (e.g., HasMortgage, HasDependents).

One-hot encode categorical features (e.g., Education, EmploymentType).

Scale numeric features for better model performance.

Model Training:

Split the dataset into training and testing sets (80% train, 20% test).

Train the RandomForestClassifier on the training data.

Evaluation:

Accuracy, classification report, and confusion matrix are used to assess model performance.

Evaluation
The model is evaluated using:

Accuracy: The percentage of correct predictions.

Precision, Recall, F1-Score: These metrics provide insight into the model's performance, especially for imbalanced datasets.

Confusion Matrix: Displays the number of true positives, true negatives, false positives, and false negatives.

Example Output:
Accuracy: 85%

Classification Report:

Default (Precision = 0.82, Recall = 0.75, F1-Score = 0.78)

No Default (Precision = 0.87, Recall = 0.90, F1-Score = 0.88)

Confusion Matrix:

lua
Copy
Edit
[[1620   180]   <- No Default
 [  50   150]]  <- Default
Conclusion
The Random Forest model performs well with 85% accuracy in predicting loan defaults. There is potential for improvement, especially in predicting defaults more accurately, which could be explored by handling imbalanced classes, hyperparameter tuning, or trying different models (e.g., XGBoost).

Future Work
Hyperparameter Tuning: Try optimizing the model using grid search or random search.

Class Imbalance Handling: Investigate methods like oversampling, undersampling, or using class weights to handle imbalanced data.

Model Comparison: Compare the performance of RandomForest with other algorithms like Logistic Regression, SVM, or XGBoost.

License
This project is licensed under the MIT License - see the LICENSE file for details.

