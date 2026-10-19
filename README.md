Customer Churn Prediction & Business Analytics System

An end-to-end machine learning project that analyzes customer characteristics, estimates churn risk, and identifies customers who may benefit from retention efforts.

This project uses synthetic customer data to demonstrate a practical machine learning and business analytics workflow.

Project Objectives

- Analyze customer characteristics and behavior.
- Predict customer churn using a Random Forest classifier.
- Estimate churn probabilities and categorize customer risk.
- Identify high-risk customers for potential retention efforts.
- Compare baseline and class-balanced models.
- Evaluate performance using accuracy, precision, recall, and F1 score.
- Generate visualizations and export prediction reports.

Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Random Forest Classification
- Git and GitHub

Dataset

The project uses 5,000 synthetic customer records, generated using "generate_data.py".

Features include:

- Customer ID
- Age
- Tenure in months
- Monthly charges
- Number of support calls
- Contract length
- Internet service type
- Churn status

The dataset is intended for educational and portfolio demonstration, not for drawing conclusions about real customers.

Machine Learning Workflow

1. Generate synthetic customer data.
2. Load the dataset and select model features.
3. Split the data into training and testing sets (80%/20%).
4. Encode the categorical internet-service feature.
5. Train a Random Forest classifier.
6. Evaluate predictions on the test set.
7. Calculate churn scores and categorize risk levels.
8. Export customer predictions and high-risk customer reports.
9. Train a class-balanced Random Forest configuration.
10. Compare model evaluation metrics and generate visualizations.

Model Performance

The following results were produced by the current script using the same test split.

Metric| Baseline Model| Class-Balanced Model
Accuracy| 75.80%| 66.50%
Precision| 46.67%| 35.96%
Recall| 20.85%| 54.47%
F1 Score| 28.82%| 43.32%

The class-balanced model has higher recall and F1 score but lower accuracy and precision. This illustrates the trade-off between detecting more churn cases and producing more false-positive alerts.

These results are from a synthetic dataset and a single train/test split. They do not establish real-world predictive performance.

Customer Risk Analysis

Each customer receives a model-generated churn score and a risk category:

- Low Risk
- Medium Risk
- High Risk

The project exports high-risk customers to a separate CSV report for further analysis.

A predicted score is an estimate produced by the model, not a guarantee that a customer will leave. The scores have not been established as calibrated real-world probabilities.

Generated Reports

- "customer_churn_predictions.csv" — customer records with predicted churn scores and risk levels.
- "high_risk_customers.csv" — customers classified as high risk.
- "model_comparison.csv" — baseline and class-balanced model metrics.

Visualizations

Churn Distribution

"Churn Distribution" (churn_distribution.png)

Monthly Charges by Churn Status

"Monthly Charges vs Churn" (monthly_charges_vs_churn.png)

Churn Rate by Contract Length

"Churn by Contract Length" (churn_by_contract_length.png)

Churn Rate by Support Calls

"Churn by Support Calls" (churn_by_support_calls.png)

Confusion Matrix

"Confusion Matrix" (confusion_matrix.png)

Project Structure

Customer-Churn-Prediction-System/
├── customer_churn_data.csv
├── customer_churn_predictions.csv
├── high_risk_customers.csv
├── model_comparison.csv
├── generate_data.py
├── churn_prediction.py
├── churn_distribution.png
├── monthly_charges_vs_churn.png
├── churn_by_contract_length.png
├── churn_by_support_calls.png
├── confusion_matrix.png
├── requirements.txt
├── .gitignore
└── README.md

Installation and Usage

Run these commands in PowerShell from the project directory.

1. Install dependencies

python -m pip install -r requirements.txt

2. Generate the synthetic dataset

python generate_data.py

3. Run the prediction system

python churn_prediction.py

The script trains both model configurations, prints evaluation metrics, saves customer prediction reports, and generates visualizations.

Note: If your "requirements.txt" includes the libraries used by the project, the installation command will install them. The generated output files may be overwritten when the script runs again.

Limitations and Responsible Use

- The dataset is synthetic and may not reflect real customer behavior.
- Performance is measured on one train/test split.
- Churn scores should not be treated as calibrated probabilities without further validation.
- High-risk classifications should support investigation, not automatically determine customer treatment.
- Additional validation, threshold tuning, and testing on representative real-world data would be required before practical deployment.

Author

Developed as a machine learning and business analytics portfolio project.
