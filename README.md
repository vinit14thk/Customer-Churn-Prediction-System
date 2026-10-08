\# Customer Churn Prediction \& Business Analytics System



\## Overview



A machine learning-based customer churn prediction system that analyzes customer behavior, predicts the likelihood of churn, and identifies high-risk customers for potential retention efforts.



The project uses a synthetic customer dataset and a Random Forest classification model to demonstrate an end-to-end machine learning workflow.



\## Problem Statement



Customer churn is an important business problem because losing customers can reduce revenue and increase customer acquisition costs.



This project aims to:



\- Analyze customer characteristics and behavior

\- Predict whether a customer is likely to churn

\- Estimate churn probability

\- Classify customers into risk levels

\- Identify high-risk customers

\- Evaluate machine learning model performance

\- Generate business-oriented visualizations



\## Technologies Used



\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Matplotlib

\- Seaborn

\- Random Forest

\- Git \& GitHub



\## Dataset



The project uses a synthetic dataset containing \*5,000 customer records\*.



Features include:



\- Customer ID

\- Age

\- Tenure in months

\- Monthly charges

\- Number of support calls

\- Contract length

\- Internet service type

\- Churn status



The dataset is generated programmatically using generate\_data.py.



\## Machine Learning Methodology



The system follows these steps:



1\. Generate synthetic customer data

2\. Load and preprocess the dataset

3\. Separate features and target variable

4\. Split the data into training and testing sets

5\. Encode categorical features

6\. Train a Random Forest classification model

7\. Predict customer churn

8\. Evaluate model performance

9\. Calculate churn probabilities

10\. Identify high-risk customers

11\. Generate analytical visualizations



\## Model



The project uses a \*Random Forest Classifier\* with 200 decision trees.



The dataset is divided into:



\- 80% training data

\- 20% testing data



\### Initial Model Performance



| Metric | Score |

|---|---:|

| Accuracy | 75.80% |

| Precision | 46.67% |

| Recall | 20.85% |

| F1 Score | 28.82% |



These results represent the initial model configuration and provide a baseline for future model improvement.



After evaluating the baseline model, the Random Forest configuration was improved using class balancing and additional hyperparameters.



| Metric | Baseline | Improved |

|---|---:|---:|

| Accuracy | 75.80% | 66.50% |

| Precision | 46.67% | 35.96% |

| Recall | 20.85% | 54.47% |

| F1 Score | 28.82% | 43.32% |



The improved model substantially increased recall and F1 score, making it better suited for identifying customers who may be at risk of churn. This comes with a trade-off in overall accuracy and precision.



\## Risk Analysis



The system calculates a churn probability for every customer.



Customers are categorized into:



\- \*Low Risk\*

\- \*Medium Risk\*

\- \*High Risk\*



A separate high\_risk\_customers.csv file is generated containing customers classified as high risk.



The churn probability is a model-generated risk estimate and should not be interpreted as a guaranteed probability of customer departure.



\## Visualizations



\### Churned vs Non-Churned Customers



Shows the distribution of customers who stayed versus customers who churned.



!\[Churn Distribution](churn\_distribution.png)



\### Monthly Charges vs Churn



Compares monthly charge distributions between customers who stayed and customers who churned.



!\[Monthly Charges vs Churn](monthly\_charges\_vs\_churn.png)



\### Churn Rate by Contract Length



Shows how churn rates vary across different contract lengths.



!\[Churn by Contract Length](churn\_by\_contract\_length.png)



\### Churn Rate by Support Calls



Shows the relationship between the number of support calls and customer churn rate.



!\[Churn by Support Calls](churn\_by\_support\_calls.png)



\### Confusion Matrix



Shows correct and incorrect model predictions on the test dataset.



!\[Confusion Matrix](confusion\_matrix.png)



\## Project Structure



```text

Customer-Churn-Prediction-System/

│

├── customer\_churn\_data.csv

├── high\_risk\_customers.csv

├── generate\_data.py

├── churn\_prediction.py

├── churn\_distribution.png

├── monthly\_charges\_vs\_churn.png

├── churn\_by\_contract\_length.png

├── churn\_by\_support\_calls.png

├── confusion\_matrix.png

├── requirements.txt

├── .gitignore

└── README.md

