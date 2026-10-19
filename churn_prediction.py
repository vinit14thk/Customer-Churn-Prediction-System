import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("customer_churn_data.csv")

print("CUSTOMER CHURN PREDICTION SYSTEM")
print("=" * 50)
print("Machine Learning-Based Customer Retention Analysis")
print("=" * 50)

# Features and target
features = [
    "Age",
    "Tenure_Months",
    "Monthly_Charges",
    "Support_Calls",
    "Contract_Length",
    "Internet_Service"
]

X = df[features]
y = df["Churn"]

print("\nFeatures Used:")
for feature in features:
    print("-", feature)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nDataset Split:")
print("Training customers:", len(X_train))
print("Testing customers:", len(X_test))

# Categorical features
categorical_features = ["Internet_Service"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ],
    remainder="passthrough"
)

# Random Forest model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ]
)

print("\nRandom Forest Configuration:")
print("Number of trees:", 200)

# Train model
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Predictions
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nModel Performance")
print("=" * 40)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

# Predict churn probability for all customers

churn_probabilities = model.predict_proba(X)[:, 1]

df["Churn_Probability"] = (
    churn_probabilities * 100
).round(2)

df["Risk_Level"] = pd.cut(
    df["Churn_Probability"],
    bins=[-1, 30, 60, 100],
    labels=["Low", "Medium", "High"]
)

# Save predictions for all customers
df.to_csv("customer_churn_predictions.csv", index=False)

print("All customer predictions saved to:")
print("customer_churn_predictions.csv")

# Save high-risk customers

high_risk_customers = df[
    df["Risk_Level"] == "High"
].sort_values(
    "Churn_Probability",
    ascending=False
)

high_risk_customers.to_csv(
    "high_risk_customers.csv",
    index=False
)

print("\nCustomer Churn Risk Analysis")
print("=" * 40)
print("High-risk customers:", len(high_risk_customers))

print("\nTop 10 High-Risk Customers:")

print(
    high_risk_customers[
        [
            "Customer_ID",
            "Monthly_Charges",
            "Support_Calls",
            "Contract_Length",
            "Churn_Probability",
            "Risk_Level"
        ]
    ].head(10)
)

print("\nHigh-risk customer report saved successfully!")
print("Output file: high_risk_customers.csv")

import matplotlib.pyplot as plt

# Churn distribution

plt.figure(figsize=(7, 5))

df["Churn"].value_counts().sort_index().plot(
    kind="bar",
    xlabel="Customer Status",
    ylabel="Number of Customers",
    title="Churned vs Non-Churned Customers"
)

plt.xticks(
    [0, 1],
    ["Stayed", "Churned"],
    rotation=0
)

plt.tight_layout()

plt.savefig("churn_distribution.png")

plt.show()

# Monthly charges comparison

plt.figure(figsize=(8, 5))

df.boxplot(
    column="Monthly_Charges",
    by="Churn"
)

plt.xlabel("Customer Status")
plt.ylabel("Monthly Charges")
plt.title("Monthly Charges by Customer Churn Status")
plt.suptitle("")

plt.xticks(
    [1, 2],
    ["Stayed", "Churned"]
)

plt.tight_layout()

plt.savefig("monthly_charges_vs_churn.png")

plt.show()

# Churn by contract length

churn_by_contract = (
    df.groupby("Contract_Length")["Churn"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(8, 5))

churn_by_contract.plot(
    kind="bar",
    xlabel="Contract Length (Months)",
    ylabel="Churn Rate (%)",
    title="Churn Rate by Contract Length"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig("churn_by_contract_length.png")

plt.show()

# Churn rate by support calls

churn_by_support = (
    df.groupby("Support_Calls")["Churn"]
    .mean()
    .mul(100)
)

plt.figure(figsize=(9, 5))

churn_by_support.plot(
    kind="line",
    marker="o"
)

plt.xlabel("Number of Support Calls")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Support Calls")

plt.grid(True)

plt.tight_layout()

plt.savefig("churn_by_support_calls.png")

plt.show()

import seaborn as sns

# Confusion matrix visualization

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Churned"],
    yticklabels=["Stayed", "Churned"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Customer Churn Confusion Matrix")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()

# ==============================
# IMPROVED MODEL EVALUATION
# ==============================

from sklearn.ensemble import RandomForestClassifier

improved_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                max_depth=12,
                min_samples_split=5,
                random_state=42
            )
        )
    ]
)

# Train improved model
improved_model.fit(X_train, y_train)

# Predictions
improved_predictions = improved_model.predict(X_test)

# Evaluation
improved_accuracy = accuracy_score(
    y_test,
    improved_predictions
)

improved_precision = precision_score(
    y_test,
    improved_predictions
)

improved_recall = recall_score(
    y_test,
    improved_predictions
)

improved_f1 = f1_score(
    y_test,
    improved_predictions
)

print("\nImproved Model Performance")
print("=" * 40)

print(
    f"Accuracy:  {improved_accuracy * 100:.2f}%"
)

print(
    f"Precision: {improved_precision * 100:.2f}%"
)

print(
    f"Recall:    {improved_recall * 100:.2f}%"
)

print(
    f"F1 Score:  {improved_f1 * 100:.2f}%"
)

# Compare both models
print("\nMODEL COMPARISON")
print("=" * 45)

print(f"{'Metric':<15}{'Original':>12}{'Improved':>12}")
print("-" * 39)

metrics = [
    ("Accuracy", accuracy, improved_accuracy),
    ("Precision", precision, improved_precision),
    ("Recall", recall, improved_recall),
    ("F1 Score", f1, improved_f1),
]

for name, original, improved in metrics:
    print(
        f"{name:<15}{original * 100:>11.2f}%"
        f"{improved * 100:>11.2f}%"
    )

# Save comparison results
comparison = pd.DataFrame(
    metrics,
    columns=["Metric", "Original_Model", "Improved_Model"]
)

comparison.to_csv("model_comparison.csv", index=False)

print("\nComparison saved to model_comparison.csv")