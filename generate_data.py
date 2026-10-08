import pandas as pd
import numpy as np

np.random.seed(42)

customers = []

for i in range(1, 5001):

    age = np.random.randint(18, 70)
    tenure = np.random.randint(1, 73)
    monthly_charges = np.random.randint(20, 150)
    support_calls = np.random.randint(0, 10)
    contract_length = np.random.choice([1, 12, 24])
    internet_service = np.random.choice(
        ["DSL", "Fiber", "No"]
    )

    customers.append({
        "Customer_ID": f"C{i:05d}",
        "Age": age,
        "Tenure_Months": tenure,
        "Monthly_Charges": monthly_charges,
        "Support_Calls": support_calls,
        "Contract_Length": contract_length,
        "Internet_Service": internet_service
    })

df = pd.DataFrame(customers)

# Create realistic churn cases
churn_probability = (
    0.05
    + (df["Support_Calls"] >= 6) * 0.20
    + (df["Tenure_Months"] <= 6) * 0.15
    + (df["Monthly_Charges"] >= 100) * 0.10
    + (df["Contract_Length"] == 1) * 0.15
)

churn_probability = churn_probability.clip(upper=0.80)

df["Churn"] = (
    np.random.random(len(df)) < churn_probability
).astype(int)

df.to_csv("customer_churn_data.csv", index=False)

print("Customer churn dataset generated successfully!")
print("Total customers:", len(df))
print("Churned customers:", df["Churn"].sum())
print("Dataset saved as customer_churn_data.csv")