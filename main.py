import sqlite3
import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# Connect to the SQLite database
db_path = "employment.db"

conn = sqlite3.connect(db_path)

employment_df = pd.read_sql_query(
    "SELECT * FROM employment",
    conn
)

conn.close()


# --------------------------------------------------
# Foundational Questions
# --------------------------------------------------

print()
print("FOUNDATIONAL QUESTIONS")
print()

print("1) Are employment levels from the previous year")
print("   good predictors of employment growth in the")
print("   following year for individual industries?")
print()

print("2) Are employment levels across the four industries")
print("   from the previous year good predictors of overall")
print("   employment growth in the following year?")
print()


# --------------------------------------------------
# Question 1
# --------------------------------------------------

industries = [
    "Technology",
    "Healthcare",
    "Education",
    "Food Services"
]

for industry in industries:
    employment_df[industry + "_growth"] = (
        employment_df[industry].pct_change() * 100
    )


print("QUESTION 1 RESULTS")
print()

q1_results = []

for industry in industries:

    employment_df["Previous_Employment"] = (
        employment_df[industry].shift(1)
    )

    model_df = employment_df.dropna().copy()

    # Training data: 2015-2020
    train = model_df[model_df["year"] <= 2020]

    # Testing data: 2021-2025
    test = model_df[model_df["year"] >= 2021]

    X_train = train[["Previous_Employment"]]
    y_train = train[industry + "_growth"]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = test[["Previous_Employment"]]
    y_test = test[industry + "_growth"]

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    q1_results.append({
        "Industry": industry,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })


q1_results_df = pd.DataFrame(q1_results)

print(q1_results_df.to_string(index=False))


# --------------------------------------------------
# Question 2
# --------------------------------------------------

print()
print("QUESTION 2 RESULTS")
print()


# Total employment
employment_df["Total_Employment"] = (
    employment_df["Technology"]
    + employment_df["Healthcare"]
    + employment_df["Education"]
    + employment_df["Food Services"]
)


# Total employment growth
employment_df["Total_Growth"] = (
    employment_df["Total_Employment"].pct_change() * 100
)


# Previous year's employment for each industry
employment_df["Technology_Previous"] = (
    employment_df["Technology"].shift(1)
)

employment_df["Healthcare_Previous"] = (
    employment_df["Healthcare"].shift(1)
)

employment_df["Education_Previous"] = (
    employment_df["Education"].shift(1)
)

employment_df["Food_Previous"] = (
    employment_df["Food Services"].shift(1)
)


model_df = employment_df.dropna().copy()


# Training data: 2015-2022
train = model_df[model_df["year"] <= 2022]

# Testing data: 2023-2025
test = model_df[model_df["year"] >= 2023]


# Four predictors
predictors = [
    "Technology_Previous",
    "Healthcare_Previous",
    "Education_Previous",
    "Food_Previous"
]


X_train = train[predictors]
y_train = train["Total_Growth"]

model = LinearRegression()
model.fit(X_train, y_train)


X_test = test[predictors]
y_test = test["Total_Growth"]

predictions = model.predict(X_test)


# Evaluation metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))


print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)


# Q2 predictions
print()
print("QUESTION 2 PREDICTIONS")
print()

results = test[
    ["year", "Total_Growth"]
].copy()

results["Predicted_Growth"] = predictions

print(results.to_string(index=False))
