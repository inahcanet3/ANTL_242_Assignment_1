```python
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

print("1) Which of the four industries had the highest")
print("   employment growth from 2014 to 2025?")
print()

print("2) Are employment levels from the previous year")
print("   good predictors of overall employment growth")
print("   in the following year?")
print()


# --------------------------------------------------
# Industries
# --------------------------------------------------

industries = [
    "Technology",
    "Healthcare",
    "Education",
    "Food Services"
]


# --------------------------------------------------
# Question 1
# Overall employment growth from 2014 to 2025
# --------------------------------------------------

print("QUESTION 1")
print()
print("OVERALL EMPLOYMENT GROWTH: 2014-2025")
print()

growth_results = {}

for industry in industries:

    starting_value = employment_df.loc[
        employment_df["year"] == 2014, industry
    ].iloc[0]

    ending_value = employment_df.loc[
        employment_df["year"] == 2025, industry
    ].iloc[0]

    growth = (
        (ending_value - starting_value)
        / starting_value
    ) * 100

    growth_results[industry] = growth

    print(f"{industry}: {growth:.2f}%")


highest_growth_industry = max(
    growth_results,
    key=growth_results.get
)

print()

print(
    f"Highest growth: {highest_growth_industry} "
    f"({growth_results[highest_growth_industry]:.2f}%)"
)

print()


# --------------------------------------------------
# Calculate year-to-year employment growth
# --------------------------------------------------

for industry in industries:

    employment_df[industry + "_growth"] = (
        employment_df[industry].pct_change() * 100
    )


# --------------------------------------------------
# Question 1 Modeling
# Previous-year employment predicting
# following-year growth for individual industries
# --------------------------------------------------

print("QUESTION 1 MODELING")
print()

q1_results = []

for industry in industries:

    employment_df["Previous_Employment"] = (
        employment_df[industry].shift(1)
    )

    model_df = employment_df.dropna().copy()

    # Training data: 2015-2020
    train = model_df[
        model_df["year"] <= 2020
    ]

    # Testing data: 2021-2025
    test = model_df[
        model_df["year"] >= 2021
    ]

    X_train = train[["Previous_Employment"]]

    y_train = train[
        industry + "_growth"
    ]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    X_test = test[["Previous_Employment"]]

    y_test = test[
        industry + "_growth"
    ]

    predictions = model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    q1_results.append({
        "Industry": industry,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse
    })


q1_results_df = pd.DataFrame(
    q1_results
)

print(
    q1_results_df.to_string(
        index=False
    )
)


# --------------------------------------------------
# Question 2
# Previous-year employment across all four industries
# predicting overall employment growth
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
train = model_df[
    model_df["year"] <= 2022
]

# Testing data: 2023-2025
test = model_df[
    model_df["year"] >= 2023
]


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

model.fit(
    X_train,
    y_train
)


X_test = test[predictors]

y_test = test["Total_Growth"]


predictions = model.predict(
    X_test
)


# Evaluation metrics
r2 = r2_score(
    y_test,
    predictions
)

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)


print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)


# --------------------------------------------------
# Question 2 Predictions
# --------------------------------------------------

print()
print("QUESTION 2 PREDICTIONS")
print()


results = test[
    ["year", "Total_Growth"]
].copy()

results["Predicted_Growth"] = predictions


print(
    results.to_string(
        index=False
    )
)


# --------------------------------------------------
# Final Answers
# --------------------------------------------------

print()
print("FINAL ANSWERS")
print()

print(
    f"1) {highest_growth_industry} had the highest "
    f"employment growth from 2014 to 2025, with "
    f"{growth_results[highest_growth_industry]:.2f}% growth."
)

print()

if r2 < 0:
    print(
        "2) Previous-year employment levels were poor "
        "predictors of overall employment growth."
    )
else:
    print(
        "2) Previous-year employment levels provided "
        "some predictive value for overall employment growth."
    )
