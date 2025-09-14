import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv("train.csv")
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
numeric_features = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath"]
categorical_features = ["Neighborhood", "HouseStyle"]
target = "SalePrice"

X = df[numeric_features + categorical_features]
y = df[target]
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, ["GrLivArea", "TotalBsmtSF"]),
        ("cat", categorical_transformer, categorical_features)
    ],
    remainder="passthrough"
)

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Pipeline": pipeline,
        "Predictions": y_pred
    })
results_df = pd.DataFrame(results).drop(columns=["Pipeline", "Predictions"])
print(results_df)
results_df.to_csv("model_comparison_results.csv", index=False)
metrics = ["MAE", "RMSE", "R2"]
results_melted = results_df.melt(id_vars="Model", value_vars=metrics, var_name="Metric", value_name="Value")

plt.figure(figsize=(8, 6))
sns.barplot(x="Metric", y="Value", hue="Model", data=results_melted)
plt.title("Model Comparison Across Metrics")
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
best_model = max(results, key=lambda x: x["R2"])
print(f"\nBest Model: {best_model['Model']} (R² = {best_model['R2']:.4f})")

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=best_model["Predictions"], alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title(f"Predicted vs Actual SalePrice ({best_model['Model']})")
plt.savefig("best_model_scatter.png", dpi=300, bbox_inches="tight")

