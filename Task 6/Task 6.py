import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("train.csv")   # Kaggle House Prices dataset
print("First 5 rows:")
print(df.head())
print("\nData Info:")
print(df.info())
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
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\nModel Performance:")
print("Mean Absolute Error:", mae)
print("R² Score:", r2)
results_df = pd.DataFrame({
    "Metric": ["MAE", "R2"],
    "Value": [mae, r2]
})
results_df.to_csv("model_evaluation_results.csv", index=False)
plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red", linestyle="--")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Predicted vs Actual SalePrice")
plt.savefig("predicted_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()
cat_features = model.named_steps["preprocessor"].transformers_[1][1].get_feature_names_out(categorical_features)
all_features = np.concatenate([
    ["GrLivArea", "TotalBsmtSF"],   
    cat_features,                   
    ["OverallQual", "GarageCars", "FullBath"]  
])
coefficients = model.named_steps["regressor"].coef_

feature_importance = pd.DataFrame({
    "Feature": all_features,
    "Coefficient": coefficients
}).sort_values(by="Coefficient", key=abs, ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))
feature_importance.to_csv("feature_importance.csv", index=False)
plt.figure(figsize=(8, 6))
sns.barplot(x="Coefficient", y="Feature", data=feature_importance.head(15))
plt.title("Top 15 Feature Importance (Linear Regression Coefficients)")
plt.savefig("feature_importance.png", dpi=300, bbox_inches="tight")
plt.close()

