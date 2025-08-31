import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
df = pd.read_csv("train.csv")
print("First 5 rows:\n", df.head())
print("\nData Types:\n", df.dtypes)
for col in df.columns:
    if df[col].dtype in ["int64", "float64"]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])
def mean(values):
    return np.mean(values)
def median(values):
    return np.median(values)
def mode(values):
    return stats.mode(values, keepdims=True)[0][0]
def variance(values):
    return np.var(values)
def std_dev(values):
    return np.std(values)
# Example column: "SalePrice"
col = df["SalePrice"]
print("\nSalePrice Stats:")
print("Mean:", mean(col))
print("Median:", median(col))
print("Mode:", mode(col))
print("Variance:", variance(col))
print("Standard Deviation:", std_dev(col))

plt.hist(df["SalePrice"], bins=30, edgecolor="black")
plt.title("Histogram of SalePrice")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.savefig("histogram.png")
plt.close()
df["Neighborhood"].value_counts().head(10).plot(kind="bar")
plt.title("Top 10 Neighborhoods")
plt.xlabel("Neighborhood")
plt.ylabel("Count")

plt.savefig("bar_chart.png")
plt.close()


# Markdown Notes (for Notebook conversion)
notes = """
### Insights from House Prices Dataset

a) What did I learn?  
SalePrice distribution is right-skewed.  
Some neighborhoods dominate the dataset.  

b) How did visualization help?
Histogram showed skewness of prices.  
Bar chart showed distribution across neighborhoods.  

c) Cleaning issues faced:
Missing numeric values → replaced with mean.  
Missing categorical values → replaced with mode.  
Some columns needed conversion to numeric.  
"""
print(notes)
