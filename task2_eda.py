# ==========================================================
# Task 2 : Exploratory Data Analysis (EDA)
# AI & ML Internship - Elevate Labs
# ==========================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*60)
print("TASK 2 : EXPLORATORY DATA ANALYSIS")
print("="*60)

# ----------------------------------------------------------
# Create Required Directories
# ----------------------------------------------------------

os.makedirs("Screenshots", exist_ok=True)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

csv_path = "dataset/Titanic-Dataset.csv"

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        "Titanic-Dataset.csv not found inside dataset folder."
    )

df = pd.read_csv(csv_path)

# ----------------------------------------------------------
# Dataset Overview
# ----------------------------------------------------------

print("\n[DATASET OVERVIEW]")
print("-"*50)

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nFirst 5 Rows:")
print(df.head())

# ----------------------------------------------------------
# Missing Values Analysis
# ----------------------------------------------------------

print("\n[MISSING VALUES ANALYSIS]")
print("-"*50)

missing_values = df.isnull().sum()

print(missing_values)

# ----------------------------------------------------------
# Summary Statistics
# ----------------------------------------------------------

print("\n[SUMMARY STATISTICS]")
print("-"*90)

summary_stats = df.describe()

print(summary_stats)

summary_stats.to_csv(
    "summary_statistics.csv"
)

print(
    "\n✅Summary Statistics Saved Successfully."
)

# ----------------------------------------------------------
# Skewness Analysis
# ----------------------------------------------------------

print("\n[SKEWNESS ANALYSIS]")
print("-"*50)

print(
    df[["Age", "Fare"]].skew()
)

# ----------------------------------------------------------
# Histogram : Age Distribution
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df["Age"].dropna(),
    bins=30,
    kde=True
)

plt.title(
    "Age Distribution",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel("Age")
plt.ylabel("Frequency")

plt.savefig(
    "Screenshots/histogram_age.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Histogram : Fare Distribution
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    df["Fare"],
    bins=30,
    kde=True
)

plt.title(
    "Fare Distribution",
    fontsize=12,
    fontweight="bold"
)

plt.xlabel("Fare")
plt.ylabel("Frequency")

plt.savefig(
    "Screenshots/histogram_fare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Boxplot : Age
# ----------------------------------------------------------

plt.figure(figsize=(8,4))

sns.boxplot(
    x=df["Age"]
)

plt.title(
    "Boxplot of Age",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Screenshots/boxplot_age.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Boxplot : Fare
# ----------------------------------------------------------

plt.figure(figsize=(8,4))

sns.boxplot(
    x=df["Fare"]
)

plt.title(
    "Boxplot of Fare",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Screenshots/boxplot_fare.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Survival Analysis
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x="Survived",
    data=df
)

plt.title(
    "Passenger Survival Distribution",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Screenshots/survival_analysis.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Survival by Gender
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x="Sex",
    hue="Survived",
    data=df
)

plt.title(
    "Gender-wise Survival Analysis",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Screenshots/survival_by_gender.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Survival by Passenger Class
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

sns.countplot(
    x="Pclass",
    hue="Survived",
    data=df
)

plt.title(
    "Passenger Class vs Survival",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Screenshots/survival_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Fare vs Survival
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

sns.boxplot(
    x="Survived",
    y="Fare",
    data=df
)

plt.title(
    "Fare Distribution by Survival Status",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Screenshots/fare_vs_survival.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Age vs Survival
# ----------------------------------------------------------

plt.figure(figsize=(8,5))

sns.histplot(
    data=df,
    x="Age",
    hue="Survived",
    kde=True
)

plt.title(
    "Age Distribution by Survival",
    fontsize=12,
    fontweight="bold"
)

plt.savefig(
    "Screenshots/age_survival.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Correlation Heatmap
# ----------------------------------------------------------

numeric_df = df.select_dtypes(
    include=np.number
)

plt.figure(figsize=(10,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap",
    fontsize=14,
    fontweight="bold"
)

plt.savefig(
    "Screenshots/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ----------------------------------------------------------
# Pairplot Analysis
# ----------------------------------------------------------

pairplot_df = df[
    [
        "Survived",
        "Pclass",
        "Age",
        "Fare"
    ]
].dropna()

pair_plot = sns.pairplot(
    pairplot_df,
    hue="Survived"
)

pair_plot.savefig(
    "Screenshots/pairplot.png"
)

plt.close()

# ----------------------------------------------------------
# Final EDA Insights
# ----------------------------------------------------------

print("\n" + "="*60)
print("EDA INSIGHTS")
print("="*60)

print("""
1. Majority of passengers were between 20 and 40 years old.

2. Fare distribution contains several extreme outliers.

3. Female passengers had a significantly higher survival rate.

4. First-class passengers survived more often.

5. Higher fare passengers generally had better survival chances.

6. Fare is highly right-skewed.

7. Age shows a moderate relationship with survival.

8. Missing values mainly exist in Cabin and Age columns.

9. Passenger Class and Fare have strong relationships.

10. Gender, Fare, and Passenger Class are important predictors.
""")

print("\n✅EDA Completed Successfully!")
print("All charts saved in Screenshots folder.")
print("Summary statistics saved as summary_statistics.csv")