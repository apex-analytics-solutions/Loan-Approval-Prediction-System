# ============================================================
# Loan Eligibility Prediction System
# Advanced Exploratory Data Analysis (EDA) - Part 1
# Author: Anees Ur Rehman
# ============================================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Create Output Folder
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# Plot Style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10,6)

print("="*60)
print(" LOAN ELIGIBILITY PREDICTION SYSTEM ")
print(" Advanced Exploratory Data Analysis ")
print("="*60)

# -----------------------------
# Load Dataset
# -----------------------------
print("\nLoading Dataset...\n")

df = pd.read_csv("data/loan_dataset_2025.csv")

print("Dataset Loaded Successfully!\n")

# -----------------------------
# Basic Information
# -----------------------------
print("="*60)
print("DATASET SHAPE")
print("="*60)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

print("\n")

# -----------------------------
# First Five Rows
# -----------------------------
print("="*60)
print("FIRST FIVE ROWS")
print("="*60)

print(df.head())

print("\n")

# -----------------------------
# Dataset Information
# -----------------------------
print("="*60)
print("DATASET INFO")
print("="*60)

print(df.info())

print("\n")

# -----------------------------
# Data Types
# -----------------------------
print("="*60)
print("DATA TYPES")
print("="*60)

print(df.dtypes)

print("\n")

# -----------------------------
# Missing Values
# -----------------------------
print("="*60)
print("MISSING VALUES")
print("="*60)

missing = df.isnull().sum()

print(missing)

missing_df = pd.DataFrame({
    "Column": missing.index,
    "Missing Values": missing.values
})

missing_df.to_csv("outputs/missing_values_report.csv", index=False)

print("\nMissing Value Report Saved")

# -----------------------------
# Duplicate Records
# -----------------------------
print("="*60)
print("DUPLICATE RECORDS")
print("="*60)

duplicates = df.duplicated().sum()

print(f"Duplicate Rows : {duplicates}")

# -----------------------------
# Statistical Summary
# -----------------------------
print("="*60)
print("STATISTICAL SUMMARY")
print("="*60)

print(df.describe())

print("\n")

# -----------------------------
# Target Variable
# -----------------------------
TARGET = "loan_paid_back"

print("="*60)
print("TARGET VARIABLE ANALYSIS")
print("="*60)

print(df[TARGET].value_counts())

print("\n")

print(df[TARGET].value_counts(normalize=True) * 100)

# ============================================================
# BAR CHART
# ============================================================

plt.figure(figsize=(7,5))

ax = sns.countplot(
    x=TARGET,
    data=df,
    palette="viridis"
)

plt.title(
    "Loan Approval Distribution",
    fontsize=16,
    weight="bold"
)

plt.xlabel("Loan Status")
plt.ylabel("Number of Applicants")

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()

plt.savefig(
    "outputs/01_target_distribution_bar.png",
    dpi=300
)

plt.close()

# ============================================================
# PIE CHART
# ============================================================

labels = ["Rejected","Approved"]

sizes = df[TARGET].value_counts().sort_index()

colors = ["#ff6b6b","#4ecdc4"]

plt.figure(figsize=(6,6))

plt.pie(
    sizes,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90,
    colors=colors,
    explode=(0.03,0.03),
    shadow=True
)

plt.title(
    "Loan Approval Percentage",
    fontsize=15,
    weight="bold"
)

plt.savefig(
    "outputs/02_target_distribution_pie.png",
    dpi=300
)

plt.close()

print("\nTarget Distribution Graph Saved")

# ============================================================
# BUSINESS INSIGHT
# ============================================================

approved = df[TARGET].sum()
rejected = len(df) - approved

print("\n")
print("="*60)
print("BUSINESS INSIGHT")
print("="*60)

print(f"Approved Applications : {approved}")
print(f"Rejected Applications : {rejected}")

approval_rate = (approved/len(df))*100

print(f"Approval Rate : {approval_rate:.2f}%")

if approval_rate > 60:
    print("\nInsight:")
    print("The dataset contains a higher number of approved loans.")
    print("Machine Learning model may learn approval patterns effectively.")
else:
    print("\nInsight:")
    print("Dataset is relatively balanced.")
    print("Model training should avoid bias.")

print("\n")
print("="*60)
print("EDA PART-1 COMPLETED SUCCESSFULLY")
print("="*60)