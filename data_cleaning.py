"""
=============================================================================
Project: Data Cleaning & Visualization Project
File: data_cleaning.py
Author: Data Science Intern
Description:
    This script performs an end-to-end data cleaning, exploratory data analysis
    (EDA), and visualization pipeline on messy customer data using Pandas,
    Matplotlib, and Seaborn.
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual styling for seaborn and matplotlib
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'figure.titlesize': 14,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.autolayout': True
})

# Define paths relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "messy_customer_data.csv")
CLEANED_DATA_PATH = os.path.join(SCRIPT_DIR, "cleaned_customer_data.csv")
CHARTS_DIR = os.path.join(SCRIPT_DIR, "charts")

os.makedirs(CHARTS_DIR, exist_ok=True)

print("=" * 80)
print("              DATA CLEANING & VISUALIZATION PIPELINE")
print("=" * 80)

# ---------------------------------------------------------------------------
# STEP 1: Read the CSV file using Pandas
# ---------------------------------------------------------------------------
print("\n>>> STEP 1: Reading raw dataset from CSV...")
df_raw = pd.read_csv(RAW_DATA_PATH)
print(f"Successfully loaded '{RAW_DATA_PATH}'.")
print("\nFirst 5 rows of raw data:")
print(df_raw.head())

# ---------------------------------------------------------------------------
# STEP 2: Show the number of rows and columns
# ---------------------------------------------------------------------------
print("\n>>> STEP 2: Dataset Dimensions (Shape)")
rows, cols = df_raw.shape
print(f"Total Rows: {rows}")
print(f"Total Columns: {cols}")

# ---------------------------------------------------------------------------
# STEP 3: Show all column names
# ---------------------------------------------------------------------------
print("\n>>> STEP 3: Column Names")
for i, col in enumerate(df_raw.columns, start=1):
    print(f"  {i}. {col}")

# ---------------------------------------------------------------------------
# STEP 4: Show data types
# ---------------------------------------------------------------------------
print("\n>>> STEP 4: Initial Data Types")
print(df_raw.dtypes)

# ---------------------------------------------------------------------------
# STEP 5: Check missing values
# ---------------------------------------------------------------------------
print("\n>>> STEP 5: Missing Values (Null / NaN Counts)")
missing_counts = df_raw.isnull().sum()
missing_percent = (missing_counts / len(df_raw)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Percentage': missing_percent
})
print(missing_df[missing_df['Missing_Count'] > 0])

# ---------------------------------------------------------------------------
# STEP 6: Check duplicate rows
# ---------------------------------------------------------------------------
print("\n>>> STEP 6: Checking for Duplicate Rows")
duplicate_rows = df_raw[df_raw.duplicated(keep=False)]
num_duplicates = df_raw.duplicated().sum()
print(f"Number of duplicate rows found: {num_duplicates}")
if num_duplicates > 0:
    print("Duplicate records found in dataset:")
    print(duplicate_rows)

# ---------------------------------------------------------------------------
# STEP 7: Check numerical statistics
# ---------------------------------------------------------------------------
print("\n>>> STEP 7: Summary Statistics of Numerical Columns (Raw Data)")
print(df_raw.describe())

# ---------------------------------------------------------------------------
# STEP 8: Detect possible outliers using IQR (Interquartile Range) Method
# ---------------------------------------------------------------------------
print("\n>>> STEP 8: Outlier Detection using IQR Method")
numeric_cols = ['Age', 'Annual_Income', 'Satisfaction_Score']
iqr_summary = {}

for col in numeric_cols:
    series = df_raw[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = df_raw[(df_raw[col] < lower_bound) | (df_raw[col] > upper_bound)]
    iqr_summary[col] = {
        'Q1': q1, 'Q3': q3, 'IQR': iqr,
        'Lower_Bound': lower_bound, 'Upper_Bound': upper_bound,
        'Outlier_Count': len(outliers)
    }
    print(f"\n--- Outlier Analysis for '{col}' ---")
    print(f"  Q1 (25th percentile): {q1}")
    print(f"  Q3 (75th percentile): {q3}")
    print(f"  IQR (Interquartile Range): {iqr}")
    print(f"  Lower Bound (Q1 - 1.5*IQR): {lower_bound}")
    print(f"  Upper Bound (Q3 + 1.5*IQR): {upper_bound}")
    if len(outliers) > 0:
        print(f"  Outliers detected ({len(outliers)}):")
        print(outliers[['Customer_ID', 'Name', col]])
    else:
        print("  No statistical outliers detected.")

# ---------------------------------------------------------------------------
# DATA CLEANING PHASE
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("                       DATA CLEANING PHASE")
print("=" * 80)

# Start cleaning from a copy of raw data
df_clean = df_raw.copy()

# ---------------------------------------------------------------------------
# STEP 10: Remove duplicate records first
# Best practice: Removing duplicates before calculating statistics prevents bias
# ---------------------------------------------------------------------------
print("\n>>> STEP 10: Removing Duplicate Records...")
initial_row_count = len(df_clean)
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
print(f"Rows before deduplication: {initial_row_count}")
print(f"Rows after deduplication: {len(df_clean)}")
print(f"Removed {initial_row_count - len(df_clean)} duplicate record(s).")

# ---------------------------------------------------------------------------
# STEP 9: Handle missing values appropriately
# ---------------------------------------------------------------------------
print("\n>>> STEP 9: Handling Missing Values...")

# 1. Missing Age: Impute with median
# Age is a discrete demographic variable. Median is robust to skewness.
age_median = df_clean['Age'].median()
print(f"Imputing missing 'Age' values with median age: {age_median:.1f}")
df_clean['Age'] = df_clean['Age'].fillna(age_median)

# 2. Missing Annual_Income: Impute with median
# Income is heavily skewed by high earner outlier, so median is much safer than mean.
income_median = df_clean['Annual_Income'].median()
print(f"Imputing missing 'Annual_Income' values with median income: INR {income_median:,.2f}")
df_clean['Annual_Income'] = df_clean['Annual_Income'].fillna(income_median)

# 3. Missing Satisfaction_Score: Impute with median (or mode)
# Satisfaction is an ordinal score (1 to 5). Median is 4.0.
satisfaction_median = df_clean['Satisfaction_Score'].median()
print(f"Imputing missing 'Satisfaction_Score' with median score: {satisfaction_median:.1f}")
df_clean['Satisfaction_Score'] = df_clean['Satisfaction_Score'].fillna(satisfaction_median)

print("\nMissing values after imputation:")
print(df_clean.isnull().sum())

# ---------------------------------------------------------------------------
# STEP 11: Detect and handle outliers with a suitable method (IQR Capping / Winsorization)
# ---------------------------------------------------------------------------
print("\n>>> STEP 11: Handling Outliers...")
# Recalculate bounds on cleaned deduplicated data
q1_inc = df_clean['Annual_Income'].quantile(0.25)
q3_inc = df_clean['Annual_Income'].quantile(0.75)
iqr_inc = q3_inc - q1_inc
lower_inc = q1_inc - 1.5 * iqr_inc
upper_inc = q3_inc + 1.5 * iqr_inc

print(f"Annual Income IQR bounds: [INR {lower_inc:,.2f}, INR {upper_inc:,.2f}]")
outlier_incomes = df_clean[df_clean['Annual_Income'] > upper_inc]
print(f"Detected {len(outlier_incomes)} extreme income outlier(s):")
print(outlier_incomes[['Customer_ID', 'Name', 'Annual_Income']])

# Handling Method: Capping (Winsorization)
# Instead of deleting customer records, capping preserves the customer while
# limiting the distortion on downstream metrics and visualizations.
print("\nApplying IQR Capping (Winsorization) to 'Annual_Income'...")
df_clean['Annual_Income_Capped'] = df_clean['Annual_Income'].clip(lower=lower_inc, upper=upper_inc)
print(f"Customer Kiran's Annual Income was adjusted from INR {outlier_incomes['Annual_Income'].values[0]:,.2f} to upper bound INR {upper_inc:,.2f}.")
df_clean['Annual_Income'] = df_clean['Annual_Income_Capped']
df_clean.drop(columns=['Annual_Income_Capped'], inplace=True)

# Note on Age: Ramesh (Age 40) is slightly above statistical bound (39.5),
# but 40 is a valid real-world human age. We retain it as valid domain data.
print("Note on Age: Customer Ramesh (Age 40) is kept as a valid real-world customer age.")

# ---------------------------------------------------------------------------
# STEP 12: Make sure data types are correct
# ---------------------------------------------------------------------------
print("\n>>> STEP 12: Correcting Data Types...")
df_clean['Customer_ID'] = df_clean['Customer_ID'].astype(int)
df_clean['Name'] = df_clean['Name'].astype(str)
df_clean['Age'] = df_clean['Age'].astype(int)
df_clean['Gender'] = df_clean['Gender'].astype(str)
df_clean['City'] = df_clean['City'].astype(str)
df_clean['Annual_Income'] = df_clean['Annual_Income'].round(2)
df_clean['Satisfaction_Score'] = df_clean['Satisfaction_Score'].astype(int)
df_clean['Purchased'] = df_clean['Purchased'].astype(str)

print("Cleaned Data Types:")
print(df_clean.dtypes)

# ---------------------------------------------------------------------------
# STEP 13: Save the cleaned dataset as "cleaned_customer_data.csv"
# ---------------------------------------------------------------------------
print("\n>>> STEP 13: Saving Cleaned Dataset...")
df_clean.to_csv(CLEANED_DATA_PATH, index=False)
print(f"Cleaned dataset successfully saved to: '{CLEANED_DATA_PATH}'")
print("\nPreview of Cleaned Dataset:")
print(df_clean.to_string())

# ---------------------------------------------------------------------------
# DATA ANALYSIS & INSIGHTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("                       EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print("\n>>> STEP 14 & 15: In-depth Analysis & Pattern Finding")

print("\n1. Customer Age Analysis:")
print(f"   - Age Range: {df_clean['Age'].min()} to {df_clean['Age'].max()} years")
print(f"   - Average Age: {df_clean['Age'].mean():.1f} years | Median Age: {df_clean['Age'].median():.1f} years")

print("\n2. Customer Income Analysis:")
print(f"   - Income Range: INR {df_clean['Annual_Income'].min():,.2f} to INR {df_clean['Annual_Income'].max():,.2f}")
print(f"   - Average Income: INR {df_clean['Annual_Income'].mean():,.2f} | Median: INR {df_clean['Annual_Income'].median():,.2f}")

print("\n3. Customer Distribution by City:")
city_counts = df_clean['City'].value_counts()
for city, count in city_counts.items():
    print(f"   - {city}: {count} customers ({count/len(df_clean)*100:.1f}%)")

print("\n4. Purchase Status Breakdown:")
purchased_counts = df_clean['Purchased'].value_counts()
print(f"   - Purchased (Yes): {purchased_counts.get('Yes', 0)} ({purchased_counts.get('Yes', 0)/len(df_clean)*100:.1f}%)")
print(f"   - Not Purchased (No): {purchased_counts.get('No', 0)} ({purchased_counts.get('No', 0)/len(df_clean)*100:.1f}%)")

print("\n5. Customer Satisfaction Score Distribution:")
sat_counts = df_clean['Satisfaction_Score'].value_counts().sort_index()
for score, count in sat_counts.items():
    print(f"   - Score {score}/5: {count} customer(s)")

print("\n6. Cross-Analysis: Purchase Rate by City:")
city_purchase = pd.crosstab(df_clean['City'], df_clean['Purchased'], normalize='index') * 100
print(city_purchase.round(1))

print("\n7. Cross-Analysis: Satisfaction Score vs Purchase Decision:")
sat_purchase = pd.crosstab(df_clean['Satisfaction_Score'], df_clean['Purchased'])
print(sat_purchase)

print("\n8. Comparison: Metrics by Purchase Status (Means):")
metrics_by_purchase = df_clean.groupby('Purchased')[['Age', 'Annual_Income', 'Satisfaction_Score']].mean()
print(metrics_by_purchase.round(2))

# ---------------------------------------------------------------------------
# VISUALIZATION PHASE
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("                       CREATING VISUALIZATIONS")
print("=" * 80)

# --- Visualization 1: Missing Values in Raw Data ---
plt.figure(figsize=(8, 5))
missing_counts_raw = df_raw.isnull().sum()
missing_filtered = missing_counts_raw[missing_counts_raw > 0]
colors = ['#e74c3c', '#e67e22', '#f39c12']
bars = plt.bar(missing_filtered.index, missing_filtered.values, color=colors, edgecolor='black', width=0.5)
plt.title("Missing Values Count in Raw Dataset (Before Cleaning)", fontsize=13, pad=15, fontweight='bold')
plt.xlabel("Columns with Missing Data", labelpad=10)
plt.ylabel("Number of Missing Records", labelpad=10)
plt.ylim(0, 3)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{int(yval)} missing", ha='center', va='bottom', fontweight='bold')
chart1_path = os.path.join(CHARTS_DIR, "01_missing_values_before_cleaning.png")
plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart1_path}")

# --- Visualization 2: Customer Distribution by City ---
plt.figure(figsize=(8, 5))
city_order = df_clean['City'].value_counts().index
palette = sns.color_palette("Blues_r", len(city_order))
ax = sns.countplot(data=df_clean, x='City', hue='City', order=city_order, palette=palette, legend=False, edgecolor='black')
plt.title("Customer Distribution by City", fontsize=13, pad=15, fontweight='bold')
plt.xlabel("City", labelpad=10)
plt.ylabel("Number of Customers", labelpad=10)
plt.ylim(0, 7)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)} ({height/len(df_clean)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points', fontweight='bold')
chart2_path = os.path.join(CHARTS_DIR, "02_customer_distribution_by_city.png")
plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart2_path}")

# --- Visualization 3: Income Distribution (Histogram + KDE + Boxplot) ---
fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'height_ratios': [0.25, 0.75]})
sns.boxplot(x=df_clean['Annual_Income'], ax=ax_box, color='#3498db', fliersize=5)
ax_box.set(xlabel='')
ax_box.set_title("Annual Income Distribution (Post-Cleaning & Capping)", fontsize=13, pad=10, fontweight='bold')

sns.histplot(df_clean['Annual_Income'], kde=True, ax=ax_hist, color='#2980b9', bins=7, edgecolor='black')
ax_hist.set_xlabel("Annual Income (INR)", labelpad=10)
ax_hist.set_ylabel("Customer Count", labelpad=10)
median_inc = df_clean['Annual_Income'].median()
mean_inc = df_clean['Annual_Income'].mean()
ax_hist.axvline(median_inc, color='green', linestyle='--', linewidth=2, label=f'Median: INR {median_inc:,.0f}')
ax_hist.axvline(mean_inc, color='red', linestyle=':', linewidth=2, label=f'Mean: INR {mean_inc:,.0f}')
ax_hist.legend(loc='upper right')

chart3_path = os.path.join(CHARTS_DIR, "03_income_distribution.png")
plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart3_path}")

# --- Visualization 4: Satisfaction Score Distribution ---
plt.figure(figsize=(8, 5))
sat_palette = sns.color_palette("YlGnBu", 5)
ax = sns.countplot(data=df_clean, x='Satisfaction_Score', hue='Satisfaction_Score', palette=sat_palette, legend=False, edgecolor='black')
plt.title("Customer Satisfaction Score Distribution (1 = Lowest, 5 = Highest)", fontsize=13, pad=15, fontweight='bold')
plt.xlabel("Satisfaction Score (Rating)", labelpad=10)
plt.ylabel("Number of Customers", labelpad=10)
plt.ylim(0, 8)
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)} customers\n({height/len(df_clean)*100:.1f}%)',
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                xytext=(0, 4), textcoords='offset points', fontweight='bold')
chart4_path = os.path.join(CHARTS_DIR, "04_satisfaction_score_distribution.png")
plt.savefig(chart4_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart4_path}")

# --- Visualization 5: Purchased vs Not Purchased Comparison ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
# Donut Chart
purch_counts = df_clean['Purchased'].value_counts()
colors = ['#2ecc71', '#e74c3c']
ax1.pie(purch_counts, labels=purch_counts.index, autopct='%1.1f%%', startangle=90,
        colors=colors, wedgeprops=dict(width=0.4, edgecolor='white', linewidth=2),
        textprops={'fontweight': 'bold', 'fontsize': 11})
ax1.set_title("Purchase Conversion Rate", fontsize=13, fontweight='bold', pad=10)

# Bar breakdown
sns.countplot(data=df_clean, x='Purchased', hue='Purchased', palette=colors, legend=False, edgecolor='black', ax=ax2)
ax2.set_title("Customer Count by Purchase Status", fontsize=13, fontweight='bold', pad=10)
ax2.set_xlabel("Purchased", labelpad=10)
ax2.set_ylabel("Count", labelpad=10)
ax2.set_ylim(0, 14)
for p in ax2.patches:
    height = p.get_height()
    ax2.annotate(f'{int(height)} ({height/len(df_clean)*100:.1f}%)',
                 (p.get_x() + p.get_width() / 2., height),
                 ha='center', va='bottom',
                 xytext=(0, 4), textcoords='offset points', fontweight='bold')

chart5_path = os.path.join(CHARTS_DIR, "05_purchased_vs_not_purchased.png")
plt.savefig(chart5_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart5_path}")

# --- Visualization 6: Satisfaction Score vs Purchase Decision ---
plt.figure(figsize=(8, 5))
palette_pur = {'Yes': '#2ecc71', 'No': '#e74c3c'}
ax = sns.countplot(data=df_clean, x='Satisfaction_Score', hue='Purchased', palette=palette_pur, edgecolor='black')
plt.title("Purchase Decision by Customer Satisfaction Score", fontsize=13, pad=15, fontweight='bold')
plt.xlabel("Satisfaction Score", labelpad=10)
plt.ylabel("Number of Customers", labelpad=10)
plt.ylim(0, 7)
plt.legend(title='Purchased', loc='upper left')
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{int(height)}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', fontweight='bold')
chart6_path = os.path.join(CHARTS_DIR, "06_satisfaction_vs_purchase.png")
plt.savefig(chart6_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart6_path}")

# --- Visualization 7: Correlation Heatmap ---
plt.figure(figsize=(7, 5))
df_corr = df_clean[['Age', 'Annual_Income', 'Satisfaction_Score']].copy()
df_corr['Purchased_Numeric'] = df_clean['Purchased'].map({'Yes': 1, 'No': 0})
corr_matrix = df_corr.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1,
            linewidths=1, linecolor='white', cbar_kws={'label': 'Correlation Coefficient'})
plt.title("Correlation Matrix of Customer Attributes", fontsize=13, pad=15, fontweight='bold')
chart7_path = os.path.join(CHARTS_DIR, "07_correlation_matrix.png")
plt.savefig(chart7_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart7_path}")

# --- Visualization 8: Age vs Annual Income by Purchase Status ---
plt.figure(figsize=(9, 5.5))
sns.scatterplot(
    data=df_clean,
    x='Age',
    y='Annual_Income',
    hue='Purchased',
    style='Purchased',
    s=120,
    palette=palette_pur,
    edgecolor='black'
)
plt.title("Customer Segmentation: Age vs. Annual Income by Purchase Status", fontsize=13, pad=15, fontweight='bold')
plt.xlabel("Age (Years)", labelpad=10)
plt.ylabel("Annual Income (INR)", labelpad=10)
plt.legend(title="Purchased", loc='upper left')
chart8_path = os.path.join(CHARTS_DIR, "08_age_vs_income_by_purchase.png")
plt.savefig(chart8_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {chart8_path}")

print("\n" + "=" * 80)
print("             ALL PROCESSING & VISUALIZATIONS COMPLETE!")
print("=" * 80)
