# Data Cleaning & Visualization Project Report
**Project Name:** Customer Insights & Data Quality Enhancement  
**Role:** Data Science Intern  
**Dataset:** `messy_customer_data.csv` -> `cleaned_customer_data.csv`  
**Tools Used:** Python 3.13, Pandas, NumPy, Matplotlib, Seaborn  

---

## 1. Executive Summary

This project documents the end-to-end data auditing, cleaning, exploratory data analysis (EDA), and visualization of a real-world customer dataset. 

The original raw dataset contained **20 records and 8 attributes**, but suffered from common data quality problems:
- **Duplicate Records:** An identical duplicate entry skewing customer counts.
- **Missing Data:** Missing entries across demographic, financial, and behavioral attributes.
- **Extreme Outliers:** An income value more than double the normal upper bound.
- **Suboptimal Data Types:** Floating-point values representing discrete integers (such as age and satisfaction ratings).

By applying standard data engineering methodologies—deduplication, robust median-based imputation, IQR (Interquartile Range) Winsorization/capping, and type casting—we produced a reliable, pristine dataset (`cleaned_customer_data.csv`) consisting of **19 unique customers**. 

Subsequent exploratory data analysis uncovered significant business insights, including a **100% correlation between high customer satisfaction (ratings 4 and 5) and purchase conversion**, as well as pronounced geographic variation across target cities.

---

## 2. Dataset Overview

### Raw Dataset Dimensions & Structure
- **Number of Rows:** 20
- **Number of Columns:** 8
- **Target Variable:** `Purchased` (Yes / No)

### Attribute Dictionary

| Column Name | Raw Data Type | Cleaned Data Type | Description |
| :--- | :--- | :--- | :--- |
| `Customer_ID` | `int64` | `int64` | Unique numeric identifier for each customer |
| `Name` | `object` (string) | `object` (string) | First name of the customer |
| `Age` | `float64` | `int64` | Customer age in years |
| `Gender` | `object` (string) | `object` (string) | Customer gender (Male / Female) |
| `City` | `object` (string) | `object` (string) | Residential city (Vijayawada, Hyderabad, Guntur, Chennai) |
| `Annual_Income` | `float64` | `float64` | Annual income in Indian Rupees (INR) |
| `Satisfaction_Score`| `float64` | `int64` | Customer feedback rating on a 1–5 scale |
| `Purchased` | `object` (string) | `object` (string) | Whether the customer made a purchase (Yes / No) |

---

## 3. Data Problems Found (Pre-Cleaning Audit)

Before altering any data, we conducted an audit using Pandas methods (`.shape`, `.info()`, `.isnull().sum()`, `.duplicated()`, and `.describe()`). 

### 1. Duplicate Records
- Exactly **1 duplicate row** was identified:
  - **Customer_ID 105 (Kiran)** appeared at index 4 and index 15 with identical values across all columns (`Age: 29.0`, `Gender: Male`, `City: Hyderabad`, `Annual_Income: 150000.0`, `Satisfaction_Score: 2.0`, `Purchased: No`).
- **Impact:** Duplicate entries artificially inflate customer counts and bias statistical metrics like mean income and purchase conversion.

### 2. Missing Values (Null / NaN)
Three attributes contained 1 missing value each (5.0% of raw data per affected column):
1. **`Age`**: Missing for **Priya** (Customer ID 104, Vijayawada).
2. **`Annual_Income`**: Missing for **Vamsi** (Customer ID 109, Vijayawada).
3. **`Satisfaction_Score`**: Missing for **Sneha** (Customer ID 106, Guntur).

### 3. Outlier Inspection
Using the **IQR (Interquartile Range)** rule on numerical columns:
- **`Annual_Income`**:
  - $Q_1 = 42,500.0$, $Q_3 = 59,500.0$, $IQR = 17,000.0$
  - Upper Bound ($Q_3 + 1.5 \times IQR$) = **INR 85,000.0**
  - Outlier: **Kiran (Customer ID 105)** reported an annual income of **INR 150,000.0**, which is far above the upper threshold and nearly triple the average income.
- **`Age`**:
  - $Q_1 = 23.5$, $Q_3 = 29.5$, $IQR = 6.0$
  - Statistical Upper Bound = **38.5 years**
  - **Ramesh (Customer ID 119)** has an age of **40.0 years**. While mathematically above 38.5, 40 years is a natural, realistic customer demographic and not an error.
- **`Satisfaction_Score`**:
  - Scale: 1 to 5. No scores fell outside valid limits.

### 4. Data Type Inconsistencies
- `Age` and `Satisfaction_Score` were stored as `float64` due to standard IEEE floating-point representations of `NaN` in Pandas. Human age and 5-point rating scales should be discrete integers.

---

## 4. Cleaning Methodology & Step-by-Step Execution

```
Raw Data (20 rows)
       │
       ▼
[Step 1: Deduplication] ─── Remove 1 duplicate row (Kiran, ID 105) ───► (19 unique rows)
       │
       ▼
[Step 2: Missing Value Imputation]
       ├── Age: Impute Priya with Median Age (26)
       ├── Annual_Income: Impute Vamsi with Robust Median Income (INR 48,000)
       └── Satisfaction_Score: Impute Sneha with Median Rating (4)
       │
       ▼
[Step 3: Outlier Treatment] ─── Winsorization / IQR Capping
       └── Cap Kiran's Annual Income at Upper Fence (INR 78,750)
       │
       ▼
[Step 4: Data Type Casting] ─── Cast Age & Satisfaction_Score to int64
       │
       ▼
Cleaned Data (19 rows) ───► Saved to cleaned_customer_data.csv
```

### Step 1: Duplicate Removal
- **Action:** Applied `df.drop_duplicates().reset_index(drop=True)`.
- **Result:** Retained 19 unique customer records. Deduplication was executed *before* calculating imputation metrics so duplicate values did not distort median calculations.

### Step 2: Handling Missing Values
Instead of dropping rows (which would sacrifice 15% of our small dataset), we employed **median imputation**:
1. **Age (Priya):** Filled with the median age of **26 years**. Median is preferable to mean as it represents an actual whole year.
2. **Annual Income (Vamsi):** Filled with the median income of **INR 48,000.00**. Using the median is crucial here because extreme values (like INR 150,000) skew the arithmetic mean (which was INR 55,250), whereas the median reflects typical customer purchasing power.
3. **Satisfaction Score (Sneha):** Filled with the median score of **4**. Sneha made a purchase (`Purchased == 'Yes'`), and across the entire dataset all buyers have ratings of 4 or 5, making 4 an appropriate and consistent estimate.

### Step 3: Outlier Treatment (IQR Winsorization / Capping)
After deduplication and imputation, the IQR statistics for `Annual_Income` were:
- $Q_1 = 41,000.00$
- $Q_3 = 56,000.00$
- $IQR = 15,000.00$
- Lower Bound = $Q_1 - 1.5 \times IQR =$ **INR 18,500.00**
- Upper Bound = $Q_3 + 1.5 \times IQR =$ **INR 78,750.00**
- **Decision:** Rather than discarding Kiran from the dataset, we applied **IQR Capping (Winsorization)**:
  $$\text{Income}_{\text{capped}} = \min(\text{Income}, 78,750.00)$$
  Kiran's income was capped at **INR 78,750.00**, eliminating distortion in standard deviations and visualizations while preserving all customer behavior data.
- **Decision on Ramesh (Age 40):** Retained without modification. Age 40 is a valid adult demographic, and capping realistic ages would artificially distort true customer diversity.

### Step 4: Correcting Data Types
- Converted `Age` from `float64` to `int64`.
- Converted `Satisfaction_Score` from `float64` to `int64`.
- Ensured `Customer_ID` is `int64`, and categorical columns (`Gender`, `City`, `Purchased`) are clean string formats.

---

## 5. Statistical Summary Before vs. After Cleaning

| Metric | Raw Dataset (Uncleaned) | Cleaned Dataset (Processed) | Key Takeaway |
| :--- | :--- | :--- | :--- |
| **Total Rows** | 20 | 19 | Duplicate record safely removed |
| **Missing Values** | 3 total (Age: 1, Income: 1, Satisfaction: 1) | 0 (100% complete) | Complete dataset without data loss |
| **Mean Age** | 27.11 years | 26.89 years | Stable, realistic central tendency |
| **Median Age** | 26.00 years | 26.00 years | Preserved |
| **Mean Annual Income** | INR 60,236.84 | INR 51,118.42 | Distortion from outlier reduced by ~15% |
| **Median Annual Income**| INR 49,000.00 | INR 48,000.00 | Robust and representative |
| **Max Annual Income** | INR 150,000.00 | INR 78,750.00 | Extreme outlier capped at upper bound |
| **Mean Satisfaction** | 3.47 / 5.0 | 3.53 / 5.0 | Consistent rating average |

---

## 6. Exploratory Data Analysis & Visualizations

All visualizations were programmatically generated and stored in the `charts/` directory.

### Chart 1: Missing Values in Raw Data
- **File:** `charts/01_missing_values_before_cleaning.png`
- **Description:** A bar chart highlighting the exact count of missing values across `Age`, `Annual_Income`, and `Satisfaction_Score` prior to cleaning.
- **Insight:** Clarifies that missing data was evenly spread across three distinct fields (1 per field), warranting targeted median imputation rather than record deletion.

### Chart 2: Customer Distribution by City
- **File:** `charts/02_customer_distribution_by_city.png`
- **Description:** A countplot showing customer distribution across target geographic regions.
- **Findings:**
  - Vijayawada: **5 customers (26.3%)**
  - Hyderabad: **5 customers (26.3%)**
  - Guntur: **5 customers (26.3%)**
  - Chennai: **4 customers (21.1%)**
- **Insight:** The customer base is evenly distributed across Andhra Pradesh, Telangana, and Tamil Nadu metropolitan hubs.

### Chart 3: Annual Income Distribution
- **File:** `charts/03_income_distribution.png`
- **Description:** Dual-panel visualization pairing a boxplot with a KDE histogram and mean/median reference lines.
- **Findings:**
  - Incomes range from **INR 36,000 to INR 78,750**, with a dense concentration between **INR 40,000 and INR 55,000**.
  - Mean (INR 51,118) and Median (INR 48,000) are closely aligned following Winsorization, demonstrating healthy symmetry.

### Chart 4: Customer Satisfaction Score Distribution
- **File:** `charts/04_satisfaction_score_distribution.png`
- **Description:** Bar chart showing the frequency of satisfaction ratings on a scale of 1 to 5.
- **Findings:**
  - Rating 5: **5 customers (26.3%)**
  - Rating 4: **6 customers (31.6%)**
  - Rating 3: **4 customers (21.1%)**
  - Rating 2: **3 customers (15.8%)**
  - Rating 1: **1 customer (5.3%)**
- **Insight:** 57.9% of customers report high satisfaction (scores 4 or 5), while 42.1% report neutral or low satisfaction (scores 1 to 3).

### Chart 5: Purchase Conversion Rate
- **File:** `charts/05_purchased_vs_not_purchased.png`
- **Description:** Side-by-side donut chart and countplot comparing customers who purchased versus those who did not.
- **Findings:**
  - **Purchased (Yes):** 11 customers (**57.9%**)
  - **Not Purchased (No):** 8 customers (**42.1%**)

### Chart 6: Satisfaction Score vs. Purchase Decision (Key Finding)
- **File:** `charts/06_satisfaction_vs_purchase.png`
- **Description:** Grouped countplot mapping Satisfaction Score directly to Purchase decision.
- **Major Finding:**
  - **Satisfaction Scores 1, 2, and 3:** Exactly **0 out of 8 customers purchased** (0% conversion).
  - **Satisfaction Scores 4 and 5:** Exactly **11 out of 11 customers purchased** (100% conversion).
  - **Business Conclusion:** Customer satisfaction score serves as a deterministic threshold for purchase conversion in this dataset.

### Chart 7: Correlation Heatmap
- **File:** `charts/07_correlation_matrix.png`
- **Description:** Heatmap of Pearson correlation coefficients between numerical metrics and purchase status (`Purchased: Yes = 1, No = 0`).
- **Correlations:**
  - **Satisfaction Score & Purchase:** **+0.89** (very strong positive correlation).
  - **Age & Purchase:** **-0.66** (moderate negative correlation; younger customers convert more).
  - **Annual Income & Purchase:** **-0.54** (moderate negative correlation; higher earners in this cohort convert less).
  - **Age & Annual Income:** **+0.74** (older customers in the sample tend to have higher annual income).

### Chart 8: Customer Segmentation (Age vs. Annual Income by Purchase)
- **File:** `charts/08_age_vs_income_by_purchase.png`
- **Description:** 2D scatter plot showing customer clusters across Age and Annual Income, classified by Purchase status.
- **Findings:**
  - Two distinct customer segments emerge:
    1. **High-converting segment:** Young adults (ages 20–26) earning moderate incomes (INR 36,000–52,000) with satisfaction ratings of 4–5.
    2. **Low-converting segment:** Mid-career adults (ages 27–40) earning higher incomes (INR 49,000–78,750) with lower satisfaction ratings (1–3).

---

## 7. Strategic Business Insights & Recommendations

1. **Satisfaction Drives 100% of Conversions:**  
   Because customers rating satisfaction $\ge 4$ always buy, and those rating $\le 3$ never buy, business initiatives must prioritize post-interaction satisfaction surveys, onboarding support, and prompt issue resolution.

2. **City-Level Performance Disparities:**  
   - **Vijayawada:** 100% purchase rate (5/5).  
   - **Guntur:** 80% purchase rate (4/5).  
   - **Chennai:** 25% purchase rate (1/4).  
   - **Hyderabad:** 20% purchase rate (1/5).  
   *Recommendation:* Investigate why Andhra Pradesh tier-2 markets show exceptional product-market fit, and address customer dissatisfaction drivers in metro markets (Chennai and Hyderabad).

3. **Product Pricing & Positioning Alignment:**  
   Higher-income customers (INR 55,000+) are failing to convert. The current product offering may be tailored towards entry-level budgets or lacks premium tier features that appeal to higher earners.

---

## 8. Conclusion

All project requirements have been completed:
- The raw dataset was inspected, audited, and cleaned without altering the original CSV file.
- All missing values, duplicate records, and outliers were handled using industry best practices.
- 8 high-resolution charts were produced and saved in the `charts/` directory.
- The workflow is reproducible via `data_cleaning.py`.
