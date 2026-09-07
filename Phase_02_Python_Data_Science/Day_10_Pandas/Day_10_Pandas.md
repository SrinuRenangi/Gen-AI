# 🐼 Day 10: Pandas — Working with Real-World Data
## The Programmable Spreadsheet: Cleaning & Preparing Datasets for AI

[![Phase](https://img.shields.io/badge/Phase_02-Python_Data_Science-brightgreen.svg?style=for-the-badge)](../../README.md)
[![Day](https://img.shields.io/badge/Day-10_of_50-blue.svg?style=for-the-badge)](../../README.md)
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner_Friendly-00cc00.svg?style=for-the-badge)](../../README.md)
[![Prerequisites](https://img.shields.io/badge/Prerequisites-Day_09_NumPy-yellow.svg?style=for-the-badge)](../Day_09_NumPy/Day_09_NumPy.md)

---

## 📌 What Will You Learn Today?

Yesterday on [Day 09](../Day_09_NumPy/Day_09_NumPy.md), you mastered **NumPy**: hardware-accelerated math on raw numbers.

But real-world data is messy:
- Datasets have **column names** (`"Customer_Name"`, `"Salary"`, `"Signup_Date"`).
- Datasets contain **missing values** (`NaN` / empty cells).
- Datasets contain **text categories** (`"Male"`, `"Female"`, `"India"`, `"USA"`).

If you try to load a messy 2-million-row CSV into Microsoft Excel, Excel freezes and crashes.
In Python, we use **Pandas**: the world's most popular library for loading, cleaning, transforming, and analyzing tabular data.

By the end of today, you will understand:
- ✅ The **Excel on Rocket Fuel** analogy.
- ✅ The two core structures: **Series** (1 column) vs. **DataFrame** (full table).
- ✅ How to inspect any dataset using `.head()`, `.info()`, and `.describe()`.
- ✅ How to slice and filter rows using conditions (`df[df["age"] > 25]`).
- ✅ How to detect and fix missing data (`NaN`) without crashing your AI model.
- ✅ How to group data using `.groupby()` (like SQL `GROUP BY` or Excel pivot tables).
- ✅ **One-Hot Encoding**: How to turn text categories into numbers so AI models can read them.
- ✅ How to prepare **Features ($X$)** and **Labels ($y$)** for Machine Learning!

---

## 🗺️ Table of Contents

- [1. Real-World Analogy: Excel on Rocket Fuel](#1-real-world-analogy-excel-on-rocket-fuel)
- [2. Series vs. DataFrame: The Two Core Structures](#2-series-vs-dataframe-the-two-core-structures)
- [3. Creating and Loading Data in Pandas](#3-creating-and-loading-data-in-pandas)
- [4. Inspecting Any Dataset in 10 Seconds](#4-inspecting-any-dataset-in-10-seconds)
- [5. Filtering & Slicing: Asking Questions of Your Data](#5-filtering--slicing-asking-questions-of-your-data)
- [6. The Silent Killer of AI Models: Missing Data (NaN)](#6-the-silent-killer-of-ai-models-missing-data-nan)
- [7. Aggregations & GroupBy: Instant Business Analytics](#7-aggregations--groupby-instant-business-analytics)
- [8. One-Hot Encoding: Converting Words to Numbers](#8-one-hot-encoding-converting-words-to-numbers)
- [9. Preparing Features (X) and Labels (y) for Machine Learning](#9-preparing-features-x-and-labels-y-for-machine-learning)
- [10. Key Takeaways & Cheat Sheet](#10-key-takeaways--cheat-sheet)
- [11. Practice Exercises with Solutions](#11-practice-exercises-with-solutions)

---

# 1. Real-World Analogy: Excel on Rocket Fuel

Every software engineer knows Microsoft Excel:
- You have rows, columns, and cell formulas.
- But Excel has severe limits:
  - Try opening a 5 GB CSV file with 10 million customer transactions $\to$ **Excel crashes**.
  - Try automating an 18-step data cleaning pipeline $\to$ **Manual copying, pasting, and human errors**.

**Pandas is a programmable, automated spreadsheet engine built right into Python:**
- It can process millions of rows in seconds.
- It never makes manual copy-paste errors.
- It integrates seamlessly with NumPy, PyTorch, and all AI frameworks.

---

# 2. Series vs. DataFrame: The Two Core Structures

Pandas has only two main data structures you need to master:

```
           PANDAS SERIES (1D)                 PANDAS DATAFRAME (2D)
           
              Age (Column)                   Name      Age     Salary
           ┌──────────────┐               ┌─────────┬───────┬──────────┐
         0 │      25      │             0 │ Rahul   │   25  │  60,000  │
         1 │      30      │             1 │ Sneha   │   30  │  85,000  │
         2 │      28      │             2 │ Aman    │   28  │  72,000  │
           └──────────────┘               └─────────┴───────┴──────────┘
           A single column!               A table of multiple columns!
```

1. **Series**: A single column of data with an index. Think of it as a 1D NumPy vector with labels!
2. **DataFrame**: The entire 2D table. Think of it as a 2D NumPy matrix where every column has a name (`"Age"`, `"Salary"`) and every row has an index (`0, 1, 2`).

---

# 3. Creating and Loading Data in Pandas

To use Pandas, we import it with the universal community convention:
```python
import pandas as pd
import numpy as np
```

### Method 1: Creating a DataFrame from a Python Dictionary
```python
# A dictionary where keys are column names, and values are lists of data:
data = {
    "Name":       ["Rahul", "Sneha", "Aman", "Priya", "Vikram"],
    "City":       ["Bangalore", "Hyderabad", "Bangalore", "Mumbai", "Delhi"],
    "Age":        [25, 31, 28, 22, 35],
    "Experience": [2.5, 7.0, 4.5, 1.0, 11.0],
    "Salary":     [60000, 120000, 85000, 45000, 180000]
}

df = pd.DataFrame(data)
print(df)
```

**Output:**
```
     Name       City  Age  Experience  Salary
0   Rahul  Bangalore   25         2.5   60000
1   Sneha  Hyderabad   31         7.0  120000
2    Aman  Bangalore   28         4.5   85000
3   Priya     Mumbai   22         1.0   45000
4  Vikram      Delhi   35        11.0  180000
```

### Method 2: Loading from Real-World Files
In real-world AI projects, data lives in CSV, Excel, or JSON files:
```python
# Loading data from CSV or Excel:
# df = pd.read_csv("customers.csv")
# df = pd.read_excel("sales.xlsx")
# df = pd.read_json("records.json")
```

---

# 4. Inspecting Any Dataset in 10 Seconds

Whenever an AI engineer receives a new dataset, they run these four commands before doing anything else:

```python
# 1. Look at the first 3 rows:
print("--- HEAD (First 3 rows) ---")
print(df.head(3))

# 2. Check table shape: (Rows, Columns)
print("\nShape (Rows, Columns):", df.shape)  # (5, 5)

# 3. Inspect column data types & non-null counts:
print("\n--- INFO ---")
df.info()

# 4. Quick statistical summary of all numerical columns:
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())
```

Look at what `.describe()` gives you in one line:
- `count`: How many values exist
- `mean`: The average
- `std`: Standard deviation from Day 06
- `min`, `25%`, `50%` (median!), `75%`, `max`

---

# 5. Filtering & Slicing: Asking Questions of Your Data

In traditional software, you'd write complex nested `for` loops and `if` checks to filter data. 
In Pandas, you filter declaratively in one line:

### 1. Selecting Specific Columns
```python
# Select a single column (returns a Series):
salaries = df["Salary"]

# Select multiple columns (returns a smaller DataFrame):
subset = df[["Name", "Salary"]]
print(subset)
```

### 2. Filtering Rows by Condition (Boolean Indexing)
```python
# Question: Who earns more than ₹80,000?
high_earners = df[df["Salary"] > 80000]
print(high_earners[["Name", "City", "Salary"]])
```

**Output:**
```
     Name       City  Salary
1   Sneha  Hyderabad  120000
2    Aman  Bangalore   85000
4  Vikram      Delhi  180000
```

### 3. Combining Multiple Conditions
In Pandas, use `&` for AND, and `|` for OR (wrap each condition in parentheses `()`):

```python
# Question: Who is based in Bangalore AND older than 26?
bangalore_seniors = df[(df["City"] == "Bangalore") & (df["Age"] > 26)]
print(bangalore_seniors)
```

---

# 6. The Silent Killer of AI Models: Missing Data (NaN)

In real-world data, users leave fields blank, sensors disconnect, and values go missing.
Pandas represents missing numbers as **`NaN`** (*Not a Number*).

> **Why Missing Data Kills AI Models**:
> Neural networks and linear regression algorithms cannot do math with `NaN`.
> If you feed a single `NaN` into a matrix multiplication, your entire model outputs `NaN`!

Let's create a dataset with missing values:

```python
messy_data = {
    "Bedrooms":  [3, np.nan, 4, 2, np.nan],
    "Bathrooms": [2, 1, np.nan, 1, 3],
    "Price":     [5000000, 3500000, 7500000, np.nan, 9000000]
}

housing_df = pd.DataFrame(messy_data)
print("Messy Housing Data:\n", housing_df)
```

### Step 1: Detect Missing Values
```python
# Count how many missing values exist in each column:
print("Missing values per column:")
print(housing_df.isna().sum())
```

### Step 2: Fixing Missing Values
You have two primary strategies:

#### Strategy A: Drop Rows with Missing Values (`dropna`)
Use this when you have millions of rows and losing a few doesn't hurt:
```python
clean_dropped = housing_df.dropna()
print("After dropping rows with NaN:\n", clean_dropped)
```

#### Strategy B: Impute (Fill) with the Column Average (`fillna`)
Use this when you cannot afford to throw away valuable data:
```python
# Fill missing bedrooms with the average bedroom count:
avg_bedrooms = housing_df["Bedrooms"].mean()
housing_df["Bedrooms"] = housing_df["Bedrooms"].fillna(avg_bedrooms)

print("After filling missing bedrooms with average (", avg_bedrooms, "):\n", housing_df)
```

---

# 7. Aggregations & GroupBy: Instant Business Analytics

Just like SQL `GROUP BY`, Pandas allows you to split a dataset by a category and calculate metrics:

```python
# What is the average salary and average age per City?
city_analytics = df.groupby("City")[["Salary", "Age"]].mean()
print("Analytics by City:\n", city_analytics)
```

**Output:**
```
Analytics by City:
              Salary   Age
City                      
Bangalore    72500.0  26.5
Delhi       180000.0  35.0
Hyderabad   120000.0  31.0
Mumbai       45000.0  22.0
```

---

# 8. One-Hot Encoding: Converting Words to Numbers

Remember the fundamental rule from [Day 01](../Phase_01_Math_Foundations/Day_01_Numbers_Variables_Functions/Day_01_Numbers_Variables_Functions.md):
> **Computers and AI models can ONLY do math on NUMBERS.**

Look at the `City` column in our DataFrame:
`["Bangalore", "Hyderabad", "Mumbai", "Delhi"]`

You cannot multiply `"Bangalore"` by a weight matrix $W$!
We must convert text categories into numbers.

### The Bad Idea: Arbitrary Labeling
What if we set `Bangalore = 1, Hyderabad = 2, Delhi = 3`?
The AI will think Delhi ($3$) is "three times greater" than Bangalore ($1$)! But cities have no mathematical order!

### The Right Idea: One-Hot Encoding (`pd.get_dummies`)
We create a new column for each city with a binary `1` (Yes) or `0` (No):

```python
# One-Hot Encoding the 'City' column:
df_encoded = pd.get_dummies(df, columns=["City"], dtype=int)
print(df_encoded)
```

**Output:**
```
     Name  Age  Experience  Salary  City_Bangalore  City_Delhi  City_Hyderabad  City_Mumbai
0   Rahul   25         2.5   60000               1           0               0            0
1   Sneha   31         7.0  120000               0           0               1            0
2    Aman   28         4.5   85000               1           0               0            0
3   Priya   22         1.0   45000               0           0               0            1
4  Vikram   35        11.0  180000               0           1               0            0
```

Now every city is represented by pure binary numbers (`1` or `0`) that our neural network can easily multiply!

---

# 9. Preparing Features (X) and Labels (y) for Machine Learning

In [Phase 3 (Classical Machine Learning)](../../README.md#🟡-phase-3-classical-machine-learning-days-1217), every single model expects data divided into two components:
1. **Features ($X$)**: The input columns used to make the prediction (a 2D Matrix).
2. **Target / Label ($y$)**: The answer column we want to predict (a 1D Vector).

Suppose we want to predict **Salary** based on Age, Experience, and City:

```python
# 1. Features (X): Everything EXCEPT the Name and the Target (Salary)
X = df_encoded.drop(columns=["Name", "Salary"])

# 2. Target Label (y): Only the Salary column
y = df_encoded["Salary"]

print("--- FEATURES (X: 2D Matrix) ---")
print(X)

print("\n--- TARGET LABEL (y: 1D Vector) ---")
print(y.values)
```

**Output:**
```
--- FEATURES (X: 2D Matrix) ---
   Age  Experience  City_Bangalore  City_Delhi  City_Hyderabad  City_Mumbai
0   25         2.5               1           0               0            0
1   31         7.0               0           1               0            0
2   28         4.5               1           0               0            0
3   22         1.0               0           0               0            1
4   35        11.0               0           0               1            0

--- TARGET LABEL (y: 1D Vector) ---
[ 60000 120000  85000  45000 180000]
```

Your data is now 100% clean, numerical, and ready to be fed directly into an AI model!

---

# 10. Key Takeaways & Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DAY 10 CHEAT SHEET                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  🐼 PANDAS = High-performance tabular data library for Python          │
│     • Series: 1D labeled column                                        │
│     • DataFrame: 2D labeled table (rows x columns)                     │
│                                                                        │
│  🔍 INSPECTING DATA:                                                   │
│     • df.head(n)       -> View first n rows                            │
│     • df.shape         -> (num_rows, num_cols)                         │
│     • df.info()        -> Column data types & missing value check      │
│     • df.describe()    -> Mean, std, min, median, max summary          │
│                                                                        │
│  🎯 FILTERING:                                                         │
│     • df[df["col"] > value]                                            │
│     • df[(df["c1"] == "A") & (df["c2"] > 10)]                          │
│                                                                        │
│  🩹 MISSING DATA (NaN):                                                │
│     • df.isna().sum()            -> Count missing values               │
│     • df.dropna()                -> Remove rows with missing data      │
│     • df.fillna(df.mean())       -> Impute missing values with average │
│                                                                        │
│  🔤 ONE-HOT ENCODING:                                                  │
│     • pd.get_dummies(df, columns=["Category"])                         │
│     • Converts text categories into binary 0 and 1 columns for AI!     │
│                                                                        │
│  🤖 ML PREPARATION:                                                    │
│     • X = df.drop(columns=["target"])  (2D Features Matrix)            │
│     • y = df["target"]                 (1D Target Vector)              │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

# 11. Practice Exercises with Solutions

<details>
<summary><b>🏋️ Exercise 1: Clean a Customer Churn Table</b></summary>
<br/>

Given this raw customer dataset:
```python
raw_data = {
    "CustomerID": [101, 102, 103, 104, 105],
    "Age":        [24, np.nan, 45, 32, np.nan],
    "Plan":       ["Basic", "Premium", "Basic", "Standard", "Premium"],
    "MonthlyFee": [199, 499, np.nan, 299, 499],
    "Churned":    [0, 1, 0, 0, 1]
}
df_cust = pd.DataFrame(raw_data)
```

Write code to:
1. Fill missing `Age` with the median age.
2. Fill missing `MonthlyFee` with the average fee.
3. One-hot encode the `Plan` column.
4. Separate the table into Features $X$ (drop `CustomerID` and `Churned`) and Target $y$ (`Churned`).

**Solution:**
```python
import pandas as pd
import numpy as np

raw_data = {
    "CustomerID": [101, 102, 103, 104, 105],
    "Age":        [24, np.nan, 45, 32, np.nan],
    "Plan":       ["Basic", "Premium", "Basic", "Standard", "Premium"],
    "MonthlyFee": [199, 499, np.nan, 299, 499],
    "Churned":    [0, 1, 0, 0, 1]
}
df_cust = pd.DataFrame(raw_data)

# 1. Fill missing Age with median
median_age = df_cust["Age"].median()
df_cust["Age"] = df_cust["Age"].fillna(median_age)

# 2. Fill missing MonthlyFee with mean
mean_fee = df_cust["MonthlyFee"].mean()
df_cust["MonthlyFee"] = df_cust["MonthlyFee"].fillna(mean_fee)

# 3. One-hot encode 'Plan'
df_cust = pd.get_dummies(df_cust, columns=["Plan"], dtype=int)

# 4. Separate X and y
X = df_cust.drop(columns=["CustomerID", "Churned"])
y = df_cust["Churned"]

print("Clean Features Matrix (X):\n", X)
print("\nTarget Vector (y):\n", y.values)
```
</details>

<details>
<summary><b>🏋️ Exercise 2: Top Spender Analysis (GroupBy & Sort)</b></summary>
<br/>

Given an e-commerce order table:
```python
orders = pd.DataFrame({
    "Order_ID": [1, 2, 3, 4, 5, 6],
    "User":     ["Rahul", "Sneha", "Rahul", "Aman", "Sneha", "Rahul"],
    "Amount":   [1200, 4500, 800, 3100, 2200, 1500]
})
```

Write code to:
1. Calculate the **total amount spent** by each user using `.groupby()`.
2. Sort the users from **highest spender to lowest spender** using `.sort_values()`.

**Solution:**
```python
orders = pd.DataFrame({
    "Order_ID": [1, 2, 3, 4, 5, 6],
    "User":     ["Rahul", "Sneha", "Rahul", "Aman", "Sneha", "Rahul"],
    "Amount":   [1200, 4500, 800, 3100, 2200, 1500]
})

# Group by User, sum the Amount, sort descending
top_spenders = orders.groupby("User")["Amount"].sum().sort_values(ascending=False)

print("Top Spenders by Total Amount:")
print(top_spenders)
# Sneha: 6700
# Rahul: 3500
# Aman:  3100
```
</details>

<details>
<summary><b>🏋️ Exercise 3: Vectorized Column Transformation</b></summary>
<br/>

Create a new column called `"Seniority"` based on Experience:
- If Experience $\ge 5.0$, `"Senior"`
- If Experience $< 5.0$, `"Junior"`

*(Hint: You can use `np.where(condition, if_true, if_false)` for instant vectorized `if-else`!)*

**Solution:**
```python
df["Seniority"] = np.where(df["Experience"] >= 5.0, "Senior", "Junior")
print(df[["Name", "Experience", "Seniority"]])
```
</details>

---

## ⏭️ What's Next: Day 11 — Matplotlib & Data Visualization

Today, you mastered **Pandas**: the tool for organizing, cleaning, and encoding tabular datasets.

Tomorrow on **Day 11**, we complete Phase 2 with **Matplotlib**:
- Why staring at rows of numbers is dangerous (Anscombe's Quartet).
- Plotting line graphs, scatter plots, and histograms in Python.
- Visualizing model predictions vs. actual data.
- **Heatmaps**: How AI researchers visualize **Attention Weights** inside Transformers!

---

<p align="center">
  <b>🌟 End of Day 10 — You've mastered Pandas, the dataset engine of AI! 🌟</b>
</p>
