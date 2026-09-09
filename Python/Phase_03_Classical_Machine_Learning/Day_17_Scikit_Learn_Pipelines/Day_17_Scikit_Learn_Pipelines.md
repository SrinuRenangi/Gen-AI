# Day 17: Scikit-Learn Hands-On — Building Complete ML Pipelines


| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 16: Model Evaluation](../Day_16_Model_Evaluation/Day_16_Model_Evaluation.md) | [All 50 Days Overview](../../README.md) | [Day 18: The Artificial Neuron →](../../Phase_04_Deep_Learning_Foundations/Day_18_The_Artificial_Neuron/Day_18_The_Artificial_Neuron.md) |

> **"Amateur data scientists write 500 lines of messy Pandas glue code. Senior ML engineers write a 10-line production pipeline that never leaks data and deploys in one command."**  
> Welcome to Day 17 — the grand finale of **Phase 3: Classical Machine Learning**! Today, we bridge the gap between experimental notebook code and bulletproof production software.

---

## 🧭 The Mental Compass: The Automobile Assembly Line

Imagine building a car:

```
    APPROACH A: The Artisanal Garage (Spaghetti Code)
    ────────────────────────────────────────────────
    1. Mechanic A sands individual door panels on the floor.
    2. Mechanic B paints the chassis before the wheels are mounted.
    3. Mechanic C forgets which bolts belong to the engine.
    4. When a customer orders a second car, the team starts from scratch by hand.
    ❌ Error-prone, irreproducible, impossible to automate.

    APPROACH B: The Modern Henry Ford Assembly Line (Pipelines)
    ──────────────────────────────────────────────────────────
    1. Raw steel and unpainted components enter Station 1 (Automated Imputation).
    2. Station 2 standardizes dimensions to millimeter tolerances (StandardScaler).
    3. Station 3 converts raw paint codes to finish coats (OneHotEncoder).
    4. Station 4 installs the powertrain (Estimator / Classifier).
    5. A fully certified, road-ready car rolls off the belt every 60 seconds.
    ✅ 100% reproducible, zero human error, instant production scaling!
```

In software engineering, you would never deploy raw database queries and unvalidated user inputs directly into your backend business logic. In Machine Learning, a **Pipeline** is your end-to-end assembly line.

---

## 1. The Cardinal Sin of Machine Learning: Data Leakage

Before writing a single line of pipeline code, we must understand the #1 mistake that costs AI companies millions of dollars: **Data Leakage**.

```
                           THE DATA LEAKAGE TRAP
                           ─────────────────────

   ❌ THE AMATEUR WORKFLOW:
   Raw Data ──► [Compute Global Mean & Std Dev on ENTIRE dataset] ──► Split Train / Test
                                                                             │
                      Information from TEST set leaked into TRAIN set! ◄─────┘
                      Model gets 97% test accuracy... then crashes in production!

   ✅ THE PROFESSIONAL WORKFLOW:
   Raw Data ──► Split Train / Test FIRST!
                      │
                      ├──► Train Set ──► [FIT Scaler on Train ONLY] ──► [TRANSFORM Train] ──► Train Model
                      │                                                        │
                      └──► Test Set  ──────────────────────────────────► [TRANSFORM Test]  ──► Evaluate
                                                                         (Using Train Mean!)
```

### Why Fitting Scalers on Full Data Destroys Models
Suppose you want to standardize house prices using $z = \frac{x - \mu}{\sigma}$.
* If you compute the mean $\mu$ and standard deviation $\sigma$ on the **entire dataset** before splitting, your training features now contain subtle hints about the distribution of the test set.
* When your model goes to production and receives a customer input from next year, the future mean is unknown. The model encounters a catastrophic distribution mismatch.

> [!CAUTION]
> **Golden Rule of ML Engineering:**  
> Your model and preprocessors must **NEVER** see, touch, or calculate statistics on test data during training. You only call `.fit()` on training data; you only call `.transform()` on validation/test/production data.

---

## 2. Scikit-Learn Design Philosophy: Transformers vs Estimators

Every component in Scikit-Learn obeys a strict, predictable object-oriented interface:

```
               ┌─────────────────────────────────────────────────────────┐
               │              THE SCIKIT-LEARN INTERFACE                 │
               └─────────────────────────────────────────────────────────┘

           TRANSFORMERS (Preprocessors)             ESTIMATORS (Predictors)
           e.g. StandardScaler, OneHotEncoder       e.g. RandomForest, LogisticRegression
           ─────────────────────────────────        ─────────────────────────────────────
    .fit(X)          Learns internal stats          .fit(X, y)       Learns patterns, weights,
                     (e.g., means, standard devs,                    or decision tree splits.
                     unique category labels).

    .transform(X)    Applies learned stats to       .predict(X)      Infers continuous values
                     transform data into numbers.                    or discrete class labels.

    .fit_transform() Convenience: fit and           .predict_proba() Infers class probabilities
                     transform in one call.                          (for classifiers).
```

---

## 3. The Full Pipeline Architecture: ColumnTransformer & Estimators

Real-world datasets are messy and heterogeneous:
* **Numerical columns** (`Age`, `Annual_Income`, `Credit_Score`): Have missing `NaN` values and vastly different scales ($25$ vs $\$120,000$).
* **Categorical columns** (`City`, `Education_Level`, `Marital_Status`): Stored as strings with missing values and unseen categories in production.

Scikit-Learn provides **`ColumnTransformer`** to split features into parallel processing branches, transform them independently, concatenate them together, and feed them straight into the model!

![Scikit-Learn Pipeline Architecture](assets/ml_pipeline_column_transformer_flow.svg)

---

## 4. K-Fold Cross-Validation: Banishing "Split Luck"

A standard `train_test_split` creates a single random split. But what if your 20% test split accidentally got all the easiest examples? You get an artificially high 96% score. What if it got all the outliers? You get a depressing 79% score.

**K-Fold Cross-Validation** solves this by splitting the training data into $K$ equal slices (typically $K=5$).

![5-Fold Cross-Validation Architecture](assets/k_fold_cross_validation_diagram.svg)

Each fold gets an exact turn serving as the validation benchmark while the other 4 folds train the model. The final evaluation is the **mean $\pm$ standard deviation** across all 5 iterations.

---

## 5. End-to-End Hands-On Python Lab

Let's build a complete, real-world loan approval pipeline from raw, dirty tabular data all the way to a serialized `.joblib` model ready for microservice deployment!

```python
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# =====================================================================
# STEP 1: CREATE A REALISTIC, DIRTY TABULAR DATASET
# =====================================================================
np.random.seed(42)
n_samples = 1200

data = {
    # Numerical features (with missing NaNs and different scales)
    "age": np.random.choice([22, 28, 35, 45, 52, np.nan, 60], size=n_samples),
    "income": np.random.exponential(scale=65000, size=n_samples) + 20000,
    "debt_ratio": np.random.uniform(0.05, 0.60, size=n_samples),
    
    # Categorical features (strings with missing values)
    "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", None], p=[0.4, 0.2, 0.35, 0.05], size=n_samples),
    "loan_intent": np.random.choice(["EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT"], size=n_samples),
}

df = pd.DataFrame(data)

# Target: Loan Approved (1) or Rejected (0)
# High income & low debt ratio increase approval probability
approval_score = (
    0.00003 * df["income"].fillna(45000) 
    - 4.0 * df["debt_ratio"] 
    + 0.02 * df["age"].fillna(30)
    + (df["home_ownership"] == "OWN").astype(int) * 1.5
    - 0.5
)
df["loan_approved"] = (approval_score > approval_score.median()).astype(int)

print("--- RAW DATA SAMPLE (FIRST 3 ROWS) ---")
print(df.head(3))
print("\nMissing values per column:\n", df.isnull().sum())

# =====================================================================
# STEP 2: SPLIT FEATURES AND TARGET, THEN TRAIN / TEST SPLIT
# =====================================================================
X = df.drop(columns=["loan_approved"])
y = df["loan_approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Identify numerical and categorical column names
num_cols = ["age", "income", "debt_ratio"]
cat_cols = ["home_ownership", "loan_intent"]

# =====================================================================
# STEP 3: CONSTRUCT THE SUB-PIPELINES & COLUMN TRANSFORMER
# =====================================================================

# Branch 1: Numerical Transformer
# 1) Fill missing numbers with median
# 2) Standardize to zero mean and unit variance
num_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Branch 2: Categorical Transformer
# 1) Fill missing strings with "MISSING"
# 2) Convert strings to one-hot binary columns
cat_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

# Assemble branches into a unified ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])

# =====================================================================
# STEP 4: ASSEMBLE FULL PIPELINE (PREPROCESSOR + ESTIMATOR)
# =====================================================================
full_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# =====================================================================
# STEP 5: 5-FOLD CROSS-VALIDATION
# =====================================================================
cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring="roc_auc")
print("\n" + "=" * 60)
print(f"5-Fold Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Individual fold scores: {[round(s, 4) for s in cv_scores]}")
print("=" * 60)

# =====================================================================
# STEP 6: HYPERPARAMETER TUNING VIA GRIDSEARCHCV
# =====================================================================
# Note the double underscore syntax: <step_name>__<parameter_name>
param_grid = {
    "classifier__n_estimators": [50, 100],
    "classifier__max_depth": [4, 8, None],
    "classifier__min_samples_leaf": [1, 4]
}

print("\nTuning hyperparameters across folds...")
grid_search = GridSearchCV(
    full_pipeline, 
    param_grid, 
    cv=3, 
    scoring="roc_auc", 
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# Extract optimal pipeline
best_pipeline = grid_search.best_estimator_

# =====================================================================
# STEP 7: FINAL EVALUATION ON UNTOUCHED TEST SET
# =====================================================================
test_preds = best_pipeline.predict(X_test)
test_probs = best_pipeline.predict_proba(X_test)[:, 1]

print("\n--- FINAL TEST SET PERFORMANCE ---")
print(classification_report(y_test, test_preds, target_names=["Rejected", "Approved"]))
print(f"Test Set ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")

# =====================================================================
# STEP 8: PRODUCTION SERIALIZATION WITH JOBLIB
# =====================================================================
model_filename = "loan_approval_pipeline.joblib"
joblib.dump(best_pipeline, model_filename)
print(f"\n✅ Pipeline successfully serialized to: {model_filename}")

# =====================================================================
# STEP 9: SIMULATE PRODUCTION API INFERENCE (ZERO MANUAL PREPROCESSING!)
# =====================================================================
# Load serialized pipeline from disk in another process / server
deployed_pipeline = joblib.load(model_filename)

# Raw JSON payload from an applicant's web browser
raw_applicant_payload = pd.DataFrame([{
    "age": np.nan,            # Missing value! Pipeline imputes it.
    "income": 115000.0,       # Raw currency! Pipeline standardizes it.
    "debt_ratio": 0.18,       # Float feature.
    "home_ownership": "OWN",  # String category! Pipeline one-hot encodes it.
    "loan_intent": "VENTURE"  # String category!
}])

predicted_approval = deployed_pipeline.predict(raw_applicant_payload)[0]
confidence = deployed_pipeline.predict_proba(raw_applicant_payload)[0, 1]

print("\n--- PRODUCTION INFERENCE SIMULATION ---")
print(f"Incoming Raw Payload:\n{raw_applicant_payload.to_dict(orient='records')[0]}")
print(f"Decision   : {'APPROVED (1)' if predicted_approval == 1 else 'REJECTED (0)'}")
print(f"Confidence : {confidence:.2%}")
```

---

## 6. The Architect's Production Checklist

When promoting ML code to production, audit your system against these 5 engineering criteria:

| Production Standard | The Anti-Pattern (Amateur) | The Production Pattern (Pro) |
| :--- | :--- | :--- |
| **Imputation & Scaling** | Hardcoded Pandas `.fillna()` scripts and manually saved `.csv` stats. | Bundled inside a Scikit-Learn `ColumnTransformer`. |
| **Unknown Categories** | Model crashes with `ValueError: unseen label` when a new city appears. | `OneHotEncoder(handle_unknown="ignore")` safely zeros out unseen categories. |
| **Data Leakage** | Scaling before train/test splitting. | Pipeline fits parameters **strictly inside each training fold**. |
| **Model Artifact** | Saving weights in one file, preprocessing dictionaries in another file. | Single serialized `.joblib` file containing preprocessor + model together. |
| **Serving Interface** | API endpoint requires 40 lines of feature transformation code. | Endpoint calls `model.predict(raw_dataframe)` in 1 line. |

---

## ✍️ Self-Check Exercises & Practice Problems

Solidify your understanding of production pipelines and data leakage prevention with these practical questions!

### 🏋️ Problem 1: Catching the Data Leakage Bug
A junior engineer wrote the following pre-processing code for an insurance claim risk model:

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
```

**Your Tasks:**
1. What critical machine learning violation occurred here?
2. Why will the validation score on `X_test` look deceptively optimistic?
3. How does wrapping preprocessing inside a `sklearn.pipeline.Pipeline` eliminate this bug permanently?

---

### 🏋️ Problem 2: Tracing ColumnTransformer Output Dimensions
You are building an end-to-end ColumnTransformer for a customer churn dataset with the following features:
- **Numerical Features (4 columns):** `['MonthlyCharges', 'TotalCharges', 'TenureMonths', 'SupportTickets']` &rarr; Preprocessed with `StandardScaler()`.
- **Categorical Feature 1 (`'Contract'`):** 3 unique categories (`'Month-to-month'`, `'One-year'`, `'Two-year'`) &rarr; Preprocessed with `OneHotEncoder(drop='first')`.
- **Categorical Feature 2 (`'PaymentMethod'`):** 4 unique categories (`'Electronic check'`, `'Mailed check'`, `'Bank transfer'`, `'Credit card'`) &rarr; Preprocessed with `OneHotEncoder(drop=None)`.

**Your Tasks:**
1. How many feature columns will the numerical transformer output?
2. How many feature columns will `'Contract'` output with `drop='first'`?
3. How many feature columns will `'PaymentMethod'` output with `drop=None`?
4. What is the total feature dimension ($D$) fed into the classifier estimator?

<details>
<summary><b>🔍 Click to Reveal Step-by-Step Solutions</b></summary>

### Solution 1:
1. **Data Leakage (Lookahead Bias):** Calling `scaler.fit_transform(X)` on the entire dataset calculates the global mean $\mu_{\text{all}}$ and variance $\sigma^2_{\text{all}}$ across all rows, including the test set. Information from the unseen test set leaked into the feature representation of the training data!
2. **Deceptive Validation:** Because the scaler already "saw" the test distribution parameters, the model is tested on data it partially peeked at during scaling. When deployed to live production on genuinely new customers, the model's accuracy will plummet.
3. **Pipeline Solution:** A `Pipeline` ensures that during cross-validation, `scaler.fit()` is executed **strictly on `X_train`**, and `scaler.transform()` is applied to `X_test` using the training mean and variance.

---

### Solution 2:
1. **Numerical Transformer:** Outputs **4 columns** (each numerical column is centered and scaled independently).
2. **Contract Feature (`drop='first'`):** 3 categories $- 1$ dropped baseline $= \mathbf{2\text{ columns}}$.
3. **PaymentMethod Feature (`drop=None`):** All 4 categories retained $= \mathbf{4\text{ columns}}$.
4. **Total Dimension $D$:**
   $$D_{\text{total}} = 4 + 2 + 4 = \mathbf{10\text{ feature columns}}$$
   The classifier receives an input matrix $X \in \mathbb{R}^{N \times 10}$.
</details>

---

## 7. Phase 3 Milestone Complete! 🏆

Congratulations! You have completed **Phase 3: Classical Machine Learning (Days 12–17)**:

* [x] **Day 12:** Machine Learning Paradigm Shift (Rules vs Signal, ML Taxonomy).
* [x] **Day 13:** Linear Regression, MSE Loss, Gradient Optimization.
* [x] **Day 14:** Logistic Regression, Sigmoid Curves, Log Loss, Decision Boundaries.
* [x] **Day 15:** Decision Trees, Gini Impurity, Random Forests, Bagging Ensembles.
* [x] **Day 16:** Model Evaluation, Confusion Matrix, Precision, Recall, F1, ROC-AUC.
* [x] **Day 17:** End-to-End Scikit-Learn Pipelines, Cross-Validation, Serialization.

---

## 🚀 Coming Up Next: Phase 4 — Deep Learning Foundations

In classical machine learning, we had to carefully engineer and select features. What happens when our data isn't a neat spreadsheet of numbers, but raw pixels, audio waves, or sentences?

Tomorrow in **Day 18**, we enter **Deep Learning**:
* **The Artificial Neuron (Perceptron):** How biology inspired mathematical computation.
* Weights, biases, and the dot-product accumulator inside a single synthetic cell.
* The historical spark that launched the modern AI revolution!


---

## 🧭 Navigation & Next Steps

| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 16: Model Evaluation](../Day_16_Model_Evaluation/Day_16_Model_Evaluation.md) | [All 50 Days Overview](../../README.md) | [Day 18: The Artificial Neuron →](../../Phase_04_Deep_Learning_Foundations/Day_18_The_Artificial_Neuron/Day_18_The_Artificial_Neuron.md) |
