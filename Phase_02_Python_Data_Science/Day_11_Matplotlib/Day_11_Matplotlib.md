# 📊 Day 11: Matplotlib & Data Visualization
## The X-Ray of AI: Plotting Loss Curves, Distributions, and Attention Heatmaps

[![Phase](https://img.shields.io/badge/Phase_02-Python_Data_Science-brightgreen.svg?style=for-the-badge)](../../README.md)
[![Day](https://img.shields.io/badge/Day-11_of_50-blue.svg?style=for-the-badge)](../../README.md)
[![Difficulty](https://img.shields.io/badge/Difficulty-Beginner_Friendly-00cc00.svg?style=for-the-badge)](../../README.md)
[![Prerequisites](https://img.shields.io/badge/Prerequisites-Day_10_Pandas-yellow.svg?style=for-the-badge)](../Day_10_Pandas/Day_10_Pandas.md)

---

## 📌 What Will You Learn Today?

Welcome to the **Grand Finale of Phase 2: Python Data Science Toolkit**!

Over the past two days, you mastered:
- **NumPy** ([Day 09](../Day_09_NumPy/Day_09_NumPy.md)): Fast numerical math on vectors and matrices.
- **Pandas** ([Day 10](../Day_10_Pandas/Day_10_Pandas.md)): Loading, cleaning, and encoding tabular datasets.

Today, we complete your data science foundation with **Matplotlib**: the industry-standard visualization library in Python.

In AI engineering, staring at raw numbers in a terminal will deceive you. You cannot train an AI model blindfolded.
- You must **see** if the error is decreasing over time.
- You must **see** if your dataset has outliers.
- You must **see** how your model's predictions compare to real ground truth.
- You must **see** which words in a sentence a Transformer model is paying attention to!

By the end of today, you will understand:
- ✅ The **X-Ray analogy**: Why visualizing data prevents catastrophic AI mistakes.
- ✅ The **4 Essential Charts** every AI engineer uses daily: Line plots, Scatter plots, Histograms, and Bar charts.
- ✅ Anatomy of a clean chart: Titles, axis labels, legends, grids, and styling.
- ✅ How to plot **Subplots** (multiple graphs side-by-side).
- ✅ How to plot **Model Predictions vs. Actual Data** (evaluating regression lines).
- ✅ **Heatmaps**: How AI researchers visually inspect **Attention Weights** in Transformers.

---

## 🗺️ Table of Contents

- [1. Real-World Analogy: The Medical X-Ray](#1-real-world-analogy-the-medical-x-ray)
- [2. The 4 Essential Charts in AI Engineering](#2-the-4-essential-charts-in-ai-engineering)
- [3. Chart 1: Line Plots (Tracking Training Loss Over Time)](#3-chart-1-line-plots-tracking-training-loss-over-time)
- [4. Chart 2: Scatter Plots (Feature Relationships & Patterns)](#4-chart-2-scatter-plots-feature-relationships--patterns)
- [5. Chart 3: Histograms (Checking Data Distributions & Bell Curves)](#5-chart-3-histograms-checking-data-distributions--bell-curves)
- [6. Subplots: Side-by-Side Comparisons](#6-subplots-side-by-side-comparisons)
- [7. Visualizing AI Predictions: Fitting the Curve](#7-visualizing-ai-predictions-fitting-the-curve)
- [8. Heatmaps: How AI Visualizes Transformer Attention](#8-heatmaps-how-ai-visualizes-transformer-attention)
- [9. Phase 2 Milestone Reflection: The Data Science Toolkit is Complete!](#9-phase-2-milestone-reflection-the-data-science-toolkit-is-complete)
- [10. Key Takeaways & Cheat Sheet](#10-key-takeaways--cheat-sheet)
- [11. Practice Exercises with Solutions](#11-practice-exercises-with-solutions)

---

# 1. Real-World Analogy: The Medical X-Ray

Imagine a doctor examining a patient with a broken leg:
- **Option A (Staring at raw numbers)**: The doctor reads a 20-page printout of blood cell counts, calcium density values, and bone coordinate tables. *(It takes 2 hours, and they might still miss the hairline fracture!)*
- **Option B (The X-Ray)**: The doctor looks at an illuminated X-ray picture on the wall. In **half a second**, the break is obvious.

**Data Visualization is the X-ray of Artificial Intelligence:**
- In 1973, statistician Francis Anscombe created **Anscombe's Quartet**: four different datasets that have the **exact same mean, exact same variance, and exact same correlation numbers**.
- Yet when plotted on a graph, one was a straight line, one was a parabola, one had an extreme outlier, and one was totally vertical!

Never trust summary numbers alone. **Always look at the picture!**

---

# 2. The 4 Essential Charts in AI Engineering

To build and debug AI models, you don't need 50 fancy chart types. You only need these **4 workhorses**:

```
┌─────────────────┬───────────────────────────────┬──────────────────────────────┐
│ Chart Type      │ What It Shows                 │ Where It Is Used in AI       │
├─────────────────┼───────────────────────────────┼──────────────────────────────┤
│ 📈 Line Plot    │ Continuous change over time   │ Training Loss curves (error) │
│ 🔵 Scatter Plot │ Relationship between 2 features│ Inputs vs. Outputs / Clusters│
│ 📊 Histogram    │ Distribution of a single value│ Checking for Bell Curves     │
│ 📶 Bar Chart    │ Comparison across categories  │ Model Accuracy / F1 scores   │
└─────────────────┴───────────────────────────────┴──────────────────────────────┘
```

---

# 3. Chart 1: Line Plots (Tracking Training Loss Over Time)

When training a neural network or running Gradient Descent (from Day 08), the most critical question is:
> *"Is the model's error (loss) actually going down as it trains?"*

We track this using a **Line Plot**:

```python
import matplotlib.pyplot as plt
import numpy as np

# Simulated training progress over 20 epochs (iterations):
epochs = list(range(1, 21))

# Loss steadily decreasing from 2.5 down to 0.15:
training_loss = [2.50, 1.80, 1.30, 0.95, 0.72, 0.58, 0.47, 0.39, 0.33, 0.28,
                 0.25, 0.22, 0.20, 0.18, 0.17, 0.16, 0.15, 0.15, 0.14, 0.14]

# Create the figure
plt.figure(figsize=(8, 4.5))

# Plot the line (color='royalblue', with dots at each point)
plt.plot(epochs, training_loss, color="royalblue", marker="o", linewidth=2, label="Training Error")

# Add Labels, Title, and Legend
plt.title("Model Training Progress: Error Over Time", fontsize=14, fontweight="bold")
plt.xlabel("Epoch (Training Round)", fontsize=11)
plt.ylabel("Loss (Error)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

# Save the plot as an image:
# plt.savefig("loss_curve.png", dpi=150)
# plt.show()
```

### What You Are Looking For:
- If the line slopes **steadily downward and flattens out**: Your model is learning properly! 🎯
- If the line **shoots upward**: Your learning rate is too high and weights are exploding! 💥
- If the line is **completely flat from step 1**: Your learning rate is too small or gradients are zero! 🛑

---

# 4. Chart 2: Scatter Plots (Feature Relationships & Patterns)

Before feeding data into an AI model, you use a **Scatter Plot** to see if there is a real pattern between the input and output.

Suppose we want to predict **House Price** based on **Square Footage**:

```python
# 15 houses: Square Footage vs Actual Price (in Lakhs ₹)
sqft = np.array([800, 1000, 1200, 1400, 1500, 1750, 1900, 2100, 2300, 2500, 2700, 2900, 3100, 3300, 3500])
prices = np.array([35, 42, 50, 58, 62, 70, 75, 84, 91, 100, 108, 115, 122, 131, 140])

plt.figure(figsize=(8, 4.5))

# Scatter plot: each house is a distinct blue dot
plt.scatter(sqft, prices, color="dodgerblue", s=70, edgecolors="navy", alpha=0.8, label="Houses")

plt.title("House Area vs. Selling Price", fontsize=14, fontweight="bold")
plt.xlabel("Area (Square Feet)", fontsize=11)
plt.ylabel("Price (in Lakhs ₹)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
```

Look at the scatter plot: the dots form a clear upward diagonal line! That tells the engineer: **Linear Regression (Day 13) will work brilliantly here!**

---

# 5. Chart 3: Histograms (Checking Data Distributions & Bell Curves)

Remember on [Day 06](../Phase_01_Math_Foundations/Day_06_Probability_and_Statistics/Day_06_Probability_and_Statistics.md), we learned about the **Normal Distribution (Bell Curve)**.

A **Histogram** groups numbers into buckets (bins) and shows how frequently values occur:

```python
# Generate 10,000 random initial neural network weights drawn from a Normal Distribution:
np.random.seed(42)
weights = np.random.randn(10000) * 0.02  # Mean 0, Std Dev 0.02

plt.figure(figsize=(8, 4.5))

# Plot histogram with 50 bins
plt.hist(weights, bins=50, color="mediumpurple", edgecolor="black", alpha=0.7)

plt.title("Distribution of 10,000 Initial AI Weights (Bell Curve)", fontsize=13, fontweight="bold")
plt.xlabel("Weight Value", fontsize=11)
plt.ylabel("Frequency (Count)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.6)
```

The histogram clearly forms the classic symmetric **Bell Curve** centered around `0.0`, proving our weights were initialized properly!

---

# 6. Subplots: Side-by-Side Comparisons

In AI engineering, we often compare two things at once:
- **Training Loss vs. Validation Loss**
- **Original Image vs. Filtered Image**

We create multiple sub-panels using `plt.subplots()`:

```python
# Comparing Training Loss vs Validation Loss
epochs = np.arange(1, 16)
train_loss = 1.0 / (epochs ** 0.5)
val_loss   = (1.0 / (epochs ** 0.5)) + (0.02 * epochs) # Starts going back up (Overfitting!)

# Create a figure with 1 row, 2 columns of subplots:
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel 1: Training Loss
axes[0].plot(epochs, train_loss, color="green", marker="o", label="Training Loss")
axes[0].set_title("Training Loss (On Known Data)", fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].legend()

# Panel 2: Validation Loss (Unseen Data)
axes[1].plot(epochs, val_loss, color="crimson", marker="s", label="Validation Loss")
axes[1].set_title("Validation Loss (On Unseen Data)", fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].legend()

plt.tight_layout()  # Automatically adjusts spacing so labels don't overlap!
```

---

# 7. Visualizing AI Predictions: Fitting the Curve

Now let's combine two plots:
1. The **Actual Data Points** (Scatter plot)
2. The **AI Model's Prediction Line** (Line plot)

```python
# 1. Actual house data
sqft = np.array([800, 1200, 1500, 2000, 2500, 3000])
actual_prices = np.array([36, 51, 63, 82, 99, 121])

# 2. Suppose our trained AI model learned: price = 0.038 * sqft + 5.5
# Compute model predictions:
model_predictions = (0.038 * sqft) + 5.5

plt.figure(figsize=(8, 4.5))

# Plot actual data as blue scatter dots:
plt.scatter(sqft, actual_prices, color="royalblue", s=80, label="Actual Ground Truth", zorder=3)

# Plot AI model's prediction as a red line cutting through the points:
plt.plot(sqft, model_predictions, color="red", linewidth=2.5, linestyle="-", label="AI Model's Prediction Line")

plt.title("Machine Learning: Fitted Regression Line", fontsize=13, fontweight="bold")
plt.xlabel("Square Footage", fontsize=11)
plt.ylabel("Price (Lakhs ₹)", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
```

You can immediately see that the red line cuts through the center of the points. The visual confirms that the model has learned the underlying relationship!

---

# 8. Heatmaps: How AI Visualizes Transformer Attention

Here is the grand AI visualization reveal: **Heatmaps**.

A **Heatmap** is a 2D matrix where every number is represented by a **color intensity** (e.g. dark blue = 0.0, bright yellow = 1.0).

### In Transformer Models (ChatGPT, Claude):
When you read about the **Self-Attention Mechanism** on Day 35, you will learn that the model calculates an $N \times N$ attention matrix showing which words focus on which other words.

AI researchers plot these attention matrices as **Heatmaps** using Matplotlib's `plt.imshow()`:

```python
words = ["The", "animal", "didn't", "cross", "street", "it", "tired"]

# Simulated Self-Attention Matrix (7 words x 7 words):
# Notice row 5 ("it"): High attention on "animal" (0.85) and "tired" (0.75)!
attention_matrix = np.array([
    [0.60, 0.10, 0.05, 0.05, 0.10, 0.05, 0.05], # The
    [0.10, 0.70, 0.05, 0.05, 0.05, 0.02, 0.03], # animal
    [0.05, 0.05, 0.80, 0.05, 0.02, 0.01, 0.02], # didn't
    [0.05, 0.05, 0.10, 0.65, 0.10, 0.02, 0.03], # cross
    [0.10, 0.05, 0.02, 0.10, 0.70, 0.01, 0.02], # street
    [0.02, 0.85, 0.01, 0.02, 0.05, 0.30, 0.75], # 'it' -> Attends to 'animal' & 'tired'!
    [0.05, 0.70, 0.02, 0.01, 0.02, 0.10, 0.60]  # tired
])

plt.figure(figsize=(7, 6))

# Display the 2D matrix as an image with 'viridis' color gradient:
heatmap = plt.imshow(attention_matrix, cmap="viridis", interpolation="nearest")

# Add a color bar indicating values from 0.0 to 1.0:
plt.colorbar(heatmap, label="Attention Weight (0.0 to 1.0)")

# Label the ticks with the actual sentence words:
plt.xticks(ticks=range(len(words)), labels=words, rotation=45, fontsize=10)
plt.yticks(ticks=range(len(words)), labels=words, fontsize=10)

plt.title("Transformer Self-Attention Heatmap", fontsize=13, fontweight="bold")
plt.xlabel("Key Words (Being Looked At)")
plt.ylabel("Query Words (Looking)")
plt.tight_layout()
```

Look at row `"it"`: The brightest glowing cells appear under `"animal"` ($0.85$) and `"tired"` ($0.75$). 

The heatmap instantly makes the internal thoughts of a Large Language Model visible to the human eye!

---

# 9. Phase 2 Milestone Reflection: The Data Science Toolkit is Complete!

Take a moment to look at your toolkit:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   YOUR AI DATA SCIENCE WORKBENCH                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  🔢 NUMPY (Day 09)      -> Hardware-accelerated vectors & matrices     │
│  🐼 PANDAS (Day 10)     -> Loading, cleaning & encoding datasets       │
│  📊 MATPLOTLIB (Day 11) -> Plotting loss curves, fits & heatmaps       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

You now possess the exact same data science foundation used by AI researchers at OpenAI, Google DeepMind, and Meta!

Starting tomorrow in **Phase 3 (Classical Machine Learning)**, we will build our first real predictive AI models:
- **Linear Regression** (Predicting continuous numbers)
- **Logistic Regression** (Predicting categories and spam)
- **Decision Trees & Random Forests** (Flowchart AI)
- **Model Evaluation** (Precision, Recall, F1, Overfitting)

---

# 10. Key Takeaways & Cheat Sheet

```
┌────────────────────────────────────────────────────────────────────────┐
│                        DAY 11 CHEAT SHEET                              │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  📊 4 CORE PLOTS IN AI:                                                │
│     • plt.plot(x, y)          -> Line plot (Loss curves over epochs)   │
│     • plt.scatter(x, y)       -> Scatter plot (Feature relationships)  │
│     • plt.hist(data, bins)    -> Histogram (Checking Bell Curves)      │
│     • plt.bar(cats, values)   -> Bar chart (Model comparison)          │
│                                                                        │
│  🎨 ANATOMY OF A CLEAN PLOT:                                           │
│     • plt.title("...")        -> Informative header                    │
│     • plt.xlabel("..."), ylabel("...") -> Clear unit labels            │
│     • plt.grid(True)          -> Visual guidance                       │
│     • plt.legend()            -> Explains multi-line colors            │
│     • plt.savefig("name.png") -> Save image to disk                    │
│                                                                        │
│  🪟 SUBPLOTS:                                                          │
│     • fig, axes = plt.subplots(rows, cols, figsize=(w, h))             │
│     • axes[0].plot(...), axes[1].plot(...)                             │
│     • plt.tight_layout() prevents overlapping labels!                  │
│                                                                        │
│  🔥 HEATMAPS (plt.imshow):                                             │
│     • Visualizes 2D matrices as color maps (cmap='viridis').           │
│     • Used to inspect Transformer Attention Weights!                   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

# 11. Practice Exercises with Solutions

<details>
<summary><b>🏋️ Exercise 1: Plotting Training vs. Validation Loss (Detect Overfitting)</b></summary>
<br/>

In machine learning, when Training Loss keeps decreasing but Validation Loss starts increasing, the model is **Overfitting** (memorizing the training data).

Given:
```python
epochs = list(range(1, 11))
train_loss = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
val_loss   = [1.1, 0.8, 0.6, 0.50, 0.48, 0.52, 0.60, 0.72, 0.85, 1.05]
```

Write Matplotlib code to:
1. Plot both curves on the same figure with distinct colors and markers.
2. Label both curves in a legend.
3. Mark epoch 5 with a vertical dashed line (`plt.axvline(x=5)`) indicating the optimal stopping point!

**Solution:**
```python
import matplotlib.pyplot as plt

epochs = list(range(1, 11))
train_loss = [1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03]
val_loss   = [1.1, 0.8, 0.6, 0.50, 0.48, 0.52, 0.60, 0.72, 0.85, 1.05]

plt.figure(figsize=(8, 5))

plt.plot(epochs, train_loss, color="blue", marker="o", label="Training Loss")
plt.plot(epochs, val_loss, color="red", marker="s", label="Validation Loss (Unseen)")

# Highlight optimal stopping point:
plt.axvline(x=5, color="green", linestyle="--", linewidth=2, label="Optimal Stopping Point (Epoch 5)")

plt.title("Detecting Overfitting: Train vs. Validation Loss", fontsize=13, fontweight="bold")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
```
</details>

<details>
<summary><b>🏋️ Exercise 2: Comparing Model Accuracies with a Bar Chart</b></summary>
<br/>

Suppose you tested 4 machine learning models on a spam detection task:
- Models: `["Logistic Regression", "Decision Tree", "Random Forest", "Neural Net"]`
- Accuracy: `[0.86, 0.81, 0.94, 0.97]`

Write code to plot a bar chart comparing the models, with accuracy displayed as percentages (e.g. 86%, 94%). Set the y-axis limits between 0.7 and 1.0 (`plt.ylim(0.7, 1.0)`).

**Solution:**
```python
models = ["Logistic Regression", "Decision Tree", "Random Forest", "Neural Net"]
accuracies = [0.86, 0.81, 0.94, 0.97]
colors = ["skyblue", "salmon", "lightgreen", "gold"]

plt.figure(figsize=(8, 4.5))

bars = plt.bar(models, accuracies, color=colors, edgecolor="black", width=0.6)

plt.title("Spam Classifier Accuracy Comparison", fontsize=13, fontweight="bold")
plt.ylabel("Accuracy Score")
plt.ylim(0.75, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Print percentage labels on top of each bar:
for bar, acc in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + 0.15, yval + 0.008, f"{acc * 100:.1f}%", fontweight="bold")
```
</details>

<details>
<summary><b>🏋️ Exercise 3: Plotting a Mini Attention Heatmap</b></summary>
<br/>

Create a $3 \times 3$ attention matrix representing the sentence `["AI", "is", "cool"]`:
```python
import numpy as np
import matplotlib.pyplot as plt

tokens = ["AI", "is", "cool"]
matrix = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.8, 0.1],
    [0.4, 0.1, 0.5]
])
```

Write code using `plt.imshow()` to render the matrix as a heatmap with annotations displaying the exact number inside each cell!

**Solution:**
```python
import numpy as np
import matplotlib.pyplot as plt

tokens = ["AI", "is", "cool"]
matrix = np.array([
    [0.7, 0.1, 0.2],
    [0.1, 0.8, 0.1],
    [0.4, 0.1, 0.5]
])

fig, ax = plt.subplots(figsize=(5, 5))
heatmap = ax.imshow(matrix, cmap="Blues")

# Set tick labels
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels(tokens, fontsize=12)
ax.set_yticklabels(tokens, fontsize=12)

# Annotate each cell with its number
for i in range(3):
    for j in range(3):
        color = "white" if matrix[i, j] > 0.5 else "black"
        ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=color, fontweight="bold")

plt.title("3x3 Attention Matrix Heatmap", fontweight="bold")
plt.colorbar(heatmap, shrink=0.8)
```
</details>

---

## ⏭️ What's Next: Phase 3 — Classical Machine Learning (Day 12)

You have completed **Phase 2: Python Data Science Toolkit**! 🎓

Tomorrow on **Day 12**, we officially enter **Phase 3: Classical Machine Learning**:
- What is Machine Learning? The paradigm shift from traditional rules to learned models.
- **Supervised Learning** vs. **Unsupervised Learning** vs. **Reinforcement Learning**.
- Features, labels, and the essential **Train/Test Split**!

---

<p align="center">
  <b>🏆 CONGRATULATIONS! PHASE 2 (PYTHON DATA SCIENCE TOOLKIT) IS 100% COMPLETE! 🏆</b>
</p>
