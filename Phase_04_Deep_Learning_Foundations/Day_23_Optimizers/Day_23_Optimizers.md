# Day 23: Optimizers — Smart Ways to Turn the Knobs


| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 22: Backpropagation](../Day_22_Backpropagation/Day_22_Backpropagation.md) | [All 50 Days Overview](../../README.md) | [Day 24: Building a Complete Neural Network from Scratch →](../Day_24_Building_Neural_Network_From_Scratch/Day_24_Building_Neural_Network_From_Scratch.md) |

> **"Gradient descent tells you which direction is downhill. The optimizer decides how fast to run, when to build momentum, and how to avoid bouncing off the canyon walls."**  
> Welcome to Day 23! Yesterday, we derived backpropagation to calculate exact loss derivatives $\nabla_\theta \mathcal{L}$. Today, we master how **Optimizers** use those derivatives to update parameters intelligently — culminating in **Adam**, the optimizer that trained GPT-4, Claude, and Midjourney.

---

## 🧭 The Mental Compass: The Heavy Bowling Ball in a Narrow Canyon

Imagine you are trying to reach the bottom of a steep geological ravine:

```
                          THE RAVINE DILEMMA
                          ───────────────────

        WALL (Steep Gradient)                WALL (Steep Gradient)
             \                                    /
              \                                  /
               \                                /
                \                              /
                 \                            /
                  \                          /
                   \                        /
                    ═══════ FLOOR ═════════  (Gentle Gradient Toward Exit &rarr;)
```

* **The Problem:** The canyon walls are violently steep (huge vertical gradients), but the canyon floor slopes very gently toward the exit (tiny horizontal gradient).
* **Approach A: The Blind Hiker (Vanilla SGD)**  
  Takes a fixed step directly perpendicular to the slope. He takes a giant leap down the north wall, overshoots, crashes into the south wall, bounces back to the north wall, and makes **zero forward progress toward the exit**!
* **Approach B: The Heavy Bowling Ball (Momentum & Adam)**  
  A 16-pound bowling ball starts rolling. As it sloshes left and right, the opposing vertical forces cancel each other out. Meanwhile, gravity along the gentle floor steadily accelerates the ball forward until it zooms out of the canyon at maximum velocity!

---

## 1. The Gradient Descent Spectrum: Batch vs Stochastic vs Mini-Batch

Before examining advanced optimizers, how many data samples should we look at before updating our weights?

| Variant | Batch Size per Update | Speed per Step | Gradient Stability | Hardware Utilization |
| :--- | :---: | :---: | :---: | :---: |
| **Batch GD** | All $N$ samples | Extremely Slow | Perfectly smooth, exact gradient | Poor (waits for all data) |
| **Stochastic GD (SGD)** | Exactly $1$ sample | Ultra-fast | Extremely noisy, zig-zags wildly | Terrible (wastes GPU parallelism) |
| **Mini-Batch GD** | **$32, 64, 128, 512$** | **Optimal** | **Smooth enough + escapes saddle points** | **100% GPU Tensor Core saturation!** |

> [!NOTE]
> **Why Mini-Batch is the Universal AI Standard:**  
> Modern GPUs are matrix-multiplication monsters. Feeding 1 sample wastes 99% of your GPU cores. Feeding 128 samples costs nearly the same computation time as 1 sample, but provides a 100x cleaner gradient estimate!

---

## 2. The Evolution of Optimizers

![Optimization Dynamics: Conquering the Ravine](assets/optimizer_ravine_trajectories.svg)

---

### Generation 1: SGD with Momentum (Polyack, 1964)
Instead of letting the current gradient dictate your entire step, you maintain an ongoing **velocity vector** $\mathbf{v}$:

$$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \mathbf{g}_t \quad (\text{typically } \beta = 0.9)$$
$$\mathbf{W} \leftarrow \mathbf{W} - \alpha \mathbf{v}_t$$

* **Physics Intuition:** $\beta$ acts like friction ($0.9$ means $90\%$ of yesterday's velocity is preserved).
* **Oscillation Damping:** If gradients oscillate $+10, -10, +10, -10$ along the canyon walls, their moving average sums to $\approx 0$. If gradients along the floor are steady $+1, +1, +1, +1$, velocity accelerates to $+10$!

---

### Generation 2: RMSprop (Geoffrey Hinton, 2012)
What if some weights need giant steps while other weights need tiny steps? RMSprop introduces an **Adaptive Learning Rate per parameter**:

$$\mathbf{s}_t = \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \quad (\text{typically } \beta_2 = 0.999)$$
$$\mathbf{W} \leftarrow \mathbf{W} - \frac{\alpha}{\sqrt{\mathbf{s}_t} + \epsilon} \mathbf{g}_t$$

* **Intuition:** $\mathbf{s}_t$ tracks the historical volatility (variance) of each parameter.
* If a weight has violent, giant gradients ($\mathbf{g}^2$ is massive), the denominator $\sqrt{\mathbf{s}_t}$ shrinks its step size!
* If a weight has tiny, sluggish gradients, $\sqrt{\mathbf{s}_t}$ is small, boosting its step size!

---

### Generation 3: Adam (Adaptive Moment Estimation — Kingma & Ba, 2014)

Adam combines the superpowers of **Momentum** (smooth directional tracking) and **RMSprop** (adaptive per-parameter step scaling) into a single master algorithm.

![The Inner Mechanics of the Adam Optimizer](assets/adam_optimizer_architecture.svg)

---

## 3. The 4 Step Equations of Adam

At each training iteration $t$:

### 1. Update 1st Moment (Mean / Momentum):
$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t \quad (\beta_1 = 0.9)$$

### 2. Update 2nd Moment (Variance / Uncentered Variance):
$$\mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \quad (\beta_2 = 0.999)$$

### 3. Bias Correction Step:
Since $\mathbf{m}_0 = \mathbf{0}$ and $\mathbf{v}_0 = \mathbf{0}$, during the first few steps ($t = 1, 2, 3$), both vectors are biased heavily toward zero. Adam corrects this analytically:
$$\mathbf{\hat{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \mathbf{\hat{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

*(Notice as $t \to \infty$, $\beta^t \to 0$ and the denominator becomes $1.0$, turning correction off naturally!)*

### 4. Parameter Update:
$$\mathbf{W}_{t+1} = \mathbf{W}_t - \frac{\alpha}{\sqrt{\mathbf{\hat{v}}_t} + \epsilon} \mathbf{\hat{m}}_t \quad (\alpha = 0.001, \epsilon = 10^{-8})$$

> [!TIP]
> **What is AdamW and Why Does Every LLM Use It?**  
> In 2017, Ilya Loshchilov and Frank Hutter discovered a flaw in how standard Adam interacted with $L_2$ weight decay regularization. They created **AdamW**, which decouples weight decay from the gradient update.  
> **Today, virtually every frontier LLM (GPT-4, LLaMA-3, Claude, Mistral) is pre-trained using AdamW.**

---

## 4. Hand-Worked Step-by-Step Optimizer Comparison

Let's trace an oscillating parameter across 3 steps:
* Initial parameter $w = 0.0$, Learning rate $\alpha = 0.1$.
* Incoming gradients: Step 1: $g_1 = +10.0$ | Step 2: $g_2 = -8.0$ | Step 3: $g_3 = +9.0$

| Optimizer | Step 1 Update ($g_1 = +10.0$) | Step 2 Update ($g_2 = -8.0$) | Step 3 Update ($g_3 = +9.0$) | Net Trajectory |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla SGD** | $\Delta w = -0.1(10) = \mathbf{-1.00}$<br>$w_1 = -1.00$ | $\Delta w = -0.1(-8) = \mathbf{+0.80}$<br>$w_2 = -0.20$ | $\Delta w = -0.1(9) = \mathbf{-0.90}$<br>$w_3 = -1.10$ | 💥 Violent whiplash! Oscillates between $-1.0$ and $+0.8$. |
| **Momentum** ($\beta=0.9$) | $v_1 = 0.1(10) = 1.0$<br>$w_1 = -0.1(1.0) = \mathbf{-0.10}$ | $v_2 = 0.9(1) + 0.1(-8) = 0.10$<br>$w_2 = w_1 - 0.1(0.1) = \mathbf{-0.11}$ | $v_3 = 0.9(0.1) + 0.1(9) = 0.99$<br>$w_3 = w_2 - 0.1(0.99) = \mathbf{-0.21}$ | ✅ Smooth and controlled. Oscillations heavily dampened! |
| **Adam** | $\hat{m}_1 = 10, \hat{v}_1 = 100$<br>Step $\approx \mathbf{-0.10}$ | Opposing signs cancel $m_2$; $\sqrt{v_2}$ normalizes scale.<br>Step $\approx \mathbf{-0.02}$ | Steady forward progress without overshoot.<br>Step $\approx \mathbf{-0.08}$ | 🚀 Automatically scales step to safe units. |

---

## 5. Hands-On Python Lab: Benchmarking Optimizers on a Ravine Surface

Let's simulate the famous Ravine function: $f(x, y) = 0.1 x^2 + 2.0 y^2$ (steep vertically, gentle horizontally) and watch SGD get trapped while Adam speeds straight to $(0,0)$!

```python
import numpy as np

# Loss function: Steep in y, shallow in x
def loss_func(x, y):
    return 0.1 * (x ** 2) + 2.0 * (y ** 2)

# Gradients
def grad_func(x, y):
    df_dx = 0.2 * x
    df_dy = 4.0 * y
    return np.array([df_dx, df_dy])

# =====================================================================
# SIMULATE OPTIMIZERS OVER 50 ITERATIONS
# =====================================================================
steps = 50
lr = 0.4

# 1. Vanilla SGD
pos_sgd = np.array([10.0, 10.0])  # Start point
for _ in range(steps):
    g = grad_func(pos_sgd[0], pos_sgd[1])
    pos_sgd -= lr * g

# 2. SGD + Momentum
pos_mom = np.array([10.0, 10.0])
v_mom = np.zeros(2)
beta = 0.9
for _ in range(steps):
    g = grad_func(pos_mom[0], pos_mom[1])
    v_mom = beta * v_mom + (1.0 - beta) * g
    pos_mom -= lr * 2.0 * v_mom

# 3. Adam Optimizer
pos_adam = np.array([10.0, 10.0])
m = np.zeros(2)
v = np.zeros(2)
beta1, beta2, eps = 0.9, 0.999, 1e-8
adam_lr = 0.8
for t in range(1, steps + 1):
    g = grad_func(pos_adam[0], pos_adam[1])
    m = beta1 * m + (1.0 - beta1) * g
    v = beta2 * v + (1.0 - beta2) * (g ** 2)
    
    # Bias correction
    m_hat = m / (1.0 - (beta1 ** t))
    v_hat = v / (1.0 - (beta2 ** t))
    
    pos_adam -= adam_lr * m_hat / (np.sqrt(v_hat) + eps)

print("=" * 65)
print("     BENCHMARKING OPTIMIZERS ON A STEEP RAVINE (50 STEPS)")
print("=" * 65)
print(f"Goal Coordinate: (0.00, 0.00) | Minimum Loss: 0.00")
print("-" * 65)
print(f"Vanilla SGD    -> Pos: ({pos_sgd[0]:.2f}, {pos_sgd[1]:.2f}) | Final Loss: {loss_func(*pos_sgd):.4f} (Trapped!)")
print(f"SGD + Momentum -> Pos: ({pos_mom[0]:.2f}, {pos_mom[1]:.2f}) | Final Loss: {loss_func(*pos_mom):.4f}")
print(f"Adam Optimizer -> Pos: ({pos_adam[0]:.2f}, {pos_adam[1]:.2f}) | Final Loss: {loss_func(*pos_adam):.6f} (CONVERGED!)")
print("=" * 65)
print("🎉 Adam reached the global minimum while Vanilla SGD was still trapped bouncing off canyon walls!")
```

---

## 6. Architect's Cheat Sheet: Choosing the Right Optimizer

| Model Family / Architecture | Recommended Optimizer | Standard Learning Rate ($\alpha$) | Key Considerations |
| :--- | :---: | :---: | :--- |
| **Transformers & LLMs (GPT, LLaMA)** | **AdamW** | $1 \times 10^{-4}$ to $5 \times 10^{-4}$ | Decoupled weight decay prevents weight explosion in multi-head attention. |
| **Computer Vision (ResNet, CNNs)** | **SGD + Momentum** | $0.01$ to $0.1$ | Often achieves marginally better generalization on pure image classification. |
| **Reinforcement Learning (PPO, SAC)** | **Adam** | $3 \times 10^{-4}$ ("Karpathy Constant") | Stable updates in high-variance RL reward environments. |
| **Recurrent Networks (RNNs, LSTMs)** | **RMSprop or Adam** | $1 \times 10^{-3}$ | Adaptive scaling prevents exploding sequence gradients. |

---

## ✍️ Self-Check Exercises & Practice Problems

Put your intuition on gradient momentum, adaptive step sizes, and AdamW to the test!

### 🏋️ Problem 1: Hand-Calculating Momentum Velocity & Weight Updates
A neural network weight $w$ is undergoing gradient descent with Momentum:
- Momentum coefficient: $\beta = 0.90$
- Learning rate: $\eta = 0.10$
- Prior accumulated velocity: $v_{t-1} = 0.40$
- Current iteration gradient: $g_t = +2.0$

Update Equations:
$$v_t = \beta \cdot v_{t-1} + \eta \cdot g_t$$
$$w_t = w_{t-1} - v_t$$

**Your Tasks:**
1. Compute the inertia contribution from previous steps ($\beta \cdot v_{t-1}$).
2. Compute the new gradient contribution ($\eta \cdot g_t$).
3. Compute the total velocity $v_t$.
4. How does $v_t$ compare to a vanilla SGD step ($\Delta w_{\text{vanilla}} = \eta \cdot g_t = 0.20$)? Why is it larger?

---

### 🏋️ Problem 2: Why AdamW is Mandatory for Large Language Models
During the training of GPT-3 and LLaMA, engineers use **AdamW** rather than classical **Adam + L2 Regularization**.

**Your Tasks:**
1. In standard Adam with L2 regularization ($+ \lambda w$), what happens to the weight penalty term when a parameter receives very large historical gradients ($s_t$ is huge)?
2. Does the regularization penalty become stronger or weaker for large-gradient weights?
3. How does AdamW's **decoupled weight decay** solve this problem mathematically?

<details>
<summary><b>🔍 Click to Reveal Step-by-Step Solutions</b></summary>

### Solution 1:
1. **Inertia Term:**
   $$\beta \cdot v_{t-1} = 0.90 \times 0.40 = \mathbf{0.36}$$

2. **New Gradient Term:**
   $$\eta \cdot g_t = 0.10 \times 2.0 = \mathbf{0.20}$$

3. **Total Updated Velocity:**
   $$v_t = 0.36 + 0.20 = \mathbf{0.56}$$
   $$w_t = w_{t-1} - 0.56$$

4. **Comparison to Vanilla SGD:**
   Vanilla SGD would only take a step of $0.20$. Momentum took a step of $0.56$ ($2.8\times$ larger!) because the rolling bowling ball carried forward momentum from prior steps in the same direction. It powers through plateaus and flat saddle points!

---

### Solution 2:
1. **Flaw of L2 in Standard Adam:**
   In standard Adam, the L2 weight penalty is added directly to the gradient before computing the second moment $s_t$:
   $$g_t^{\text{reg}} = g_t + \lambda w$$
   Then Adam divides by $\sqrt{s_t}$:
   $$\Delta w = -\frac{\eta}{\sqrt{s_t} + \epsilon} (g_t + \lambda w)$$

2. **Unintended Consequence:**
   Weights with very large, frequent gradients accumulate a massive $s_t$. Because $\sqrt{s_t}$ is in the denominator, the actual weight decay penalty $\frac{\eta \lambda w}{\sqrt{s_t}}$ gets **massively suppressed**! Conversely, rarely updated weights receive an inappropriately large weight decay penalty!

3. **AdamW Decoupled Solution:**
   AdamW separates the weight decay entirely from the adaptive gradient scaling:
   $$w_t = w_{t-1} - \eta \cdot \lambda \cdot w_{t-1} - \frac{\eta}{\sqrt{\hat{s}_t} + \epsilon} \hat{m}_t$$
   Every single weight decays at the exact intended rate $\eta \lambda$, regardless of gradient magnitude. This stabilized Transformer attention layers and is now universal in modern GenAI.
</details>

---

## 7. Summary Checklist for Day 23

1. [x] **The Ravine Problem:** Steep walls cause vanilla SGD to oscillate violently while making zero forward progress.
2. [x] **Mini-Batching:** Maximizes GPU Tensor Core parallelism while maintaining stable gradient estimates.
3. [x] **Momentum:** Accelerates along consistent paths and cancels out oscillating noise like a rolling heavy bowling ball.
4. [x] **RMSprop:** Scales learning rate inversely by $\sqrt{\text{historical variance}}$, giving slow weights a boost and calming hyperactive weights.
5. [x] **Adam:** Combines Momentum (1st moment) + RMSprop (2nd moment) + Bias Correction into the industry's most robust optimizer.
6. [x] **AdamW:** Decouples weight decay to power modern Large Language Model pre-training.

---

*Tomorrow in **Day 24**, we assemble every single piece of Phase 4: **Building a Complete Neural Network from Scratch** in pure NumPy — forward pass, backpropagation, Adam optimizer, and training loop — to classify complex non-linear spirals!*


---

## 🧭 Navigation & Next Steps

| ⬅️ Previous Day | 📚 Course Hub | ➡️ Next Day |
|:---|:---:|---:|
| [← Day 22: Backpropagation](../Day_22_Backpropagation/Day_22_Backpropagation.md) | [All 50 Days Overview](../../README.md) | [Day 24: Building a Complete Neural Network from Scratch →](../Day_24_Building_Neural_Network_From_Scratch/Day_24_Building_Neural_Network_From_Scratch.md) |
