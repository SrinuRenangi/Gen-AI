# Day 28: RNNs — Giving Neural Networks Memory for Sequences

> **"A feedforward network reads every input with total amnesia. A Recurrent Neural Network carries a rolling mental tape of the past, allowing machines to read sentences, hear audio waves, and predict the future."**  
> Welcome to Day 28! Up to this point, our models treated every sample as an isolated, independent event. Today, we conquer temporal dependencies using **Recurrent Neural Networks (RNNs)**.

---

## 🧭 The Mental Compass: The River Bank vs The Wall Street Bank

Look at these two English sentences:

```
    Sentence A: "The tired fisherman sat quietly by the muddy river BANK."
    Sentence B: "The masked robber sprinted out of the downtown commercial BANK."
```

If you look strictly at the final word **"BANK"** in isolation, it is impossible to know what it means.  
* In Sentence A, it is a geological strip of soil.
* In Sentence B, it is a fortified financial vault.

How does your human brain instantly disambiguate them? **Because as you read each word, you maintain an ongoing mental summary of what came before.** When you reach word 9, your internal memory state has already accumulated context from words 1 through 8!

A **Recurrent Neural Network (RNN)** is a neural network with a **memory tape (the Hidden State $h_t$)** that updates at every time step.

---

## 1. The Recurrent Cell & The Unrolling Concept

An RNN can be visualized in two equivalent ways: as a **compact recurring loop** or **unrolled across time**:

![Recurrent Neural Networks (RNN): Unrolling Through Time](assets/rnn_unrolled_through_time.svg)

### The Recurrent Mathematics:
At each time step $t$, the cell receives two inputs:
1. The **current input vector** $\mathbf{x}_t$ (e.g., the current word or character).
2. The **previous hidden state** $\mathbf{h}_{t-1}$ (the memory from all past time steps).

$$\mathbf{h}_t = \tanh\left(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h\right)$$
$$\mathbf{\hat{y}}_t = \text{Softmax}\left(\mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y\right)$$

### The 3 Core Weight Matrices:
* $\mathbf{W}_{xh}$ (Input-to-Hidden): How much new information from $\mathbf{x}_t$ enters the memory.
* $\mathbf{W}_{hh}$ (Hidden-to-Hidden): How much past memory from $\mathbf{h}_{t-1}$ is preserved.
* $\mathbf{W}_{hy}$ (Hidden-to-Output): Translates current memory state into predictions.

> [!NOTE]
> **Weight Sharing Across Time:**  
> The exact same matrices $\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$ are reused at **every single time step**! Whether a sequence has 5 words or 500 words, the parameter count never changes.

---

## 2. Hand-Calculated Step-by-Step Arithmetic Trace

Let's trace an RNN over two consecutive time steps by hand on a scalar toy model:
* **Weights:** $w_{xh} = 0.5, \quad w_{hh} = 0.8, \quad b_h = 0.0$
* **Initial Memory:** $h_0 = 0.0$
* **Input Sequence:** $x_1 = 1.0 \quad \text{followed by} \quad x_2 = 2.0$

### Time Step $t = 1$:
$$z_1 = (w_{xh} \cdot x_1) + (w_{hh} \cdot h_0) + b_h = (0.5 \cdot 1.0) + (0.8 \cdot 0.0) + 0.0 = \mathbf{0.50}$$
$$h_1 = \tanh(0.50) \approx \mathbf{0.4621}$$
*(The network's memory now holds $0.4621$)*

---

### Time Step $t = 2$:
Now the second input $x_2 = 2.0$ arrives, but it is combined with the memory $h_1 = 0.4621$:
$$z_2 = (w_{xh} \cdot x_2) + (w_{hh} \cdot h_1) + b_h = (0.5 \cdot 2.0) + (0.8 \cdot 0.4621) + 0.0$$
$$z_2 = 1.0 + 0.3697 = \mathbf{1.3697}$$
$$h_2 = \tanh(1.3697) \approx \mathbf{0.8784}$$

Notice how $h_2$ carries information from **both $x_2$ AND $x_1$**!

---

## 3. Backpropagation Through Time (BPTT) & The Amnesia Horizon

How do RNNs learn? We unroll the entire sequence over $T$ steps and apply backpropagation in reverse across time — an algorithm called **Backpropagation Through Time (BPTT)**.

![BPTT & The Amnesia Horizon](assets/bptt_vanishing_memory_horizon.svg)

### The Mathematical Cause of Vanishing Memory:
To calculate how a mistake at time step $T$ should update the weights at step $1$, the chain rule must multiply hidden-to-hidden derivatives across all intermediate time steps:

$$\frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \cdot \prod_{k=2}^{T} \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \frac{\partial \mathcal{L}_T}{\partial \mathbf{h}_T} \cdot \prod_{k=2}^{T} \Big(\mathbf{W}_{hh}^T \cdot \text{diag}(1 - \mathbf{h}_k^2)\Big)$$

Look at what happens when you multiply $\mathbf{W}_{hh}$ twenty times:
* If the largest eigenvalue of $\mathbf{W}_{hh} < 1$: The product decays exponentially to **exact zero** ($\mathbf{0.00000}$).
  * **Consequence (The Amnesia Horizon):** The model cannot remember context beyond 5 to 10 words! In the sentence *"The clouds that floated over the vast open desert valley were dark"*, the model forgets *"clouds"* and guesses randomly.
* If the largest eigenvalue of $\mathbf{W}_{hh} > 1$: The product explodes exponentially to **$\pm \infty$**.
  * **Consequence (Exploding Gradients):** The loss suddenly becomes `NaN` and training crashes!

---

## 4. The Engineering Remedy: Gradient Clipping

While the vanishing gradient required a complete architectural redesign (LSTMs, tomorrow in Day 29), exploding gradients can be solved with a simple engineering safeguard: **Gradient Clipping**.

If the $L_2$ norm of the gradient vector $\|\mathbf{g}\|$ exceeds a predefined maximum threshold $c$ (typically $c = 1.0$ or $5.0$), scale it down proportionally:

$$\mathbf{g} \leftarrow \mathbf{g} \cdot \frac{c}{\max(c, \|\mathbf{g}\|)}$$

```python
# The PyTorch life-saver line in every recurrent training loop:
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## 5. Hands-On Python Lab: Character-Level Sequence Prediction in PyTorch

Let's build an RNN that learns to predict the next character in a sequence:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# =====================================================================
# 1. DATASET PREPARATION: LEARNING THE SEQUENCE "antigravity"
# =====================================================================
text = "antigravity"
unique_chars = sorted(list(set(text)))
vocab_size = len(unique_chars)

char_to_idx = {ch: i for i, ch in enumerate(unique_chars)}
idx_to_char = {i: ch for i, ch in enumerate(unique_chars)}

print("=" * 65)
print(f"Text: '{text}' | Vocabulary Size: {vocab_size} characters")
print(f"Char Map: {char_to_idx}")
print("=" * 65)

# Input: "antigravit" -> Target: "ntigravity"
input_indices = [char_to_idx[c] for c in text[:-1]]
target_indices = [char_to_idx[c] for c in text[1:]]

# Convert to one-hot tensors: (Seq_Len, Batch, Vocab_Size)
seq_len = len(input_indices)
X_onehot = torch.zeros(seq_len, 1, vocab_size)
for t, idx in enumerate(input_indices):
    X_onehot[t, 0, idx] = 1.0

y_tensor = torch.tensor(target_indices, dtype=torch.long)

# =====================================================================
# 2. CHAR-RNN MODEL ARCHITECTURE
# =====================================================================
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        # The PyTorch recurrent cell
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h_prev):
        # out: (Seq_Len, Batch, Hidden_Size), h_new: (1, Batch, Hidden_Size)
        out, h_new = self.rnn(x, h_prev)
        logits = self.fc(out.squeeze(1))  # (Seq_Len, Output_Size)
        return logits, h_new

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# =====================================================================
# 3. TRAINING LOOP WITH GRADIENT CLIPPING
# =====================================================================
model = CharRNN(input_size=vocab_size, hidden_size=16, output_size=vocab_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.02)

epochs = 150
print(f"\nTraining Char-RNN over {epochs} epochs...")

for epoch in range(1, epochs + 1):
    h_state = model.init_hidden()
    optimizer.zero_grad()
    
    # Forward pass through all time steps
    logits, h_state = model(X_onehot, h_state)
    loss = criterion(logits, y_tensor)
    
    # Backward pass (BPTT)
    loss.backward()
    
    # Safeguard against exploding gradients!
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    if epoch % 30 == 0 or epoch == 1:
        # Generate prediction
        predicted_idx = torch.argmax(logits, dim=1).tolist()
        predicted_word = "".join([idx_to_char[i] for i in predicted_idx])
        print(f"  Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f} | Output: '{text[0] + predicted_word}'")

print("\n" + "=" * 65)
print("🎉 Model successfully memorized sequence order using recurrent states!")
print("=" * 65)
```

---

## 6. Summary Checklist for Day 28

1. [x] **The Need for Memory:** Feedforward networks suffer from temporal amnesia. RNNs maintain an ongoing hidden state $\mathbf{h}_t$.
2. [x] **The Recurrent Equation:** $\mathbf{h}_t = \tanh(\mathbf{W}_{xh} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b}_h)$.
3. [x] **Weight Sharing in Time:** The exact same weight matrices are reused at every sequence position $t$.
4. [x] **Backpropagation Through Time (BPTT):** Derivatives are propagated backward across the entire unrolled time chain.
5. [x] **The Amnesia Curse:** Exponential decay of chained $\mathbf{W}_{hh}$ derivatives wipes out memory beyond 5–10 steps.
6. [x] **Gradient Clipping:** Normalizes runaway gradients to prevent `NaN` crashes.

---

*Tomorrow in **Day 29**, we examine the legendary solution to the Amnesia Horizon: **LSTMs & GRUs — Solving the Forgetting Problem with Gated Memory Cells!***
