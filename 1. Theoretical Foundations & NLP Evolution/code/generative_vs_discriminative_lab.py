"""
=============================================================================
Hands-On Lab: Generative vs Discriminative Models in Python
=============================================================================
Course: Zero to Hero Gen AI — Module 01: Theoretical Foundations & NLP Evolution
Topic: Distinguishing Generative versus Discriminative Models

This lab proves the mathematical asymmetry:
1. Discriminative (Logistic Regression): Directly models P(Y|X) boundary.
   - Accurately classifies points.
   - CANNOT synthesize new data points.
2. Generative (Gaussian Mixture / Naive Bayes): Models joint distribution P(X, Y) = P(Y)P(X|Y).
   - Accurately classifies points via Bayes' Rule: P(Y|X) = P(X|Y)P(Y) / P(X).
   - CAN synthesize novel, realistic data samples from scratch!
=============================================================================
"""

import numpy as np


def main():
    np.random.seed(42)

    # =====================================================================
    # STEP 1: Synthesize a 2D Ground-Truth Dataset (2 Classes: Red & Blue)
    # =====================================================================
    print("=" * 70)
    print("STEP 1: Synthesizing 2D Ground-Truth Training Data")
    print("=" * 70)

    n_samples = 200

    # Class 0 (Red Cluster): Mean [-2.0, -2.0]
    true_mean_0 = np.array([-2.0, -2.0])
    true_cov_0 = np.array([[1.0, 0.4], [0.4, 1.0]])
    X_0 = np.random.multivariate_normal(true_mean_0, true_cov_0, n_samples)
    y_0 = np.zeros(n_samples, dtype=int)

    # Class 1 (Blue Cluster): Mean [2.0, 2.0]
    true_mean_1 = np.array([2.0, 2.0])
    true_cov_1 = np.array([[1.2, -0.3], [-0.3, 0.8]])
    X_1 = np.random.multivariate_normal(true_mean_1, true_cov_1, n_samples)
    y_1 = np.ones(n_samples, dtype=int)

    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    print(f"Total training data: {len(X)} samples in 2D space.")
    print(f"  Class 0 (Red):  {len(y_0)} points centered at {true_mean_0}")
    print(f"  Class 1 (Blue): {len(y_1)} points centered at {true_mean_1}")

    # =====================================================================
    # STEP 2: Train Discriminative Model (Logistic Regression from Scratch)
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Training Discriminative Model (Logistic Regression)")
    print("=" * 70)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    # Augment features with bias term
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    weights = np.zeros(X_aug.shape[1])
    learning_rate = 0.1
    epochs = 300

    for _ in range(epochs):
        preds = sigmoid(X_aug @ weights)
        gradient = (X_aug.T @ (preds - y)) / len(y)
        weights -= learning_rate * gradient

    b, w1, w2 = weights
    print(f"Optimized Discriminative Parameters (Boundary Line):")
    print(f"  Bias (w0) = {b:.4f}")
    print(f"  Weight X1 = {w1:.4f}")
    print(f"  Weight X2 = {w2:.4f}")
    print(f"  Decision Boundary: {w1:.3f} * X1 + {w2:.3f} * X2 + {b:.3f} = 0")

    # Evaluate an unseen test point
    test_pt = np.array([1.5, 1.8])
    test_pt_aug = np.array([1.0, test_pt[0], test_pt[1]])
    prob_class_1_disc = sigmoid(test_pt_aug @ weights)

    print(f"\nClassification of Unseen Query Point {test_pt}:")
    print(f"  P(Y=1 | X={test_pt}) = {prob_class_1_disc:.4f}")
    print(f"  Verdict: Class {'1 (Blue)' if prob_class_1_disc >= 0.5 else '0 (Red)'}")

    print("\nCan this Discriminative Model generate a new sample for Class 0?")
    print("  ❌ IMPOSSIBLE! It only knows where the boundary lies, not the data distribution.")

    # =====================================================================
    # STEP 3: Train Generative Model (Class-Conditional Gaussian Estimator)
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Training Generative Model (Estimating P(Y) and P(X|Y))")
    print("=" * 70)

    # Class priors P(Y)
    prior_0 = np.mean(y == 0)
    prior_1 = np.mean(y == 1)

    # Class conditional parameters: means and covariances
    learned_mu_0 = np.mean(X[y == 0], axis=0)
    learned_mu_1 = np.mean(X[y == 1], axis=0)

    learned_cov_0 = np.cov(X[y == 0], rowvar=False)
    learned_cov_1 = np.cov(X[y == 1], rowvar=False)

    print(f"Learned Generative Parameters:")
    print(f"  Class 0 Prior P(Y=0): {prior_0:.2f} | Learned Mean: {np.round(learned_mu_0, 2)}")
    print(f"  Class 1 Prior P(Y=1): {prior_1:.2f} | Learned Mean: {np.round(learned_mu_1, 2)}")

    # =====================================================================
    # STEP 4: Classify Unseen Query via Bayes' Rule
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Classifying with Generative Bayes' Inversion")
    print("=" * 70)

    def calc_gaussian_density(x, mean, cov):
        d = len(x)
        diff = x - mean
        inv_cov = np.linalg.inv(cov)
        exponent = -0.5 * (diff.T @ inv_cov @ diff)
        norm = 1.0 / (np.sqrt(((2 * np.pi) ** d) * np.linalg.det(cov)))
        return norm * np.exp(exponent)

    # Likelihoods P(X | Y)
    likelihood_0 = calc_gaussian_density(test_pt, learned_mu_0, learned_cov_0)
    likelihood_1 = calc_gaussian_density(test_pt, learned_mu_1, learned_cov_1)

    # Joint probabilities P(X, Y) = P(X|Y) * P(Y)
    joint_0 = likelihood_0 * prior_0
    joint_1 = likelihood_1 * prior_1

    # Posterior P(Y=1 | X) via Bayes' Theorem
    posterior_1_gen = joint_1 / (joint_0 + joint_1)

    print(f"Generative Bayes Inversion on Query Point {test_pt}:")
    print(f"  P(X | Class 0) = {likelihood_0:.6f}")
    print(f"  P(X | Class 1) = {likelihood_1:.6f}")
    print(f"  Posterior P(Y=1 | X) = {posterior_1_gen:.4f}")
    print(f"  Verdict: Class {'1 (Blue)' if posterior_1_gen >= 0.5 else '0 (Red)'}")

    # =====================================================================
    # STEP 5: Synthesize NOVEL Data Samples from Scratch! 🎉
    # =====================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Synthesizing Novel Samples (Generative Capability)")
    print("=" * 70)

    n_synthetic = 5
    synthetic_class_0 = np.random.multivariate_normal(learned_mu_0, learned_cov_0, n_synthetic)
    synthetic_class_1 = np.random.multivariate_normal(learned_mu_1, learned_cov_1, n_synthetic)

    print(f"Generating {n_synthetic} brand new synthetic points for Class 0 (Red):")
    for i, pt in enumerate(synthetic_class_0, 1):
        print(f"  Synthetic Sample #{i}: X1 = {pt[0]:>6.2f}, X2 = {pt[1]:>6.2f}")

    print(f"\nGenerating {n_synthetic} brand new synthetic points for Class 1 (Blue):")
    for i, pt in enumerate(synthetic_class_1, 1):
        print(f"  Synthetic Sample #{i}: X1 = {pt[0]:>6.2f}, X2 = {pt[1]:>6.2f}")

    print("\n" + "=" * 70)
    print("KEY TAKEAWAY:")
    print("• Discriminative: Superb boundary classifier, zero generative ability.")
    print("• Generative: Learns data distribution -> Classifies AND Synthesizes!")
    print("=" * 70)


if __name__ == "__main__":
    main()
