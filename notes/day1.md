# Day 1 Notes: Step Size, Smoothness L, Matrix-Vector Scores & Sigmoid

- **What is smoothness L?** Smoothness L measures how steep or curved the loss function is (for a 1D smooth curve, the maximum second derivative), setting how fast the slope can change from point to point.
- **Why does a big step explode?** Gradient descent uses the local slope to step toward a minimum; if the step size is too large (above 2/L), the update jumps past the minimum and grows bigger each step instead of settling down.
- **Hand check on f(x) = x^2 (L = 2) from x0 = 3:** step 0.1 shrinks smoothly (3 -> 2.4 -> 1.92), step 0.9 flips signs while shrinking (3 -> -2.4 -> 1.92), step 1.0 bounces back and forth (3 -> -3 -> 3), and step 1.1 grows larger each step (3 -> -3.6 -> 4.32).
- **Hand check on f(x) = 5x^2 (L = 10):** step 0.1 (1/L) gives multiplier (1 - 10*0.1) = 0 (safe), while step 0.25 gives multiplier (1 - 10*0.25) = -1.5 (explodes).
- **Sigmoid & scores:** A @ w computes a raw score for each sample, and sigmoid(z) = 1 / (1 + exp(-z)) maps any score into (0, 1); sigma(0) = 0.5, sigma(2) ≈ 0.88, and sigma(-2) ≈ 0.12 because exp(2) is large so 1 / (1 + exp(2)) is small (and 1 - 0.88 ≈ 0.12).