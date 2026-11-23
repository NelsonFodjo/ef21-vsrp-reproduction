from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Squash real-valued scores into (0, 1 ) range."""
    return 1.0 / (1.0 + np.exp(-z))

def gd_quadratic_path(
        x0: float, step_size: float, n_steps: int) -> np.ndarray:
    """Gradient descent on f(x) = x^2, where gradient is 2x."""
    xs = [x0]
    x = x0

    for _ in range(n_steps):
        grad = 2.0 * x
        x = x - step_size * grad
        xs.append(x)
    return np.array(xs)

def plot_four_step_sizes(
        x0: float = 3.0, n_steps: int = 8, out_path: str = "results/day1_steps.png"
) -> None:
    """Plot step sixes 0.1, 0.9, 1.0, and 1.1 on f(x) = x^2 in one figure."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    step_sizes = [0.1, 0.9, 1.0, 1.1]
    iterations = np.arange(n_steps + 1)

    plt.figure(figsize=(8, 5))
    for lr in step_sizes:
        path = gd_quadratic_path(x0=x0, step_size=lr, n_steps=n_steps)
        plt.plot(iterations, path, marker="o", label=f"step = {lr}")

    plt.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    plt.title("Gradient Descent on f(x) = x^2 from x0 = 3")
    plt.xlabel("Iteration")
    plt.ylabel("x_t")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved plot to {out_path}")

def run_toy_sigmoid() -> None:
    """10-sample toy matrix A @ w passed through sigmoid."""
    rng = np.random.default_rng(seed=42)
    A = rng.normal(loc=0.0, scale=1.0, size=(10,3))
    w = np.array([0.8, -0.5, 0.3])

    scores = A @ w
    probs = sigmoid(scores)

    print("Toy scores (A @ w): ", np.round(scores, 4))
    print("Toy sigmoid probabilities:", np.round(probs, 4))



if __name__ == "__main__":
    plot_four_step_sizes()
    run_toy_sigmoid()

    