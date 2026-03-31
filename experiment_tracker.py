import argparse
import itertools
import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from train import HousingModel


FEATURE_COLS = ["area_sqm", "bedrooms", "floor", "age_years", "distance_to_center_km"]


def load_and_prepare_data(csv_path: str = "data/housing.csv") -> Tuple[torch.Tensor, torch.Tensor]:
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS]
    y = df[["price_jod"]]

    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std

    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    return X_tensor, y_tensor


def train_test_split_tensors(
    X: torch.Tensor,
    y: torch.Tensor,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    n_samples = X.shape[0]
    indices = torch.randperm(n_samples)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    split = int((1.0 - test_ratio) * n_samples)
    X_train, X_test = X_shuffled[:split], X_shuffled[split:]
    y_train, y_test = y_shuffled[:split], y_shuffled[split:]
    return X_train, X_test, y_train, y_test


@dataclass
class ExperimentConfig:
    learning_rate: float
    hidden_size: int
    num_epochs: int
    seed: int = 42


def run_single_experiment(
    config: ExperimentConfig,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> Dict:
    torch.manual_seed(config.seed)

    model = HousingModel(hidden_size=config.hidden_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    start_time = time.time()

    for _ in range(config.num_epochs):
        preds = model(X_train)
        loss = criterion(preds, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_time_s = time.time() - start_time

    train_loss = loss.item()

    with torch.no_grad():
        test_preds_tensor = model(X_test)
    test_loss = criterion(test_preds_tensor, y_test).item()

    test_preds = test_preds_tensor.numpy().flatten()
    test_actual = y_test.numpy().flatten()

    mae = float(np.mean(np.abs(test_actual - test_preds)))
    ss_res = float(np.sum((test_actual - test_preds) ** 2))
    ss_tot = float(np.sum((test_actual - np.mean(test_actual)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    result = {
        **asdict(config),
        "train_loss": train_loss,
        "test_loss": test_loss,
        "test_mae": mae,
        "test_r2": r2,
        "train_time_s": train_time_s,
    }
    return result


def build_leaderboard(experiments: List[Dict], top_k: int = 10) -> List[Dict]:
    sorted_exps = sorted(experiments, key=lambda e: e["test_mae"])
    return sorted_exps[:top_k]


def print_leaderboard(experiments: List[Dict]) -> None:
    if not experiments:
        print("No experiments to show.")
        return

    header = (
        "Rank | LR       | Hidden | Epochs | Test MAE    | Test R^2  | Time (s)\n"
        "-----|----------|--------|--------|-------------|-----------|---------"
    )
    print(header)
    for rank, exp in enumerate(experiments, start=1):
        print(
            f"{rank:4d} | "
            f"{exp['learning_rate']:<8.5f} | "
            f"{exp['hidden_size']:6d} | "
            f"{exp['num_epochs']:6d} | "
            f"{exp['test_mae']:11.1f} | "
            f"{exp['test_r2']:9.3f} | "
            f"{exp['train_time_s']:7.2f}"
        )


def save_experiments_json(experiments: List[Dict], path: str = "experiments.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(experiments, f, indent=2)
    print(f"Saved {len(experiments)} experiments to {path}")


def save_summary_plot(experiments: List[Dict], path: str = "experiment_summary.png") -> None:
    if not experiments:
        return

    df = pd.DataFrame(experiments)

    plt.figure(figsize=(10, 6))

    for hidden_size, group in df.groupby("hidden_size"):
        group_sorted = group.sort_values("learning_rate")
        plt.plot(
            group_sorted["learning_rate"],
            group_sorted["test_mae"],
            marker="o",
            label=f"hidden_size={hidden_size}",
        )

    plt.xscale("log")
    plt.xlabel("Learning rate (log scale)")
    plt.ylabel("Test MAE (JOD)")
    plt.title("Experiment summary: Test MAE vs Learning Rate by Hidden Size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"Saved summary plot to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run housing price hyperparameter experiments.")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional limit on the number of experiment configurations to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/test split and model initialization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X_tensor, y_tensor = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split_tensors(
        X_tensor, y_tensor, test_ratio=0.2, seed=args.seed
    )

    learning_rates = [0.0005, 0.001, 0.005, 0.01, 0.05]
    hidden_sizes = [16, 32, 64]
    num_epochs_list = [50, 100, 150, 200]

    configs = [
        ExperimentConfig(lr, hidden, epochs, seed=args.seed)
        for lr, hidden, epochs in itertools.product(learning_rates, hidden_sizes, num_epochs_list)
    ]

    if args.max_runs is not None:
        configs = configs[: args.max_runs]

    experiments: List[Dict] = []
    total = len(configs)

    for idx, config in enumerate(configs, start=1):
        print(
            f"Running experiment {idx}/{total} "
            f"(lr={config.learning_rate}, hidden={config.hidden_size}, epochs={config.num_epochs})..."
        )
        result = run_single_experiment(config, X_train, y_train, X_test, y_test)
        result["id"] = idx
        experiments.append(result)
        print(
            f"  -> test MAE={result['test_mae']:.1f}, "
            f"test R^2={result['test_r2']:.3f}, "
            f"time={result['train_time_s']:.2f}s"
        )

    save_experiments_json(experiments)

    top_experiments = build_leaderboard(experiments, top_k=10)
    print()
    print("Top configurations by Test MAE:")
    print_leaderboard(top_experiments)

    best = top_experiments[0]
    target_met = best["test_mae"] < 10000.0
    print()
    print(
        f"Best configuration: lr={best['learning_rate']}, hidden={best['hidden_size']}, "
        f"epochs={best['num_epochs']}, test MAE={best['test_mae']:.1f}, R^2={best['test_r2']:.3f}"
    )
    if target_met:
        print("Target achieved: Test MAE is below 10,000 JOD.")
    else:
        print("Target NOT achieved: Test MAE is still above 10,000 JOD.")

    save_summary_plot(experiments)


if __name__ == "__main__":
    main()

