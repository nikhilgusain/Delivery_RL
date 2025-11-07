import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate
import os

def plot_training_comparison(times, cfg):
    df = pd.DataFrame(times.items(), columns=["Configuration", "Time (s)"])
    df.to_csv("dqn_training_comparison.csv", index=False)
    plt.figure(figsize=(8,6))
    plt.bar(df["Configuration"], df["Time (s)"])
    plt.xticks(rotation=30, ha='right'); plt.ylabel("Time (s)")
    plt.title("DQN Training Time Comparison (CPU vs GPU)")
    os.makedirs(cfg["plots_dir"], exist_ok=True)
    plt.savefig(os.path.join(cfg["plots_dir"], "dqn_training_comparison.png"), dpi=600)
    plt.close()
    print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
