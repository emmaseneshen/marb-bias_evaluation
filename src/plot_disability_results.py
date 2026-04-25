import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


RESULTS_DIR = Path("results/disability")
FIGURES_DIR = Path("figures/disability")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_summary_results():
    df = pd.read_csv(RESULTS_DIR / "all_models_disability_summary.csv")
    return df


def plot_effect_size_boxplot(df):
    """
    Figure 2-style:
    Distribution of effect sizes by model (keeps sign).
    """
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=df,
        x="model",
        y="rank_biserial_r"
    )

    # Add reference line at 0
    plt.axhline(0, linestyle="--")

    plt.xlabel("Model")
    plt.ylabel("Effect size (rank-biserial r)")
    plt.title("Spread of Disability Effect Sizes by Model")
    plt.ylim(-1.05, 1.05)
    plt.tight_layout()

    output_path = FIGURES_DIR / "disability_effect_size_boxplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved boxplot to {output_path}")


def plot_descriptor_effects(df):
    """
    Figure 5-style:
    Effect size per descriptor, colored by model.
    """
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=df,
        x="descriptor",
        y="rank_biserial_r",
        hue="model",
        marker="o"
    )

    # Add reference line at 0
    plt.axhline(0, linestyle="--")

    plt.xlabel("Disability descriptor")
    plt.ylabel("Effect size (rank-biserial r)")
    plt.title("Disability Effect Size by Descriptor and Model")
    plt.ylim(-1.05, 1.05)
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()

    output_path = FIGURES_DIR / "disability_descriptor_effects.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved descriptor plot to {output_path}")


if __name__ == "__main__":
    df = load_summary_results()

    plot_effect_size_boxplot(df)
    plot_descriptor_effects(df)