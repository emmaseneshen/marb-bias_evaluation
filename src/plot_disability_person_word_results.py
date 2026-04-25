import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


RESULTS_DIR = Path("results/disability")
FIGURES_DIR = Path("figures/disability")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_person_word_results():
    bert = pd.read_csv(RESULTS_DIR / "bert_disability_stats_by_person_word.csv")
    roberta = pd.read_csv(RESULTS_DIR / "roberta_disability_stats_by_person_word.csv")
    gpt2 = pd.read_csv(RESULTS_DIR / "gpt2_disability_stats_by_person_word.csv")

    df = pd.concat([bert, roberta, gpt2], ignore_index=True)

    # Rename models for nicer plot labels
    df["model"] = df["model"].replace({
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "gpt2": "gpt2"
    })

    return df


def plot_figure4_style_boxplot(df):
    """
    Figure 4-style plot:
    Shows spread of effect sizes across disability descriptors,
    grouped by person-word and model.
    """
    plt.figure(figsize=(10, 6))

    sns.boxplot(
        data=df,
        x="person_word",
        y="rank_biserial_r",
        hue="model"
    )

    plt.axhline(0, linestyle="--")
    plt.xlabel("")
    plt.ylabel("Effect size")
    plt.title("Spread of Disability Effect Sizes by Model and Person Word")
    plt.ylim(-1.05, 1.05)
    plt.legend(title="Model")
    plt.tight_layout()

    output_path = FIGURES_DIR / "figure4_style_disability_boxplot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved Figure 4-style plot to {output_path}")


def plot_figure5_style_descriptor_breakdown(df, model_name):
    """
    Figure 5-style plot:
    Detailed breakdown of effect sizes by descriptor for one model,
    with separate lines for person-word.
    """
    model_df = df[df["model"] == model_name]

    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=model_df,
        x="descriptor",
        y="rank_biserial_r",
        hue="person_word",
        marker="o"
    )

    plt.axhline(0, linestyle="--")
    plt.xlabel("")
    plt.ylabel("Effect size")
    plt.title(f"Detailed Disability Descriptor Results for {model_name}")
    plt.ylim(-1.05, 1.05)
    plt.xticks(rotation=60, ha="right")
    plt.legend(title="Person word")
    plt.tight_layout()

    safe_model_name = model_name.replace("/", "-").replace(" ", "_")
    output_path = FIGURES_DIR / f"figure5_style_{safe_model_name}_descriptor_breakdown.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved Figure 5-style plot to {output_path}")


if __name__ == "__main__":
    df = load_person_word_results()

    # Figure 4-style: one boxplot comparing all models/person-words
    plot_figure4_style_boxplot(df)

    # Figure 5-style: one descriptor-level plot per model
    for model_name in ["bert-base-uncased", "roberta-base", "gpt2"]:
        plot_figure5_style_descriptor_breakdown(df, model_name)