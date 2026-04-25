import argparse
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon


def get_diff_columns(df):
    return [col for col in df.columns if col.endswith("_diff")]


def clean_descriptor_name(diff_col):
    return diff_col.replace("_diff", "")


def rank_biserial_from_wilcoxon(diffs):
    """
    Computes rank-biserial correlation for paired differences.
    This is the effect size for the Wilcoxon signed-rank test.
    """
    diffs = diffs.dropna()
    diffs = diffs[diffs != 0]

    if len(diffs) == 0:
        return None

    ranks = diffs.abs().rank()
    positive_rank_sum = ranks[diffs > 0].sum()
    negative_rank_sum = ranks[diffs < 0].sum()

    total_rank_sum = ranks.sum()

    return (positive_rank_sum - negative_rank_sum) / total_rank_sum


def infer_model_name(filename):
    if filename.startswith("bert-base-uncased"):
        return "bert"
    elif filename.startswith("roberta-base"):
        return "roberta"
    elif filename.startswith("gpt2"):
        return "gpt2"
    else:
        return "unknown"


def run_stats_for_file(input_path, output_path, model_name):
    df = pd.read_csv(input_path)

    diff_cols = get_diff_columns(df)
    print("Diff columns:", diff_cols)

    results = []

    for diff_col in diff_cols:
        descriptor = clean_descriptor_name(diff_col)

        diffs = df[diff_col].dropna()
        diffs_nonzero = diffs[diffs != 0]

        if len(diffs_nonzero) == 0:
            continue

        stat, p_value = wilcoxon(diffs_nonzero)
        effect_size = rank_biserial_from_wilcoxon(diffs_nonzero)

        results.append(
            {
                "model": model_name,
                "descriptor": descriptor,
                "mean_diff": diffs.mean(),
                "median_diff": diffs.median(),
                "wilcoxon_stat": stat,
                "p_value": p_value,
                "rank_biserial_r": effect_size,
                "n": len(diffs_nonzero),
            }
        )

    results_df = pd.DataFrame(results)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Saved stats results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Wilcoxon stats for MARB category diffs."
    )

    parser.add_argument(
        "--category",
        type=str,
        choices=["disability", "race", "queerness"],
        required=True,
        help="MARB category to compute stats for",
    )

    args = parser.parse_args()
    category = args.category

    results_dir = Path("results") / category

    diff_files = sorted(results_dir.glob("*_diffs.csv"))

    if len(diff_files) == 0:
        raise FileNotFoundError(f"No *_diffs.csv files found in {results_dir}")

    for input_path in diff_files:
        model_name = infer_model_name(input_path.name)
        output_path = results_dir / f"{model_name}_{category}_stats.csv"

        print(f"\nRunning Wilcoxon tests for {model_name}...")
        print(f"Input: {input_path}")

        run_stats_for_file(input_path, output_path, model_name)