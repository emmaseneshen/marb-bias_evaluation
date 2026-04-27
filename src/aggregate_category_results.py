import argparse
import pandas as pd
from pathlib import Path


def aggregate_results(category):
    results_dir = Path("results") / category

    # Load the stats files
    bert = pd.read_csv(results_dir / f"bert_{category}_stats.csv")
    roberta = pd.read_csv(results_dir / f"roberta_{category}_stats.csv")
    gpt2 = pd.read_csv(results_dir / f"gpt2_{category}_stats.csv")

    # Combine into one dataframe
    df_all = pd.concat([bert, roberta, gpt2], ignore_index=True)

    # Keep only key columns
    df_all = df_all[
        [
            "model",
            "descriptor",
            "mean_diff",
            "median_diff",
            "rank_biserial_r",
            "p_value",
        ]
    ]

    # Round values for readability
    df_all["mean_diff"] = df_all["mean_diff"].round(2)
    df_all["median_diff"] = df_all["median_diff"].round(2)
    df_all["rank_biserial_r"] = df_all["rank_biserial_r"].round(3)
    df_all["p_value"] = df_all["p_value"].apply(lambda x: f"{x:.2e}")

    # Save combined results
    output_path = results_dir / f"all_models_{category}_summary.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)

    print(f"Saved combined results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate MARB results across models for a category."
    )

    parser.add_argument(
        "--category",
        type=str,
        choices=["disability", "race", "queerness"],
        required=True,
        help="MARB category to aggregate results for",
    )

    args = parser.parse_args()

    aggregate_results(args.category)