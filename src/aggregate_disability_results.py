import pandas as pd
from pathlib import Path


def aggregate_results():
    # Load the stats files from Step 4
    bert = pd.read_csv("results/disability/bert_disability_stats.csv")
    roberta = pd.read_csv("results/disability/roberta_disability_stats.csv")
    gpt2 = pd.read_csv("results/disability/gpt2_disability_stats.csv")

    # Combine into one dataframe
    df_all = pd.concat([bert, roberta, gpt2], ignore_index=True)

    # Keep only the key columns for presentation
    df_all = df_all[[
        "model",
        "descriptor",
        "mean_diff",
        "median_diff",
        "rank_biserial_r",
        "p_value"
    ]]

    # Optional: round values for readability
    df_all["mean_diff"] = df_all["mean_diff"].round(2)
    df_all["median_diff"] = df_all["median_diff"].round(2)
    df_all["rank_biserial_r"] = df_all["rank_biserial_r"].round(3)
    df_all["p_value"] = df_all["p_value"].apply(lambda x: f"{x:.2e}")

    # Save combined results
    output_path = "results/disability/all_models_disability_summary.csv"
    Path("results/disability").mkdir(parents=True, exist_ok=True)
    df_all.to_csv(output_path, index=False)

    print(f"Saved combined results to {output_path}")


if __name__ == "__main__":
    aggregate_results()