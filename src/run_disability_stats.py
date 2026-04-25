import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

# These should match the _diff columns you created in Step 3
DISABILITY_DESCRIPTORS = [
    "Deaf",
    "Blind",
    "With a Disability",
    "Wheelchair",
    "Cerebral Palsy",
    "Mental Illness",
    "Epilepsy",
    "Spinal Curvature",
    "Chronically Ill",
    "Short Stature",
    "Dyslexia",
    "Down's Syndrome",
    "Without a Disability",
]


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


def run_stats_for_file(input_path, output_path, model_name):
    # Load the file that already has the _diff columns
    df = pd.read_csv(input_path)

    results = []

    for descriptor in DISABILITY_DESCRIPTORS:
        diff_col = f"{descriptor}_diff"

        # Get the difference values for this descriptor
        diffs = df[diff_col].dropna()

        # Remove exact zeros because Wilcoxon tests nonzero paired differences
        diffs_nonzero = diffs[diffs != 0]

        if len(diffs_nonzero) == 0:
            continue

        # Wilcoxon signed-rank test:
        # tests whether the median difference is significantly different from 0
        stat, p_value = wilcoxon(diffs_nonzero)

        # Effect size: tells us how strong the difference is
        effect_size = rank_biserial_from_wilcoxon(diffs_nonzero)

        results.append({
            "model": model_name,
            "descriptor": descriptor,
            "mean_diff": diffs.mean(),
            "median_diff": diffs.median(),
            "wilcoxon_stat": stat,
            "p_value": p_value,
            "rank_biserial_r": effect_size,
            "n": len(diffs_nonzero),
        })

    results_df = pd.DataFrame(results)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)

    print(f"Saved stats results to {output_path}")


if __name__ == "__main__":
    files = {
        "bert": "bert-base-uncased_PPPL_all-tokens_100-ex_disability_diffs.csv",
        "roberta": "roberta-base_PPPL_all-tokens_100-ex_disability_diffs.csv",
        "gpt2": "gpt2_PPL_100-ex_disability_diffs.csv",
    }

    for model_name, filename in files.items():
        input_path = f"results/disability/{filename}"
        output_path = f"results/disability/{model_name}_disability_stats.csv"

        print(f"\nRunning Wilcoxon tests for {model_name}...")
        run_stats_for_file(input_path, output_path, model_name)