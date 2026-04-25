import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

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


def add_person_word_column(df):
    """
    Adds person_word based on row order.

    First third = a person
    Second third = a woman
    Last third = a man
    """
    n = len(df)

    if n % 3 != 0:
        raise ValueError(f"Number of rows ({n}) is not divisible by 3.")

    group_size = n // 3

    df = df.copy()
    df["person_word"] = (
        ["a person"] * group_size
        + ["a woman"] * group_size
        + ["a man"] * group_size
    )

    return df


def rank_biserial_from_wilcoxon(diffs):
    """
    Effect size for the Wilcoxon signed-rank test.
    Keeps the sign, so negative values mean most differences are negative.
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
    df = pd.read_csv(input_path)

    # Add person-word labels using the dataset’s row order
    df = add_person_word_column(df)

    results = []

    # Run stats separately for each person-word group
    for person_word in ["a person", "a woman", "a man"]:
        df_group = df[df["person_word"] == person_word]

        for descriptor in DISABILITY_DESCRIPTORS:
            diff_col = f"{descriptor}_diff"

            diffs = df_group[diff_col].dropna()
            diffs_nonzero = diffs[diffs != 0]

            if len(diffs_nonzero) == 0:
                continue

            stat, p_value = wilcoxon(diffs_nonzero)
            effect_size = rank_biserial_from_wilcoxon(diffs_nonzero)

            results.append({
                "model": model_name,
                "person_word": person_word,
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

    print(f"Saved person-word stats to {output_path}")


if __name__ == "__main__":
    files = {
        "bert": "bert-base-uncased_PPPL_all-tokens_100-ex_disability_diffs.csv",
        "roberta": "roberta-base_PPPL_all-tokens_100-ex_disability_diffs.csv",
        "gpt2": "gpt2_PPL_100-ex_disability_diffs.csv",
    }

    for model_name, filename in files.items():
        input_path = f"results/disability/{filename}"
        output_path = f"results/disability/{model_name}_disability_stats_by_person_word.csv"

        print(f"\nRunning person-word Wilcoxon tests for {model_name}...")
        run_stats_for_file(input_path, output_path, model_name)