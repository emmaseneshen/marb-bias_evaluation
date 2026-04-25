import argparse
import pandas as pd
from pathlib import Path


def get_descriptor_columns(df):
    # These are metadata/base columns, not descriptor columns
    exclude_cols = ["person_word", "original"]

    # Any remaining columns are treated as descriptors
    descriptor_cols = [
        col for col in df.columns
        if col not in exclude_cols
    ]

    return descriptor_cols


def compute_diffs(input_path, output_path):
    # Load the model results
    df = pd.read_csv(input_path)

    descriptor_cols = get_descriptor_columns(df)

    print("Descriptor columns:", descriptor_cols)

    # For each descriptor, compute the reporting bias difference
    for descriptor in descriptor_cols:
        diff_col = f"{descriptor}_diff"

        # Difference = marked sentence score - original sentence score
        df[diff_col] = df[descriptor] - df["original"]

    # Make sure the output folder exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the updated dataframe with new _diff columns
    df.to_csv(output_path, index=False)

    print(f"Saved difference results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute MARB reporting bias differences for a category."
    )

    parser.add_argument(
        "--category",
        type=str,
        choices=["disability", "race", "queerness"],
        required=True,
    )

    args = parser.parse_args()
    category = args.category

    results_dir = Path("results") / category

    # Automatically find all model result files (NO hardcoding)
    files = list(results_dir.glob("*.csv"))

    for input_path in files:
        # Skip already-processed diff files
        if "diffs" in input_path.name:
            continue

        output_filename = input_path.name.replace(".csv", "_diffs.csv")
        output_path = results_dir / output_filename

        print(f"\nProcessing {input_path.name}...")

        compute_diffs(input_path, output_path)