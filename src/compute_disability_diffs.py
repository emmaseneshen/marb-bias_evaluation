import pandas as pd
from pathlib import Path

# List of all disability descriptor columns in your dataset
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

def compute_diffs(input_path, output_path):
    # Load the model results (these contain PPPL or PPL scores)
    df = pd.read_csv(input_path)

    # For each disability descriptor, compute the reporting bias difference
    for descriptor in DISABILITY_DESCRIPTORS:
        diff_col = f"{descriptor}_diff"
        
        # Difference = (marked sentence score) - (original sentence score)
        df[diff_col] = df[descriptor] - df["original"]

    # Make sure the output folder exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the updated dataframe with new _diff columns
    df.to_csv(output_path, index=False)

    print(f"Saved disability difference results to {output_path}")


if __name__ == "__main__":
    # Change this to run for each model: "bert", "roberta", "gpt2"
    files = [
    "roberta-base_PPPL_all-tokens_all-ex_disability_person_only.csv",
    "gpt2_PPL_all-ex_disability_person_only.csv",
    ]

    for filename in files:
        input_path = f"results/disability/{filename}"

        # Create a clean output name by adding "_diffs"
        output_filename = filename.replace(".csv", "_diffs.csv")
        output_path = f"results/disability/{output_filename}"

        print(f"\nProcessing {filename}...")

        compute_diffs(input_path, output_path)