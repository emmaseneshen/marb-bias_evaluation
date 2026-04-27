"""
run_disability_models.py

Purpose:
Run the MARB disability dataset through 3 models:
- BERT
- RoBERTa
- GPT-2

This script reuses the original MARB evaluation functions instead of rewriting them.
That way, your setup stays close to the paper's implementation.

How to run:
    python src/run_disability_models.py

Optional arguments:
    python src/run_disability_models.py --n_ex 100
    python src/run_disability_models.py --device cpu

What it does:
1. Loads the disability CSV
2. Runs BERT with PPPL
3. Runs RoBERTa with PPPL
4. Runs GPT-2 with PPL
5. Saves the result CSVs in your results folder
"""

print("TOP OF FILE STARTED")

import argparse
print("ARGPARSE IMPORTED")

import os
print("OS IMPORTED")

import sys
print("SYS IMPORTED")

from pathlib import Path
print("PATHLIB IMPORTED")

import torch
print("TORCH IMPORTED")

from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    RobertaForMaskedLM,
    GPT2LMHeadModel,
    logging,
)
print("TRANSFORMERS IMPORTED")

# -------------------------------------------------------------------
# Add the MARB code directory to Python's import path
#
# This lets us import the original MARB evaluation functions from:
# external/MARB/code/utils.py
# -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MARB_CODE_DIR = REPO_ROOT / "external" / "MARB" / "code"

sys.path.append(str(MARB_CODE_DIR))

from utils import evaluate_masked, evaluate_autoregressive  # noqa: E402
print("IMPORTED MARB UTILS")

# import function to compute disability differences
from compute_disability_diffs import compute_diffs

def get_default_dataset_path() -> Path:
    """
    Returns the default path to the disability dataset in your repo.

    Based on your screenshot, this should be:
    external/MARB/data/datasets/disability.csv
    """
    return REPO_ROOT / "external" / "MARB" / "data" / "datasets" / "disability.csv"


def get_default_results_dir() -> Path:
    """
    Returns the default output directory for model results.

    We create a subfolder specifically for disability results
    so your outputs stay organized.
    """
    return REPO_ROOT / "results" / "disability"


def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")

    """
    Load the correct Hugging Face model + tokenizer
    for one of the 3 models in your project.

    Parameters
    ----------
    model_name : str
        One of: 'bert', 'roberta', 'gpt2'

    Returns
    -------
    tokenizer, model, eval_fn, metric
        tokenizer : Hugging Face tokenizer
        model     : Hugging Face model
        eval_fn   : MARB evaluation function to use
        metric    : Metric string expected by MARB code
    """
    model_name = model_name.lower()

    if model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        eval_fn = evaluate_masked
        metric = "PPPL"

    elif model_name == "roberta":
        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
        model = RobertaForMaskedLM.from_pretrained("FacebookAI/roberta-base")
        eval_fn = evaluate_masked
        metric = "PPPL"

    elif model_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
        eval_fn = evaluate_autoregressive
        metric = "PPL"

        # GPT-2 does not always have a pad token by default.
        # Setting pad_token to eos_token helps avoid tokenizer issues.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    return tokenizer, model, eval_fn, metric


def run_one_model(
    model_name: str,
    input_csv: Path,
    output_dir: Path,
    device: str,
    n_ex: int | None = None,
):
    """
    Run one model on the disability dataset using the MARB evaluation code.

    Parameters
    ----------
    model_name : str
        One of: 'bert', 'roberta', 'gpt2'
    input_csv : Path
        Path to disability.csv
    output_dir : Path
        Directory where result CSVs should be saved
    device : str
        'cuda', 'cpu', or 'mps' (depending on your machine)
    n_ex : int | None
        If provided, evaluate only a small subset of examples.
        This is very useful for testing before a full run.
    """
    print(f"\n{'=' * 70}")
    print(f"Running model: {model_name}")
    print(f"{'=' * 70}")

    tokenizer, model, eval_fn, metric = load_model_and_tokenizer(model_name)

    # These arguments match what the original MARB utils.py expects.
    eval_args = {
        "input": str(input_csv),
        "model": model,
        "tokenizer": tokenizer,
        "outdir": str(output_dir),
        "device": device,
        "metric": metric,
        "metric_type": "all-tokens",  # matches the default MARB setup
        "n_ex": n_ex,
    }

    # Run the model evaluation
    df_scores = eval_fn(eval_args)

    # Save the raw PPL / PPPL scores
    results_path = output_dir / f"{model_name}_disability_results.csv"
    df_scores.to_csv(results_path, index=False)

    # Compute disability descriptor score - original score
    diffs_path = output_dir / f"{model_name}_disability_diffs.csv"
    compute_diffs(results_path, diffs_path)

    print(f"Finished {model_name}.")
    print(f"Output shape: {df_scores.shape}")
    print(f"Saved raw results to: {results_path}")
    print(f"Saved diff results to: {diffs_path}")

    return df_scores


def choose_device(user_device: str | None = None) -> str:
    """
    Choose a device for running models.

    Priority:
    1. User-specified device
    2. CUDA if available
    3. MPS for Apple Silicon if available
    4. CPU otherwise
    """
    if user_device:
        return user_device

    if torch.cuda.is_available():
        return "cuda"

    # Apple Silicon support
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main():
    print("SCRIPT STARTED")

    parser = argparse.ArgumentParser(
        description="Run BERT, RoBERTa, and GPT-2 on the MARB disability dataset."
    )

    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(get_default_dataset_path()),
        help="Path to disability.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(get_default_results_dir()),
        help="Directory to save result CSV files",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: cuda, cpu, or mps. If omitted, auto-detects.",
    )

    parser.add_argument(
        "--n_ex",
        type=int,
        default=None,
        help="Optional subset size for testing. Example: --n_ex 100",
    )

    args = parser.parse_args()

    # Reduce Hugging Face warning spam
    logging.set_verbosity_error()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    device = choose_device(args.device)

    # Basic safety checks
    if not input_csv.exists():
        raise FileNotFoundError(f"Could not find dataset file: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting disability model runs...")
    print(f"Dataset: {input_csv}")
    print(f"Results directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Subset size: {args.n_ex if args.n_ex is not None else 'full dataset'}")

    # ---------------------------------------------------------------
    # Run all 3 models
    #
    # These match your project plan:
    # - BERT
    # - RoBERTa
    # - GPT-2
    # ---------------------------------------------------------------
    model_names = ["bert", "roberta", "gpt2"]

    for model_name in model_names:
        run_one_model(
            model_name=model_name,
            input_csv=input_csv,
            output_dir=output_dir,
            device=device,
            n_ex=args.n_ex,
        )

    print(f"\nAll model runs completed.")
    print(f"Check your result files in: {output_dir}")


if __name__ == "__main__":
    main()