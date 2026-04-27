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
# -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MARB_CODE_DIR = REPO_ROOT / "external" / "MARB" / "code"

sys.path.append(str(MARB_CODE_DIR))

from utils import evaluate_masked, evaluate_autoregressive  # noqa: E402
print("IMPORTED MARB UTILS")


def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")

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
    print(f"\n{'=' * 70}")
    print(f"Running model: {model_name}")
    print(f"{'=' * 70}")

    tokenizer, model, eval_fn, metric = load_model_and_tokenizer(model_name)

    eval_args = {
        "input": str(input_csv),
        "model": model,
        "tokenizer": tokenizer,
        "outdir": str(output_dir),
        "device": device,
        "metric": metric,
        "metric_type": "all-tokens",
        "n_ex": n_ex,
    }

    df_scores = eval_fn(eval_args)

    print(f"Finished {model_name}.")
    print(f"Output shape: {df_scores.shape}")
    return df_scores


def choose_device(user_device: str | None = None) -> str:
    if user_device:
        return user_device

    if torch.cuda.is_available():
        return "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def main():
    print("SCRIPT STARTED")

    parser = argparse.ArgumentParser(
        description="Run BERT, RoBERTa, and GPT-2 on a MARB category dataset."
    )

    # EXISTING ARGS (kept for compatibility)
    parser.add_argument("--input_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

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

    # NEW ARGUMENT
    parser.add_argument(
        "--category",
        type=str,
        choices=["disability", "race", "queerness"],
        required=True,
        help="MARB category to evaluate",
    )

    args = parser.parse_args()

    # USE CATEGORY
    category = args.category

    logging.set_verbosity_error()

    # DYNAMIC PATHS BASED ON CATEGORY
    input_csv = REPO_ROOT / "external" / "MARB" / "data" / "datasets" / f"{category}.csv"
    output_dir = REPO_ROOT / "results" / category

    device = choose_device(args.device)

    # Force CPU for RoBERTa to avoid MPS out-of-memory crashes
    device = "cpu"

    if not input_csv.exists():
        raise FileNotFoundError(f"Could not find dataset file: {input_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting {category} model runs...")
    print(f"Dataset: {input_csv}")

    import pandas as pd

    df_debug = pd.read_csv(input_csv)

    print("\nDEBUG CHECK:")
    print("Columns:", df_debug.columns.tolist())
    print(df_debug.head(2))

    # ----------------------------------------
    # FILTER TO "person" ROWS ONLY (2–9501)
    # ----------------------------------------
    df_debug = df_debug.iloc[1:9501]

    print(f"\nFiltered to person rows: {len(df_debug)} rows")

    # Save filtered dataset to a temporary file
    filtered_csv = output_dir / f"{category}_person_only.csv"
    df_debug.to_csv(filtered_csv, index=False)

    # Use filtered CSV going forward
    input_csv = filtered_csv

    print(f"Results directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Subset size: {args.n_ex if args.n_ex is not None else 'full dataset'}")

    model_names = ["bert"]

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