import pandas as pd
import torch
import math
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM

# ====== Load Models ======
deberta_name = "microsoft/deberta-base"
gpt2_name = "gpt2"

tokenizer_deberta = AutoTokenizer.from_pretrained(deberta_name)
model_deberta = AutoModelForMaskedLM.from_pretrained(deberta_name)

tokenizer_gpt2 = AutoTokenizer.from_pretrained(gpt2_name)
model_gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_name)

# GPT-2 sometimes needs an explicit pad token
if tokenizer_gpt2.pad_token is None:
    tokenizer_gpt2.pad_token = tokenizer_gpt2.eos_token

# ====== PPPL for DeBERTa ======
def pseudo_perplexity_fast(sentence):
    tokens = tokenizer_deberta(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    input_ids = tokens["input_ids"]

    seq_len = input_ids.size(1)
    if seq_len <= 2:
        return float("nan")

    masked_inputs = input_ids.repeat(seq_len - 2, 1)

    for i in range(1, seq_len - 1):
        masked_inputs[i - 1, i] = tokenizer_deberta.mask_token_id

    with torch.no_grad():
        outputs = model_deberta(masked_inputs)
        logits = outputs.logits

    loss = 0.0
    for i in range(1, seq_len - 1):
        log_prob = torch.log_softmax(logits[i - 1, i], dim=0)[input_ids[0, i]]
        loss += -log_prob.item()

    return math.exp(loss / (seq_len - 2))

# ====== PPL for GPT-2 ======
def perplexity(sentence: str) -> float:
    inputs = tokenizer_gpt2(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    with torch.no_grad():
        outputs = model_gpt2(**inputs, labels=inputs["input_ids"])
    return math.exp(outputs.loss.item())


# ====== Helper to run one model on one dataset/category ======
def run_experiment(df: pd.DataFrame, descriptors: list[str], score_fn, model_name: str, category_name: str) -> None:
    print(f"\n=== {model_name} on {category_name} ===")

    for desc in descriptors:
        print(f"\n--- Descriptor: {desc} ---")
        pos_count = 0
        neg_count = 0
        skipped_count = 0

        # keep only rows where both original and descriptor sentence exist
        sub_df = df[["original", desc]].dropna()

        for _, row in sub_df.iterrows():
            base = row["original"]
            marked = row[desc]

            try:
                unmarked_score = score_fn(base)
                marked_score = score_fn(marked)
                diff = marked_score - unmarked_score

                if math.isnan(diff):
                    skipped_count += 1
                elif diff > 0:
                    pos_count += 1
                else:
                    neg_count += 1

            except Exception:
                skipped_count += 1

        print(f"Summary for {desc}: +{pos_count} / -{neg_count} / skipped {skipped_count}")


# ====== Load Data ======
race_df = pd.read_csv("data/datasets/race.csv")
queer_df = pd.read_csv("data/datasets/queerness.csv")

print("RACE COLUMNS:")
print(race_df.columns.tolist())

print("\nQUEERNESS COLUMNS:")
print(queer_df.columns.tolist())

# ====== Full dataset: DO NOT SUBSET ======
# race_df = race_df.head(100)
# queer_df = queer_df.head(100)

# ===== descriptors we want to test =====
race_descriptors = ["Asian", "Black", "Hispanic", "white"]
queer_descriptors = ["queer", "bisexual", "transgender", "straight"]

# ===== Run all four combinations =====
run_experiment(race_df, race_descriptors, pseudo_perplexity_fast, "DeBERTa", "Race")
run_experiment(queer_df, queer_descriptors, pseudo_perplexity_fast, "DeBERTa", "Queerness")
run_experiment(race_df, race_descriptors, perplexity, "GPT-2", "Race")
run_experiment(queer_df, queer_descriptors, perplexity, "GPT-2", "Queerness")