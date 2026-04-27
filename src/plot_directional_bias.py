import matplotlib.pyplot as plt
import numpy as np
import os

# ===== Ensure figures folder exists =====
os.makedirs("figures", exist_ok=True)

# ===== Data from output =====

gpt2_race = {
    "Asian": (26753, 1747),
    "Black": (27327, 1173),
    "Hispanic": (28182, 318),
    "white": (22457, 6043),
}

gpt2_queerness = {
    "queer": (27442, 1058),
    "bisexual": (27727, 773),
    "transgender": (26099, 2401),
    "straight": (27313, 1187),
}

deberta_race = {
    "Asian": (413, 28087),
    "Black": (26327, 2173),
    "Hispanic": (1546, 26954),
    "white": (6966, 21534),
}

deberta_queerness = {
    "queer": (9585, 18915),
    "bisexual": (2196, 26304),
    "transgender": (3278, 25222),
    "straight": (5969, 22531),
}


# ===== Helper function =====

def proportion_positive(data):
    return {desc: pos / (pos + neg) for desc, (pos, neg) in data.items()}


# ===== Plotting function =====

def plot_model_comparison(gpt2_data, deberta_data, category_name):
    labels = list(gpt2_data.keys())
    x = np.arange(len(labels))
    width = 0.35

    gpt2_props = proportion_positive(gpt2_data)
    deberta_props = proportion_positive(deberta_data)

    gpt2_values = [gpt2_props[label] for label in labels]
    deberta_values = [deberta_props[label] for label in labels]

    plt.figure(figsize=(8, 5))

    plt.bar(x - width / 2, gpt2_values, width, label="GPT-2")
    plt.bar(x + width / 2, deberta_values, width, label="DeBERTa")

    plt.axhline(0.5, linestyle="--", linewidth=1)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Proportion Positive")
    plt.title(f"Directional Reporting Bias: {category_name}")
    plt.legend()

    plt.tight_layout()

    # ===== Save figure =====
    filename = f"figures/{category_name.lower()}_directional_bias.png"
    plt.savefig(filename, dpi=300)

    print(f"Saved: {filename}")

    plt.close()  # prevents overlapping figures


# ===== Run plots =====

if __name__ == "__main__":
    plot_model_comparison(gpt2_race, deberta_race, "Race")
    plot_model_comparison(gpt2_queerness, deberta_queerness, "Queerness")