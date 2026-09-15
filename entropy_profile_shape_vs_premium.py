import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Complete exact dataset from your chart
EXACT_CHART_DATA = [
    # lang, raw_entropy, mono_premium, mean_entropy, tv_h
    ("heb_Hebr", 1.52, 0.76, 1.68, 0.380),
    ("arb_Arab", 1.36, 0.94, 1.60, 0.420),
    ("amh_Ethi", 1.31, 0.77, 1.97, 0.395),
    ("tha_Thai", 1.27, 1.11, 1.50, 0.610),
    ("tam_Taml", 1.24, 1.06, 1.21, 0.550),
    ("jpn_Jpan", 1.23, 0.65, 3.16, 0.335),
    ("cmn_Hans", 1.22, 0.50, 4.72, 0.284),
    ("kat_Geor", 1.16, 1.13, 1.24, 0.580),
    ("vie_Latn", 1.16, 1.19, 1.31, 0.640),
    ("kor_Hang", 1.11, 0.64, 2.69, 0.312),
    ("deu_Latn", 1.11, 1.19, 1.24, 0.620),
    ("hrv_Latn", 1.10, 1.02, 1.41, 0.530),
    ("ron_Latn", 1.08, 1.08, 1.24, 0.540),
    ("srp_Cyrl", 1.07, 1.01, 1.36, 0.510),
    ("spa_Latn", 1.07, 1.14, 1.19, 0.590),
    ("ita_Latn", 1.05, 1.09, 1.21, 0.560),
    ("fra_Latn", 1.04, 1.11, 1.16, 0.570),
    ("fin_Latn", 1.03, 1.01, 1.33, 0.520),
    ("nya_Latn", 1.01, 1.15, 1.12, 0.600),
    ("eng_Latn", 1.00, 1.00, 1.27, 0.500),
]

CHAR_KURTOSIS = {
    "srp_Cyrl": -0.4489,
    "nya_Latn": -0.4708,
    "jpn_Jpan": -0.4730,
    "deu_Latn": -0.4872,
    "fin_Latn": -0.5457,
    "kor_Hang": -0.6201,
    "spa_Latn": -0.6535,
    "ita_Latn": -0.6628,
    "eng_Latn": -0.6921,
    "fra_Latn": -0.6929,
    "ron_Latn": -0.7026,
    "hrv_Latn": -0.7373,
    "cmn_Hans": -0.8983,
    "vie_Latn": -0.9218,
    "tha_Thai": -0.9497,
    "kat_Geor": -1.0578,
    "arb_Arab": -1.0904,
    "tam_Taml": -1.1547,
    "amh_Ethi": -1.2381,
    "heb_Hebr": -1.3450,
}


def create_triplet_plots():
    df = pd.DataFrame(
        EXACT_CHART_DATA,
        columns=[
            "lang",
            "raw_entropy",
            "mono_premium",
            "mean_entropy",
            "total_variation",
        ],
    )
    df["short_lang"] = df["lang"].apply(lambda x: x.split("_")[0])
    df["kurtosis"] = df["lang"].map(CHAR_KURTOSIS)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6), dpi=300)

    # ----------------------------------------------------
    # Plot A: Kurtosis vs Raw Entropy Premium
    # ----------------------------------------------------
    r1 = np.corrcoef(df["kurtosis"], df["raw_entropy"])[0, 1]
    sns.regplot(
        data=df,
        x="kurtosis",
        y="raw_entropy",
        scatter_kws={"s": 75, "color": "#2b5c8f", "alpha": 0.85},
        line_kws={"color": "#e74c3c", "linewidth": 2},
        ax=axes[0],
    )
    axes[0].invert_xaxis()

    for _, row in df.iterrows():
        axes[0].text(
            row["kurtosis"] - 0.012,
            row["raw_entropy"] + 0.005,
            row["short_lang"],
            fontsize=9,
            weight="bold",
            alpha=0.85,
        )

    axes[0].set_title(
        f"(A) Kurtosis vs. Raw Entropy Premium\n$r = {r1:.3f}$ | $R^2 = {r1**2:.3f}$",
        fontsize=11,
        weight="bold",
        pad=12,
    )
    axes[0].set_xlabel(
        "Character Kurtosis $\\gamma_2$ (Broad Shoulders $\\rightarrow$)",
        fontsize=10,
    )
    axes[0].set_ylabel("Raw Entropy Premium Multiplier", fontsize=10)

    # ----------------------------------------------------
    # Plot B: Step Volatility vs Monotonicity Premium
    # ----------------------------------------------------
    r2 = np.corrcoef(df["total_variation"], df["mono_premium"])[0, 1]
    sns.regplot(
        data=df,
        x="total_variation",
        y="mono_premium",
        scatter_kws={"s": 75, "color": "#27ae60", "alpha": 0.85},
        line_kws={"color": "#8e44ad", "linewidth": 2},
        ax=axes[1],
    )

    for _, row in df.iterrows():
        color = (
            "#27ae60"
            if row["short_lang"] in ["cmn", "jpn", "kor", "heb", "amh"]
            else "black"
        )
        axes[1].text(
            row["total_variation"] + 0.006,
            row["mono_premium"] + 0.005,
            row["short_lang"],
            fontsize=9,
            weight="bold",
            color=color,
            alpha=0.85,
        )

    axes[1].set_title(
        f"(B) Step Volatility vs. Monotonicity Premium\n$r = {r2:.3f}$ | $R^2 = {r2**2:.3f}$",
        fontsize=11,
        weight="bold",
        pad=12,
    )
    axes[1].set_xlabel(
        "Step Volatility / Total Variation $\\mathrm{TV}_H$", fontsize=10
    )
    axes[1].set_ylabel("Raw Monotonicity Premium Multiplier", fontsize=10)

    # ----------------------------------------------------
    # Plot C: Character Density (Mean Entropy) vs Step Volatility
    # ----------------------------------------------------
    r3 = np.corrcoef(df["mean_entropy"], df["total_variation"])[0, 1]
    sns.regplot(
        data=df,
        x="mean_entropy",
        y="total_variation",
        scatter_kws={"s": 75, "color": "#d35400", "alpha": 0.85},
        line_kws={"color": "#2980b9", "linewidth": 2},
        ax=axes[2],
    )

    for _, row in df.iterrows():
        axes[2].text(
            row["mean_entropy"] + 0.05,
            row["total_variation"] + 0.005,
            row["short_lang"],
            fontsize=9,
            weight="bold",
            alpha=0.85,
        )

    axes[2].set_title(
        f"(C) Character Density vs. Step Volatility\n$r = {r3:.3f}$ | $R^2 = {r3**2:.3f}$",
        fontsize=11,
        weight="bold",
        pad=12,
    )
    axes[2].set_xlabel(
        "Character Density / Mean Entropy $\\bar{H}$ (bits/char)", fontsize=10
    )
    axes[2].set_ylabel(
        "Step Volatility / Total Variation $\\mathrm{TV}_H$", fontsize=10
    )

    plt.tight_layout()
    plt.savefig("triplet_premium_analysis.png")
    print(f"Plot A (Kurtosis):        r = {r1:.4f}, R^2 = {r1**2:.4f}")
    print(f"Plot B (Step Volatility): r = {r2:.4f}, R^2 = {r2**2:.4f}")
    print(f"Plot C (Density -> TV):   r = {r3:.4f}, R^2 = {r3**2:.4f}")
    plt.show()


if __name__ == "__main__":
    create_triplet_plots()