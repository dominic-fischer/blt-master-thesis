import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# =====================================================================
# EXACT DATASET FROM YOUR JSON AND CHART IMAGE
# =====================================================================

JSON_DATA = {
    "amh_Ethi": {
        "n_docs": 997,
        "n_bytes_total": 170250,
        "n_codepoints": 85125,
        "codepoints_per_baseline_codepoint": 0.6799447257855807,
        "density_index": 1.4707077826725403,
    },
    "arb_Arab": {
        "n_docs": 997,
        "n_bytes_total": 221148,
        "n_codepoints": 110574,
        "codepoints_per_baseline_codepoint": 0.8832212406345352,
        "density_index": 1.13221914735833,
    },
    "cmn_Hans": {
        "n_docs": 997,
        "n_bytes_total": 84358,
        "n_codepoints": 42179,
        "codepoints_per_baseline_codepoint": 0.3369091170503379,
        "density_index": 2.968159510656962,
    },
    "deu_Latn": {
        "n_docs": 997,
        "n_bytes_total": 293116,
        "n_codepoints": 146558,
        "codepoints_per_baseline_codepoint": 1.1706471556144864,
        "density_index": 0.8542283601031674,
    },
    "eng_Latn": {
        "n_docs": 997,
        "n_bytes_total": 250388,
        "n_codepoints": 125194,
        "codepoints_per_baseline_codepoint": 1.0,
        "density_index": 1.0,
    },
    "fin_Latn": {
        "n_docs": 997,
        "n_bytes_total": 267870,
        "n_codepoints": 133935,
        "codepoints_per_baseline_codepoint": 1.069819639918846,
        "density_index": 0.9347369992907008,
    },
    "fra_Latn": {
        "n_docs": 997,
        "n_bytes_total": 297578,
        "n_codepoints": 148789,
        "codepoints_per_baseline_codepoint": 1.1884674984424173,
        "density_index": 0.8414197286089697,
    },
    "heb_Hebr": {
        "n_docs": 997,
        "n_bytes_total": 195002,
        "n_codepoints": 97501,
        "codepoints_per_baseline_codepoint": 0.7787993034809975,
        "density_index": 1.2840278561245526,
    },
    "hrv_Latn": {
        "n_docs": 997,
        "n_bytes_total": 247738,
        "n_codepoints": 123869,
        "codepoints_per_baseline_codepoint": 0.9894164257073023,
        "density_index": 1.0106967845062123,
    },
    "ita_Latn": {
        "n_docs": 997,
        "n_bytes_total": 294180,
        "n_codepoints": 147090,
        "codepoints_per_baseline_codepoint": 1.174896560538045,
        "density_index": 0.8511387585831803,
    },
    "jpn_Jpan": {
        "n_docs": 997,
        "n_bytes_total": 109316,
        "n_codepoints": 54658,
        "codepoints_per_baseline_codepoint": 0.43658641787945107,
        "density_index": 2.2904972739580667,
    },
    "kat_Geor": {
        "n_docs": 997,
        "n_bytes_total": 275886,
        "n_codepoints": 137943,
        "codepoints_per_baseline_codepoint": 1.1018339537038517,
        "density_index": 0.9075777676286582,
    },
    "kor_Hang": {
        "n_docs": 997,
        "n_bytes_total": 126022,
        "n_codepoints": 63011,
        "codepoints_per_baseline_codepoint": 0.5033068677412655,
        "density_index": 1.9868594372411166,
    },
    "nya_Latn": {
        "n_docs": 997,
        "n_bytes_total": 282924,
        "n_codepoints": 141462,
        "codepoints_per_baseline_codepoint": 1.129942329504609,
        "density_index": 0.885000918974707,
    },
    "ron_Latn": {
        "n_docs": 997,
        "n_bytes_total": 282084,
        "n_codepoints": 141042,
        "codepoints_per_baseline_codepoint": 1.1265875361439046,
        "density_index": 0.8876363069156705,
    },
    "spa_Latn": {
        "n_docs": 997,
        "n_bytes_total": 298374,
        "n_codepoints": 149187,
        "codepoints_per_baseline_codepoint": 1.1916465645318466,
        "density_index": 0.8391749951403272,
    },
    "srp_Cyrl": {
        "n_docs": 997,
        "n_bytes_total": 249010,
        "n_codepoints": 124505,
        "codepoints_per_baseline_codepoint": 0.9944965413677972,
        "density_index": 1.0055339143006305,
    },
    "tam_Taml": {
        "n_docs": 997,
        "n_bytes_total": 292256,
        "n_codepoints": 146128,
        "codepoints_per_baseline_codepoint": 1.1672124862213844,
        "density_index": 0.8567420343808169,
    },
    "tha_Thai": {
        "n_docs": 997,
        "n_bytes_total": 240536,
        "n_codepoints": 120268,
        "codepoints_per_baseline_codepoint": 0.9606530664408838,
        "density_index": 1.0409585259586922,
    },
    "vie_Latn": {
        "n_docs": 997,
        "n_bytes_total": 263980,
        "n_codepoints": 131990,
        "codepoints_per_baseline_codepoint": 1.0542837516174897,
        "density_index": 0.9485112508523373,
    },
}

PREMIUMS_AND_TV = [
    # (lang, raw_mono_premium, total_variation_tv)
    ("heb_Hebr", 0.76, 0.380),
    ("arb_Arab", 0.94, 0.420),
    ("amh_Ethi", 0.77, 0.395),
    ("tha_Thai", 1.11, 0.610),
    ("tam_Taml", 1.06, 0.550),
    ("jpn_Jpan", 0.65, 0.335),
    ("cmn_Hans", 0.50, 0.284),
    ("kat_Geor", 1.13, 0.580),
    ("vie_Latn", 1.19, 0.640),
    ("kor_Hang", 0.64, 0.312),
    ("deu_Latn", 1.19, 0.620),
    ("hrv_Latn", 1.02, 0.530),
    ("ron_Latn", 1.08, 0.540),
    ("srp_Cyrl", 1.01, 0.510),
    ("spa_Latn", 1.14, 0.590),
    ("ita_Latn", 1.09, 0.560),
    ("fra_Latn", 1.11, 0.570),
    ("fin_Latn", 1.01, 0.520),
    ("nya_Latn", 1.15, 0.600),
    ("eng_Latn", 1.00, 0.500),
]


def plot_actual_density_relationships():
    # Build Combined DataFrame
    rows = []
    for lang, mono_p, tv in PREMIUMS_AND_TV:
        rows.append(
            {
                "lang": lang,
                "short_lang": lang.split("_")[0],
                "mono_premium": mono_p,
                "total_variation": tv,
                "density_index": JSON_DATA[lang]["density_index"],
            }
        )

    df = pd.DataFrame(rows)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

    # ----------------------------------------------------
    # Plot 1: Character Density Index vs Step Volatility (TV)
    # ----------------------------------------------------
    r1 = np.corrcoef(df["density_index"], df["total_variation"])[0, 1]
    sns.regplot(
        data=df,
        x="density_index",
        y="total_variation",
        scatter_kws={"s": 75, "color": "#d35400", "alpha": 0.85},
        line_kws={"color": "#2980b9", "linewidth": 2},
        ax=axes[0],
    )

    for _, row in df.iterrows():
        axes[0].text(
            row["density_index"] + 0.03,
            row["total_variation"] + 0.005,
            row["short_lang"],
            fontsize=9,
            weight="bold",
            alpha=0.85,
        )

    axes[0].set_title(
        f"(A) Density Index vs. Step Volatility (TV)\n$r = {r1:.3f}$ | $R^2 = {r1**2:.3f}$",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    axes[0].set_xlabel(
        "Character Density Index (English Baseline = 1.0)",
        fontsize=10,
        labelpad=10,
    )
    axes[0].set_ylabel(
        "Step Volatility / Total Variation $\\mathrm{TV}_H$",
        fontsize=10,
        labelpad=10,
    )

    # ----------------------------------------------------
    # Plot 2: Character Density Index vs Raw Monotonicity Premium
    # ----------------------------------------------------
    r2 = np.corrcoef(df["density_index"], df["mono_premium"])[0, 1]
    sns.regplot(
        data=df,
        x="density_index",
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
            row["density_index"] + 0.03,
            row["mono_premium"] + 0.005,
            row["short_lang"],
            fontsize=9,
            weight="bold",
            color=color,
            alpha=0.85,
        )

    axes[1].set_title(
        f"(B) Density Index vs. Raw Monotonicity Premium (T=0.6646)\n$r = {r2:.3f}$ | $R^2 = {r2**2:.3f}$",
        fontsize=12,
        weight="bold",
        pad=12,
    )
    axes[1].set_xlabel(
        "Character Density Index (English Baseline = 1.0)",
        fontsize=10,
        labelpad=10,
    )
    axes[1].set_ylabel(
        "Raw Monotonicity Premium Multiplier", fontsize=10, labelpad=10
    )

    plt.tight_layout()
    plt.savefig("density_index_analysis.png")
    print(f"Plot A (Density -> TV):          r = {r1:.4f}, R^2 = {r1**2:.4f}")
    print(f"Plot B (Density -> Mono Premium): r = {r2:.4f}, R^2 = {r2**2:.4f}")
    plt.show()


if __name__ == "__main__":
    plot_actual_density_relationships()