import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from config import SCRIPT_LABELS, SCRIPT_COLORS

llama2_tokens = {
    'de': 3.4, 'fr': 3.2, 'sv': 3.0, 'zh': 2.6, 'es': 2.6,
    'ru': 2.6, 'nl': 2.4, 'it': 2.2, 'ja': 2.0, 'pl': 1.8,
    'pt': 1.8, 'vi': 1.6, 'uk': 1.4, 'ko': 1.2, 'ca': 0.8,
    'sr': 0.8, 'id': 0.6, 'cs': 0.6, 'fi': 0.6, 'hu': 0.6,
    'no': 0.6, 'ro': 0.6, 'bg': 0.4, 'da': 0.4, 'sl': 0.2, 'hr': 0.2
}

code_map = {
    'deu_Latn': ('de', 'Latn'), 'fra_Latn': ('fr', 'Latn'),
    'swe_Latn': ('sv', 'Latn'), 'cmn_Hans': ('zh', 'Hans'),
    'spa_Latn': ('es', 'Latn'), 'rus_Cyrl': ('ru', 'Cyrl'),
    'nld_Latn': ('nl', 'Latn'), 'ita_Latn': ('it', 'Latn'),
    'jpn_Jpan': ('ja', 'Jpan'), 'pol_Latn': ('pl', 'Latn'),
    'por_Latn': ('pt', 'Latn'), 'vie_Latn': ('vi', 'Latn'),
    'ukr_Cyrl': ('uk', 'Cyrl'), 'kor_Hang': ('ko', 'Hang'),
    'cat_Latn': ('ca', 'Latn'), 'srp_Cyrl': ('sr', 'Cyrl'),
    'ind_Latn': ('id', 'Latn'), 'ces_Latn': ('cs', 'Latn'),
    'fin_Latn': ('fi', 'Latn'), 'hun_Latn': ('hu', 'Latn'),
    'nob_Latn': ('no', 'Latn'), 'ron_Latn': ('ro', 'Latn'),
    'bul_Cyrl': ('bg', 'Cyrl'), 'dan_Latn': ('da', 'Latn'),
    'slv_Latn': ('sl', 'Latn'), 'hrv_Latn': ('hr', 'Latn'),
}

full_names = {
    'de': 'German', 'fr': 'French', 'sv': 'Swedish', 'zh': 'Chinese',
    'es': 'Spanish', 'ru': 'Russian', 'nl': 'Dutch', 'it': 'Italian',
    'ja': 'Japanese', 'pl': 'Polish', 'pt': 'Portuguese', 'vi': 'Vietnamese',
    'uk': 'Ukrainian', 'ko': 'Korean', 'ca': 'Catalan', 'sr': 'Serbian',
    'id': 'Indonesian', 'cs': 'Czech', 'fi': 'Finnish', 'hu': 'Hungarian',
    'no': 'Norwegian', 'ro': 'Romanian', 'bg': 'Bulgarian', 'da': 'Danish',
    'sl': 'Slovenian', 'hr': 'Croatian',
}

rows = []
with open('results/summary.txt', 'r') as f:
    for line in f:
        parts = line.split()
        if len(parts) < 3 or parts[0] in ('Language', '---') or line.startswith('-'):
            continue
        lang_code = None
        for field in parts:
            if '_' in field and not field.startswith('-'):
                lang_code = field
                break
        if lang_code and lang_code in code_map:
            iso, script_key = code_map[lang_code]
            tokens = llama2_tokens.get(iso)
            if tokens:
                try:
                    premium = float(parts[-1])
                    script_label = SCRIPT_LABELS.get(script_key, script_key)
                    color = SCRIPT_COLORS.get(script_label, '#333333')
                    rows.append({
                        'language': full_names.get(iso, iso),
                        'code': iso,
                        'tokens_B': tokens,
                        'premium': premium,
                        'script': script_label,
                        'color': color,
                    })
                except ValueError:
                    continue

df = pd.DataFrame(rows).drop_duplicates(subset='code')
df = df.sort_values('tokens_B')

# Log regression
log_tokens = np.log(df['tokens_B'])
slope, intercept, r, p, se = stats.linregress(log_tokens, df['premium'])
x_fit = np.linspace(df['tokens_B'].min(), df['tokens_B'].max(), 200)
y_fit = slope * np.log(x_fit) + intercept

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#f8f9fa")
ax.set_facecolor("#f8f9fa")

for _, row in df.iterrows():
    ax.scatter(row['tokens_B'], row['premium'],
               color=row['color'], edgecolors='white',
               linewidths=0.6, s=75, zorder=3, alpha=0.9)
    ax.annotate(row['language'], (row['tokens_B'], row['premium']),
                fontsize=6.5, ha='left', va='bottom',
                xytext=(4, 3), textcoords='offset points', color='#333')

ax.plot(x_fit, y_fit, color='tomato', linewidth=1.8,
        linestyle='--', label=f'Log fit  (r={r:.2f}, p={p:.3f})', zorder=2)

# script colour legend
present_scripts = df['script'].unique()
script_handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=SCRIPT_COLORS[s], markersize=8, label=s)
    for s in present_scripts if s in SCRIPT_COLORS
]
script_handles.append(
    plt.Line2D([0], [1], color='tomato', linewidth=1.8,
               linestyle='--', label=f'Log fit (r={r:.2f}, p={p:.3f})')
)
ax.legend(handles=script_handles, title='Script', fontsize=8,
          title_fontsize=9, loc='lower left', framealpha=0.9)

ax.set_xlabel('Training data in Llama 2 (billions of tokens)', fontsize=11)
ax.set_ylabel('Tokenisation premium vs. English', fontsize=11)
ax.set_title('BLT Patch Premium vs. Llama 2 Training Data',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color("#aaaaaa")

plt.tight_layout()
plt.savefig('results/premium_vs_llama2_data.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Pearson r (log tokens vs premium): {r:.3f}, p={p:.4f}, N={len(df)}")