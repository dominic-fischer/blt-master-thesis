import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Language data from the Llama 2 paper (Table 10), tokens in billions
llama2_tokens = {
    'de': 3.4, 'fr': 3.2, 'sv': 3.0, 'zh': 2.6, 'es': 2.6,
    'ru': 2.6, 'nl': 2.4, 'it': 2.2, 'ja': 2.0, 'pl': 1.8,
    'pt': 1.8, 'vi': 1.6, 'uk': 1.4, 'ko': 1.2, 'ca': 0.8,
    'sr': 0.8, 'id': 0.6, 'cs': 0.6, 'fi': 0.6, 'hu': 0.6,
    'no': 0.6, 'ro': 0.6, 'bg': 0.4, 'da': 0.4, 'sl': 0.2, 'hr': 0.2
}

code_map = {
    'deu_Latn': 'de', 'fra_Latn': 'fr', 'swe_Latn': 'sv',
    'cmn_Hans': 'zh', 'cmn_Hant': 'zh', 'spa_Latn': 'es',
    'rus_Cyrl': 'ru', 'nld_Latn': 'nl', 'ita_Latn': 'it',
    'jpn_Jpan': 'ja', 'pol_Latn': 'pl', 'por_Latn': 'pt',
    'vie_Latn': 'vi', 'ukr_Cyrl': 'uk', 'kor_Hang': 'ko',
    'cat_Latn': 'ca', 'srp_Cyrl': 'sr', 'ind_Latn': 'id',
    'ces_Latn': 'cs', 'fin_Latn': 'fi', 'hun_Latn': 'hu',
    'nob_Latn': 'no', 'ron_Latn': 'ro', 'bul_Cyrl': 'bg',
    'dan_Latn': 'da', 'slv_Latn': 'sl', 'hrv_Latn': 'hr',
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
            iso = code_map[lang_code]
            tokens = llama2_tokens.get(iso)
            if tokens:
                try:
                    premium = float(parts[-1])
                    rows.append({'language': parts[0], 'code': iso,
                                 'tokens_B': tokens, 'premium': premium})
                except ValueError:
                    continue

df = pd.DataFrame(rows).drop_duplicates(subset='code')
df = df.sort_values('tokens_B')

# Log regression
log_tokens = np.log(df['tokens_B'])
slope, intercept, r, p, se = stats.linregress(log_tokens, df['premium'])
x_fit = np.linspace(df['tokens_B'].min(), df['tokens_B'].max(), 200)
y_fit = slope * np.log(x_fit) + intercept

fig, ax = plt.subplots(figsize=(9, 6))

ax.scatter(df['tokens_B'], df['premium'],
           color='steelblue', edgecolors='white',
           linewidths=0.6, s=70, zorder=3, alpha=0.85)

ax.plot(x_fit, y_fit, color='tomato', linewidth=1.8,
        linestyle='--', label=f'Log fit  (r={r:.2f}, p={p:.3f})')

for _, row in df.iterrows():
    ax.annotate(row['code'], (row['tokens_B'], row['premium']),
                fontsize=7.5, ha='left', va='bottom',
                xytext=(3, 3), textcoords='offset points', color='#444')

ax.set_xlabel('Training data in Llama 2 (billions of tokens)', fontsize=11)
ax.set_ylabel('Tokenisation premium vs. English', fontsize=11)
ax.set_title('More training data → lower tokenisation premium?', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_xscale('log')

plt.tight_layout()
plt.savefig('results/premium_vs_tokens.png', dpi=150, bbox_inches='tight')
plt.show()
print(f"\nPearson r (log tokens vs premium): {r:.3f}, p={p:.4f}")
print(f"N = {len(df)} languages")
print(df[['language', 'code', 'tokens_B', 'premium']].to_string(index=False))