import numpy as np, matplotlib.pyplot as plt
from scipy import stats
from adjustText import adjust_text

import os, json

# --- inputs ------------------------------------------------------------
UTF8_FILE   = os.path.join("results", "txt_premiums", "t_anchor", "Balanced", "step_0000007200",
                           "Balanced_cumulative_nomono_t_3.9348_premiums_sorted.txt")
CUSTOM_FILE = os.path.join("results", "txt_premiums", "t_anchor", "Balanced-Custom", "step_0000006400",
                           "Balanced-Custom_cumulative_nomono_t_4.2619_premiums_sorted.txt")
OUT_PNG    = "entropy_delta_vs_byte_delta.png"

# JSON with per-language factors; we use "avg_length" (avg. UTF-8 bytes per char)
AVG_LEN_JSON = "avg_length_and_density.json"
OUT_TEX      = "entropy_delta_table.tex"
CUSTOM_BYTES = 2.0

NAMES = {
    "cmn_Hans": "Mandarin Chinese", "tam_Taml": "Tamil", "tha_Thai": "Thai",
    "arb_Arab": "Arabic", "srp_Cyrl": "Serbian", "deu_Latn": "German",
    "ita_Latn": "Italian", "spa_Latn": "Spanish", "fin_Latn": "Finnish",
    "ron_Latn": "Romanian", "amh_Ethi": "Amharic", "fra_Latn": "French",
    "vie_Latn": "Vietnamese", "jpn_Jpan": "Japanese", "kor_Hang": "Korean",
    "hrv_Latn": "Croatian", "kat_Geor": "Georgian", "heb_Hebr": "Hebrew",
    "nya_Latn": "Chichewa", "eng_Latn": "English",
}

def read_premiums(path):
    """Parse a whitespace-separated premiums table into {lang: {column: float}}."""
    rows, header = {}, None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if header is None:
                header = parts
                continue
            rows[parts[0]] = {h: float(v) for h, v in zip(header[1:], parts[1:])}
    return rows

with open(AVG_LEN_JSON, encoding="utf-8") as f:
    AVG_LEN = {lang: v["avg_length"] for lang, v in json.load(f).items()}

utf8   = read_premiums(UTF8_FILE)
custom = read_premiums(CUSTOM_FILE)

langs = [l for l in utf8 if l in custom and l in AVG_LEN]
missing = (set(utf8) | set(custom)) - set(langs)
if missing:
    print("skipping (missing in one of the inputs):", sorted(missing))

d = {l: (NAMES.get(l, l), AVG_LEN[l], utf8[l]["EntropyTotal"], custom[l]["EntropyTotal"])
     for l in langs}
names=[v[0] for v in d.values()]
x=np.array([CUSTOM_BYTES-v[1] for v in d.values()])
y=np.array([(v[3]-v[2])/v[2]*100 for v in d.values()])
grp=np.array([{"Latn":1,"Arab":2,"Cyrl":2,"Hebr":2}.get(k.split("_")[1],3) for k in d])

cols={1:"#4a68ad",2:"#5c9e5c",3:"#c9b672"}
plt.rcParams.update({"font.family":"DejaVu Sans"})
fig,ax=plt.subplots(figsize=(11,8.2),dpi=200)
ax.grid(True,ls="--",color="#dddddd",lw=0.8,zorder=0)
for g in (1,2,3):
    m=grp==g
    ax.scatter(x[m],y[m],s=190,c=cols[g],edgecolors="#333333",lw=1,zorder=3,
               label=f"{g} (n={m.sum()})")

r,p=stats.pearsonr(x,y)
b,a=np.polyfit(x,y,1); xs=np.linspace(x.min(),x.max(),100)
ax.plot(xs,a+b*xs,"--",color="#e8352b",lw=2.2,zorder=2,
        label=f"Linear fit: r = {r:.2f}, r$^2$ = {r**2:.2f}, p = {p:.2g}")

latin=[(xi,yi,n) for xi,yi,n,g in zip(x,y,names,grp) if g==1 and n!="Vietnamese"]
latin.sort(key=lambda t:-t[1])
ys=np.linspace(latin[0][1]+1.2,latin[-1][1]+0.3,len(latin))
for (xi,yi,n),ly in zip(latin,ys):
    ax.annotate(n,(xi,yi),xytext=(0.86,ly),fontsize=11.5,color="#333333",va="center",ha="right",zorder=4,
                arrowprops=dict(arrowstyle="-",color="#999999",lw=0.9,shrinkB=7))
texts=[ax.text(xi,yi,n,fontsize=11.5,color="#333333",zorder=4) for xi,yi,n,g in zip(x,y,names,grp) if not (g==1)]
ax.annotate("Vietnamese",(x[names.index("Vietnamese")],y[names.index("Vietnamese")]),xytext=(0.66,1.9),fontsize=11.5,color="#333333",ha="center",zorder=4,arrowprops=dict(arrowstyle="-",color="#999999",lw=0.9,shrinkB=7))
adjust_text(texts,x=x,y=y,ax=ax,expand=(1.6,2.0),force_text=(0.6,0.9),force_static=(0.5,0.8),
            arrowprops=dict(arrowstyle="-",color="#999999",lw=0.9))

ax.set_xlabel("Byte delta (custom − UTF-8 bytes per char)",fontsize=15)
ax.set_ylabel("Δ entropy total (custom vs. UTF-8, %)",fontsize=15)
ax.tick_params(labelsize=13)
for s in ax.spines.values(): s.set_color("black")
fig.suptitle("Entropy total change vs. byte delta",fontsize=19,fontweight="bold",y=0.975)
ax.set_title("Custom 2-byte encoding vs. UTF-8, measured UTF-8 bytes per char",fontsize=13,color="#666666",pad=12)
leg=ax.legend(title="Bytes per char (UTF-8)",loc="lower right",fontsize=11.5,title_fontsize=12.5,
              frameon=True,framealpha=0.95,edgecolor="#cccccc",alignment="left")
ax.set_ylim(y.min()-2.5,y.max()+1.8); ax.set_xlim(-1.05,1.1)
fig.tight_layout()
fig.savefig(OUT_PNG)

# --- LaTeX table -------------------------------------------------------
def fmt_signed(v, nd):
    s = f"{abs(v):.{nd}f}"
    return f"$+${s}" if v > 0 else (f"$-${s}" if v < 0 else s)

lines = [
    r"\begin{table}[ht]",
    r"\centering",
    r"\small",
    r"\setlength{\tabcolsep}{5pt}",
    r"\begin{tabular}{lrrrrrr}",
    r"\toprule",
    r" & \multicolumn{2}{c}{Bytes/char} & & \multicolumn{2}{c}{Entropy total} & \\",
    r"\cmidrule(lr){2-3} \cmidrule(lr){5-6}",
    r"Language & UTF-8 & Custom & Byte $\Delta$ & UTF-8 & Custom & $\Delta$ (\%) \\",
    r"\midrule",
]
for lang in langs:
    _, al, eu, ec = d[lang]
    lines.append(f"{NAMES.get(lang, lang)} & {al:.2f} & {CUSTOM_BYTES:.0f} & {fmt_signed(CUSTOM_BYTES-al,2)} & "
                 f"{eu:,.0f} & {ec:,.0f} & {fmt_signed((ec-eu)/eu*100,2)} \\\\".replace(",", "{,}"))
tot_u = sum(d[l][2] for l in langs); tot_c = sum(d[l][3] for l in langs)
mean_al = np.mean([d[l][1] for l in langs])
mean_pct = np.mean([(d[l][3]-d[l][2])/d[l][2]*100 for l in langs])
tot_pct = (tot_c-tot_u)/tot_u*100

# spread of entropy totals across languages: how much higher the highest
# total is than the lowest, in %
rng_u = (max(d[l][2] for l in langs) / min(d[l][2] for l in langs) - 1) * 100
rng_c = (max(d[l][3] for l in langs) / min(d[l][3] for l in langs) - 1) * 100

lines += [
    r"\midrule",
    f"Mean & {mean_al:.2f} & {CUSTOM_BYTES:.0f} & {fmt_signed(CUSTOM_BYTES-mean_al,2)} & & & {fmt_signed(mean_pct,2)} \\\\",
    f"Overall & & & & {tot_u:,.0f} & {tot_c:,.0f} & \\\\".replace(",", "{,}"),
    f"$\\Delta$ (max vs.\\ min) & & & & {rng_u:.1f}\\,\\% & {rng_c:.1f}\\,\\% & \\\\",
    r"\bottomrule",
    r"\end{tabular}",
    r"\caption{Entropy totals under UTF-8 and the custom 2-byte encoding. Byte $\Delta$ is custom minus UTF-8 bytes per character; $\Delta$\,(\%) is the custom entropy total relative to UTF-8. $\Delta$\,(max vs.\ min) is how much higher the highest entropy total is than the lowest.}",
    r"\label{tab:entropy-delta}",
    r"\end{table}",
]
with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"saved {OUT_TEX}")
print(f"saved {OUT_PNG}  (n={len(x)}, r={r:.3f}, p={p:.2g})")
print(f"entropy-total spread (max vs. min): UTF-8 {rng_u:.1f}%, custom {rng_c:.1f}%")