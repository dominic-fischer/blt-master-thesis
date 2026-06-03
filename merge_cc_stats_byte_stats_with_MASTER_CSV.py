import pandas as pd

# ── Script metadata ──────────────────────────────────────────────────────────
SCRIPT_TIERS = {
    "Latn":1,
    "Cyrl":2,"Grek":2,"Armn":2,"Geor":2,"Nkoo":2,"Tfng":2,"Arab":2,"Hebr":2,
    "Deva":3,"Beng":3,"Guru":3,"Gujr":3,"Orya":3,"Taml":3,"Telu":3,"Knda":3,
    "Mlym":3,"Sinh":3,"Thai":3,"Laoo":3,"Tibt":3,"Mymr":3,"Khmr":3,"Ethi":3,
    "Olck":3,"Mtei":3,"Hans":3,"Hant":3,"Jpan":3,"Hang":3,
}
SCRIPT_TYPES = {
    "Latn":"Alphabet","Cyrl":"Alphabet","Grek":"Alphabet","Armn":"Alphabet",
    "Geor":"Alphabet","Nkoo":"Alphabet","Tfng":"Alphabet",
    "Arab":"Abjad","Hebr":"Abjad",
    "Deva":"Abugida","Beng":"Abugida","Guru":"Abugida","Gujr":"Abugida",
    "Orya":"Abugida","Taml":"Abugida","Telu":"Abugida","Knda":"Abugida",
    "Mlym":"Abugida","Sinh":"Abugida","Thai":"Abugida","Laoo":"Abugida",
    "Tibt":"Abugida","Mymr":"Abugida","Khmr":"Abugida","Ethi":"Abugida",
    "Olck":"Abugida","Mtei":"Abugida",
    "Hans":"Logographic","Hant":"Logographic","Jpan":"Logographic","Hang":"Alphabet",
}
TIER_LABELS = { 1:'1 byte', 2:'2 bytes', 3:'3 bytes' }
SCRIPT_NAMES = {
    "Arab":"Arabic","Armn":"Armenian","Beng":"Bengali","Cyrl":"Cyrillic",
    "Deva":"Devanagari","Ethi":"Ethiopic","Geor":"Georgian","Grek":"Greek",
    "Gujr":"Gujarati","Guru":"Gurmukhi","Hang":"Hangul","Hans":"Han (Simplified)",
    "Hant":"Han (Traditional)","Hebr":"Hebrew","Jpan":"Japanese","Khmr":"Khmer",
    "Knda":"Kannada","Laoo":"Lao","Latn":"Latin","Mlym":"Malayalam",
    "Mtei":"Meitei","Mymr":"Burmese","Nkoo":"N\u2019Ko","Olck":"Ol Chiki",
    "Orya":"Odia","Sinh":"Sinhala","Taml":"Tamil","Telu":"Telugu",
    "Tfng":"Tifinagh","Thai":"Thai","Tibt":"Tibetan"
}

# ── Override table (Flores code → ISO 639-2/B equivalent in CC) ─────────────
OVERRIDE = {
    "cmn": "zho",   # Mandarin
    "arb": "ara",   # Modern Standard Arabic
    "pes": "fas",   # Western Persian
    "zsm": "msa",   # Standard Malay
    "nob": "nor",   # Norwegian Bokmål
    "ekk": "est",   # Estonian
    "lvs": "lav",   # Standard Latvian
    "plt": "mlg",   # Plateau Malagasy
    "swh": "swa",   # Swahili
    "fuv": "ful",   # Nigerian Fulfulde
    "uzn": "uzb",   # Northern Uzbek
    "uzs": "uzb",   # Southern Uzbek
    "pbt": "pus",   # Southern Pashto
    "ory": "ori",   # Odia
    "gaz": "orm",   # West Central Oromo
    "kmr": "kur",   # Northern Kurdish
    "ckb": "kur",   # Central Kurdish
    "azj": "aze",   # North Azerbaijani
    "azb": "aze",   # South Azerbaijani
    "als": "sqi",   # Tosk Albanian
}

# ── Load data ────────────────────────────────────────────────────────────────
cc = pd.read_csv('commoncrawl_stats.csv')
fl = pd.read_csv('floresplus_MASTER_CSV_merged.csv')

# ── Aggregate CC by language (sum pages/urls across all crawls) ──────────────
cc_agg = (
    cc[cc['primary_language'] != '<unknown>']
    .groupby('primary_language', as_index=False)
    .agg(cc_pages=('pages', 'sum'), cc_urls=('urls', 'sum'))
)
cc_map = cc_agg.set_index('primary_language')

# ── Add script metadata columns ───────────────────────────────────────────────
fl['Script_Name']    = fl['Script'].map(SCRIPT_NAMES)
fl['Script_Type']    = fl['Script'].map(SCRIPT_TYPES)
fl['Script_Tier']    = fl['Script'].map(SCRIPT_TIERS)
fl['Bytes_Per_Char'] = fl['Script_Tier'].map(TIER_LABELS)

# ── Join CC data (applying overrides where needed) ────────────────────────────
def lookup_cc(row):
    cc_code = OVERRIDE.get(row['Code'], row['Code'])
    if cc_code in cc_map.index:
        return cc_map.loc[cc_code, 'cc_pages'], cc_map.loc[cc_code, 'cc_urls']
    return None, None

fl[['cc_pages', 'cc_urls']] = fl.apply(
    lambda r: pd.Series(lookup_cc(r)), axis=1
)

# Convert to nullable int
fl['cc_pages'] = pd.to_numeric(fl['cc_pages'], errors='coerce').astype('Int64')
fl['cc_urls']  = pd.to_numeric(fl['cc_urls'],  errors='coerce').astype('Int64')

# ── Save ──────────────────────────────────────────────────────────────────────
out = 'floresplus_MASTER_CSV_merged_enriched.csv'
fl.to_csv(out, index=False)

matched = fl['cc_pages'].notna().sum()
print(f"Saved {len(fl)} rows → {out}")
print(f"CC matched: {matched}/{len(fl)} rows ({len(fl)-matched} unmatched)")