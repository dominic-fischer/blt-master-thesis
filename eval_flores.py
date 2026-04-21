"""
eval_flores.py
Run BLT patching over a diverse subset of FLORES+ languages and save results to JSON.

Usage:
    python eval_flores.py

Output: one JSON file per language at results/{lang_code}.json
"""

import json
import os
from datasets import load_dataset
from tqdm import tqdm
import subprocess

from blt_patcher import load_patcher, patch_text


# ── config ────────────────────────────────────────────────────────────────────
LIMIT = None # set to an int for quick testing

REPO = "facebook/blt-1b"
ENTROPY_REPO = "hf-weights/entropy_model"
SPLIT = "dev"
OUTPUT_DIR = "results"

FLORES_DATASET = "openlanguagedata/flores_plus"

LANGUAGES = {
    "ace_Arab": "Acehnese (Arabic)",
    "ace_Latn": "Acehnese (Latin)",
    "afr_Latn": "Afrikaans",
    "twi_Latn_akua1239": "Akuapem Twi",
    "amh_Ethi": "Amharic",
    "arg_Latn": "Aragonese",
    "oci_Latn_aran1260": "Aranese Occitan",
    "hye_Armn": "Armenian",
    "twi_Latn_asan1239": "Asante Twi",
    "asm_Beng": "Assamese",
    "ast_Latn": "Asturian",
    "awa_Deva": "Awadhi",
    "quy_Latn": "Ayacucho Quechua",
    "ban_Latn": "Balinese",
    "bam_Latn": "Bambara",
    "bjn_Arab": "Banjar (Arabic)",
    "bjn_Latn": "Banjar (Latin)",
    "bak_Cyrl": "Bashkir",
    "eus_Latn": "Basque",
    "bel_Cyrl": "Belarusian",
    "bem_Latn": "Bemba",
    "ben_Beng": "Bengali",
    "bho_Deva": "Bhojpuri",
    "brx_Deva": "Bodo",
    "bos_Latn": "Bosnian",
    "bug_Latn": "Buginese",
    "bul_Cyrl": "Bulgarian",
    "mya_Mymr": "Burmese",
    "yue_Hant": "Cantonese",
    "cat_Latn": "Catalan",
    "ceb_Latn": "Cebuano",
    "ayr_Latn": "Central Aymara",
    "knc_Arab": "Central Kanuri (Arabic)",
    "knc_Latn": "Central Kanuri (Latin)",
    "ckb_Arab": "Central Kurdish",
    "hne_Deva": "Chhattisgarhi",
    "nya_Latn": "Chichewa",
    "cmn_Hans": "Chinese (Simplified)",
    "cmn_Hant": "Chinese (Traditional)",
    "cjk_Latn": "Chokwe",
    "chv_Cyrl": "Chuvash",
    "crh_Latn": "Crimean Tatar",
    "hrv_Latn": "Croatian",
    "ces_Latn": "Czech",
    "dan_Latn": "Danish",
    "dar_Cyrl": "Dargwa",
    "prs_Arab": "Dari",
    "dgo_Deva": "Dogri",
    "nld_Latn": "Dutch",
    "dyu_Latn": "Dyula",
    "dzo_Tibt": "Dzongkha",
    "mhr_Cyrl": "Eastern Mari",
    "ydd_Hebr": "Eastern Yiddish",
    "arz_Arab": "Egyptian Arabic",
    "eng_Latn": "English",
    "myv_Cyrl": "Erzya",
    "epo_Latn": "Esperanto",
    "ekk_Latn": "Estonian",
    "ewe_Latn": "Ewe",
    "fao_Latn": "Faroese",
    "fij_Latn": "Fijian",
    "fil_Latn": "Filipino",
    "fin_Latn": "Finnish",
    "fon_Latn": "Fon",
    "fra_Latn": "French",
    "fur_Latn": "Friulian",
    "glg_Latn": "Galician",
    "lug_Latn": "Ganda",
    "kat_Geor": "Georgian",
    "deu_Latn": "German",
    "gom_Deva": "Goan Konkani",
    "ell_Grek": "Greek",
    "guj_Gujr": "Gujarati",
    "hat_Latn": "Haitian Creole",
    "khk_Cyrl": "Halh Mongolian (Cyrillic)",
    "khk_Mong": "Halh Mongolian (Traditional)",
    "hau_Latn": "Hausa",
    "heb_Hebr": "Hebrew",
    "hin_Deva": "Hindi",
    "hun_Latn": "Hungarian",
    "isl_Latn": "Icelandic",
    "ibo_Latn": "Igbo",
    "ilo_Latn": "Ilocano",
    "ind_Latn": "Indonesian",
    "gle_Latn": "Irish",
    "ita_Latn": "Italian",
    "jpn_Jpan": "Japanese",
    "jav_Latn": "Javanese",
    "kac_Latn": "Jingpho",
    "kbp_Latn": "Kabiyè",
    "kea_Latn": "Kabuverdianu",
    "kab_Latn": "Kabyle",
    "kam_Latn": "Kamba",
    "kan_Knda": "Kannada",
    "ktu_Latn": "Kanuri (Latin)",
    "kaa_Latn": "Kara-Kalpak",
    "kas_Arab": "Kashmiri (Arabic)",
    "kas_Deva": "Kashmiri (Devanagari)",
    "kaz_Cyrl": "Kazakh",
    "kjh_Cyrl": "Khakas",
    "khm_Khmr": "Khmer",
    "kik_Latn": "Kikuyu",
    "kmb_Latn": "Kimbundu",
    "kin_Latn": "Kinyarwanda",
    "kor_Hang": "Korean",
    "kir_Cyrl": "Kyrgyz",
    "lld_Latn": "Ladin",
    "lld_Latn_gard1241": "Ladin (Gardena)",
    "lao_Laoo": "Lao",
    "ltg_Latn": "Latgalian",
    "lvs_Latn": "Latvian",
    "lij_Latn": "Ligurian",
    "lim_Latn": "Limburgish",
    "lin_Latn": "Lingala",
    "lit_Latn": "Lithuanian",
    "lmo_Latn": "Lombard",
    "lua_Latn": "Luba-Kasai",
    "luo_Latn": "Luo",
    "ltz_Latn": "Luxembourgish",
    "mkd_Cyrl": "Macedonian",
    "mag_Deva": "Magahi",
    "mai_Deva": "Maithili",
    "vmw_Latn": "Makhuwa",
    "mal_Mlym": "Malayalam",
    "mlt_Latn": "Maltese",
    "mar_Deva": "Marathi",
    "mni_Beng": "Meitei (Bengali)",
    "mni_Mtei": "Meitei (Meitei Mayek)",
    "acm_Arab": "Mesopotamian Arabic",
    "min_Arab": "Minangkabau (Arabic)",
    "min_Latn": "Minangkabau (Latin)",
    "lus_Latn": "Mizo",
    "arb_Arab": "Modern Standard Arabic",
    "arb_Latn": "Modern Standard Arabic (Latin)",
    "mfe_Latn": "Morisyen",
    "ary_Arab": "Moroccan Arabic",
    "mos_Latn": "Mossi",
    "mri_Latn": "Māori",
    "nqo_Nkoo": "N'Ko",
    "ars_Arab": "Najdi Arabic",
    "npi_Deva": "Nepali",
    "fuv_Latn": "Nigerian Fulfulde",
    "azj_Latn": "North Azerbaijani",
    "apc_Arab_nort3139": "North Levantine Arabic",
    "kmr_Latn": "Northern Kurdish",
    "nso_Latn": "Northern Sotho",
    "uzn_Latn": "Northern Uzbek",
    "nob_Latn": "Norwegian Bokmål",
    "nob_Latn_radical": "Norwegian Bokmål (Radical)",
    "nno_Latn": "Norwegian Nynorsk",
    "nus_Latn": "Nuer",
    "oci_Latn": "Occitan",
    "ory_Orya": "Odia",
    "pag_Latn": "Pangasinan",
    "pap_Latn": "Papiamento",
    "gug_Latn": "Paraguayan Guaraní",
    "plt_Latn": "Plateau Malagasy",
    "pol_Latn": "Polish",
    "por_Latn": "Portuguese",
    "pan_Guru": "Punjabi",
    "ron_Latn": "Romanian",
    "run_Latn": "Rundi",
    "rus_Cyrl": "Russian",
    "smo_Latn": "Samoan",
    "sag_Latn": "Sango",
    "san_Deva": "Sanskrit",
    "sat_Olck": "Santali",
    "srd_Latn": "Sardinian",
    "gla_Latn": "Scottish Gaelic",
    "srp_Cyrl": "Serbian",
    "shn_Mymr": "Shan",
    "sna_Latn": "Shona",
    "scn_Latn": "Sicilian",
    "szl_Latn": "Silesian",
    "snd_Arab": "Sindhi (Arabic)",
    "snd_Deva": "Sindhi (Devanagari)",
    "sin_Sinh": "Sinhala",
    "slk_Latn": "Slovak",
    "slv_Latn": "Slovenian",
    "som_Latn": "Somali",
    "azb_Arab": "South Azerbaijani",
    "apc_Arab_sout3123": "South Levantine Arabic",
    "pbt_Arab": "Southern Pashto",
    "sot_Latn": "Southern Sotho",
    "uzs_Arab": "Southern Uzbek",
    "dik_Latn": "Southwestern Dinka",
    "spa_Latn": "Spanish",
    "zsm_Latn": "Standard Malay",
    "zgh_Tfng": "Standard Moroccan Tamazight",
    "sun_Latn": "Sundanese",
    "swh_Latn": "Swahili",
    "ssw_Latn": "Swati",
    "swe_Latn": "Swedish",
    "acq_Arab": "Taizzi-Adeni Arabic",
    "tgk_Cyrl": "Tajik",
    "taq_Latn": "Tamasheq (Latin)",
    "taq_Tfng": "Tamasheq (Tifinagh)",
    "tam_Taml": "Tamil",
    "tat_Cyrl": "Tatar",
    "tel_Telu": "Telugu",
    "tha_Thai": "Thai",
    "bod_Tibt": "Tibetan",
    "tir_Ethi": "Tigrinya",
    "tpi_Latn": "Tok Pisin",
    "als_Latn": "Tosk Albanian",
    "tso_Latn": "Tsonga",
    "tsn_Latn": "Tswana",
    "tum_Latn": "Tumbuka",
    "aeb_Arab": "Tunisian Arabic",
    "tur_Latn": "Turkish",
    "tuk_Latn": "Turkmen",
    "tyv_Cyrl": "Tuvinian",
    "ukr_Cyrl": "Ukrainian",
    "umb_Latn": "Umbundu",
    "urd_Arab": "Urdu",
    "uig_Arab": "Uyghur",
    "cat_Latn_vale1252": "Valencian",
    "vec_Latn": "Venetian",
    "vie_Latn": "Vietnamese",
    "war_Latn": "Waray",
    "cym_Latn": "Welsh",
    "gaz_Latn": "West Central Oromo",
    "pes_Arab": "Western Persian",
    "wol_Latn": "Wolof",
    "wuu_Hans": "Wu Chinese",
    "xho_Latn": "Xhosa",
    "yor_Latn": "Yoruba",
    "zul_Latn": "Zulu",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_language(lang_code: str):
    return load_dataset(FLORES_DATASET, lang_code, split=SPLIT)


def run_language(lang_code: str, lang_name: str, eng_dataset, tokenizer, patcher):
    print(f"\nProcessing {lang_name} ({lang_code})...")

    if lang_code == "eng_Latn":
        dataset = eng_dataset
    else:
        try:
            dataset = load_language(lang_code)
        except Exception as e:
            print(f"  Skipping: {e}")
            return None

    assert len(dataset) == len(eng_dataset), (
        f"Length mismatch for {lang_code}: {len(dataset)} vs {len(eng_dataset)}"
    )

    results = []
    for i, (row, eng_row) in enumerate(
        tqdm(zip(dataset, eng_dataset), total=len(dataset), desc=lang_name)
    ):
        if LIMIT is not None and i >= LIMIT:
            break

        text = row["text"]
        eng_text = eng_row["text"]
        result = patch_text(text, tokenizer, patcher)

        results.append({
            "id": i,
            "text": text,
            "text_bytes": result["text_bytes"],
            "eng_text": eng_text,
            "n_patches": result["n_patches"],
            "n_bytes": result["n_bytes"],
            "avg_bytes_per_patch": result["avg_bytes_per_patch"],
            "patches": [
                {"bytes": p[0], "length": p[1]}
                for p in result["patches"]
            ],
            "scores": result["scores"],
        })

    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading patcher...")
    tokenizer, patcher = load_patcher(repo=REPO, entropy_repo=ENTROPY_REPO)

    print(f"Loading English ({SPLIT})...")
    eng_dataset = load_language("eng_Latn")

    for lang_code, lang_name in LANGUAGES.items():
        results = run_language(lang_code, lang_name, eng_dataset, tokenizer, patcher)

        if results is None:
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{lang_code}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  Saved {len(results)} entries → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
    # also run summarize_results.py
    subprocess.run(["python", "summarize_results.py"])