#!/usr/bin/env python3
"""
word_length_stats.py -- Character counts, word counts and average word length
per language, from the same BLT `bytes_entropies` JSON dumps used by
byte_position_stats.py (results/base_model/{lang_script}.json).

Only the "text" field of each record is used -- no byte values, no entropies.
The results are therefore identical for UTF-8 and custom-encoding runs.

WHAT IS MEASURED
    n_char           all Unicode codepoints (incl. whitespace/punctuation);
                     under the fixed 2-byte custom encoding, this is exactly
                     half the byte count the model sees
    n_char_word      codepoints that are letters, marks or numbers
                     (Unicode categories L*, M*, N*)
    n_whitespace     whitespace codepoints
    n_words[m]       number of words under segmentation method m
    avg_word_length[m]
                     micro-average: total word characters / total words.
                     Word length counts only L/M/N codepoints, so punctuation
                     glued to a token ("word,") and joiners (pyvi's "_")
                     are ignored identically for every method. Combining
                     marks (Thai/Tamil vowel signs, Hebrew niqqud, ...) DO
                     count, since the model sees them as characters.
    word_char_coverage[m]
                     sanity check: word characters recovered by method m
                     divided by n_char_word. Should be ~1.0; anything else
                     means the segmenter dropped or altered characters.
    rel_to_eng       ratios to English (only if eng_Latn was processed and
                     the record ids match, i.e. the texts are parallel):
                       n_char, n_char_word  -> density
                       n_words[m]           -> chunking; always relative
                                               to ENGLISH WHITESPACE words

    Tokens without a single L/M/N codepoint (pure punctuation/symbols) are
    dropped before counting, for every method alike.

SEGMENTATION METHODS
    "whitespace" is computed for every language. For cmn_Hans, jpn_Jpan
    and tha_Thai it is NOT a meaningful word count (flagged via
    "has_whitespace_words": false) but is reported anyway.

    Additional segmenters (each optional -- a segmenter whose package is
    missing or that fails a probe sentence is skipped and listed under
    "segmenters_failed"):
      cmn_Hans : jieba, pkuseg (spacy-pkuseg or pkuseg), thulac, icu
      jpn_Jpan : ginza_bunsetsu -- BUNSETSU units (content word + attached
                 particles/endings), the unit Korean marks with spaces.
                 The short-unit word segmenters (fugashi_unidic,
                 janome_ipadic, sudachi_A/B/C, nagisa, icu) are commented
                 out in SEGMENTERS, since they split off every particle and
                 ending; re-enable them for a comparison.
      tha_Thai : pythainlp_newmm, pythainlp_longest, pythainlp_attacut,
                 pythainlp_deepcut, pythainlp_nlpo3, icu
      vie_Latn : pyvi, underthesea (Vietnamese spaces separate SYLLABLES;
                 these join multi-syllable words, so you can compare
                 orthographic vs lexical chunking)
      + stanza for cmn_Hans / vie_Latn with --stanza
        (downloads models on first use)

INSTALL (everything is optional; install what you want to compare)
    pip install jieba spacy-pkuseg thulac
    pip install ginza ja_ginza
    pip install fugashi unidic-lite janome sudachipy sudachidict_core nagisa  # disabled by default
    pip install pythainlp attacut deepcut nlpo3
    pip install pyvi underthesea
    pip install PyICU      # needs system ICU: apt install libicu-dev pkg-config
    pip install stanza
    Note: thulac and nagisa are unmaintained and may fail on recent Python
    versions; deepcut needs TensorFlow. The script simply skips them.

USAGE
    python3 word_length_stats.py --only-20 results/base_model/
    python3 word_length_stats.py results/base_model/cmn_Hans.json results/base_model/eng_Latn.json
    python3 word_length_stats.py --stanza --normalize NFC --out-json wl.json results/base_model/
"""

import argparse
import json
import os
import sys
import unicodedata
from importlib import metadata

OUR_20_LANGS = {
    "eng_Latn", "cmn_Hans", "deu_Latn", "jpn_Jpan", "spa_Latn", "fra_Latn",
    "ita_Latn", "vie_Latn", "arb_Arab", "tha_Thai", "kor_Hang", "ron_Latn",
    "fin_Latn", "heb_Hebr", "tam_Taml", "hrv_Latn", "srp_Cyrl", "kat_Geor",
    "amh_Ethi", "nya_Latn",
}
NO_WHITESPACE_WORDS = {"cmn_Hans", "jpn_Jpan", "tha_Thai"}

# Short sentences used to check that a segmenter actually works before
# running it on the full data.
PROBE = {
    "cmn_Hans": "我们今天去北京参观博物馆。",
    "jpn_Jpan": "今日は東京の博物館に行きました。",
    "tha_Thai": "วันนี้อากาศดีมากและเราไปพิพิธภัณฑ์",
    "vie_Latn": "Hà Nội là thủ đô của Việt Nam.",
}


# ---------------------------------------------------------------------------
# Word definition (shared by every method)
# ---------------------------------------------------------------------------

def is_word_char(ch):
    return unicodedata.category(ch)[0] in "LMN"


def word_length(token):
    return sum(1 for ch in token if is_word_char(ch))


def clean_tokens(tokens):
    """Drop tokens containing no letter/mark/number."""
    return [t for t in tokens if word_length(t) > 0]


def chunks(text, max_bytes):
    """Split text into pieces of at most ~max_bytes UTF-8 bytes, preferring
    line breaks (for tools with input-length limits, e.g. Sudachi)."""
    if len(text.encode("utf-8")) <= max_bytes:
        yield text
        return
    buf = ""
    for line in text.splitlines(keepends=True):
        while len(line.encode("utf-8")) > max_bytes:  # pathological long line
            cut = max_bytes // 4
            yield line[:cut]
            line = line[cut:]
        if buf and len((buf + line).encode("utf-8")) > max_bytes:
            yield buf
            buf = ""
        buf += line
    if buf:
        yield buf


# ---------------------------------------------------------------------------
# Segmenter factories. Each returns a function: text -> list[str].
# Imports happen inside the factories so missing packages only skip that
# one segmenter.
# ---------------------------------------------------------------------------

def make_jieba():
    import jieba
    jieba.setLogLevel(60)
    jieba.initialize()
    return lambda t: jieba.lcut(t)


def make_pkuseg():
    try:
        import spacy_pkuseg as pkuseg
    except ImportError:
        import pkuseg
    try:
        seg = pkuseg.pkuseg()
    except Exception:
        seg = pkuseg.pkuseg(model_name="mixed")
    return seg.cut


def make_thulac():
    import thulac
    th = thulac.thulac(seg_only=True)
    return lambda t: th.cut(t, text=True).split()


def make_icu(locale):
    def factory():
        from icu import BreakIterator, Locale
        bi = BreakIterator.createWordInstance(Locale(locale))

        def seg(text):
            # ICU offsets are UTF-16 code units; map them to Python indices
            # so characters outside the BMP don't shift the slices.
            idx, off = {}, 0
            for i, ch in enumerate(text):
                idx[off] = i
                off += 2 if ord(ch) > 0xFFFF else 1
            idx[off] = len(text)
            bi.setText(text)
            out = []
            start = bi.first()
            end = bi.nextBoundary()
            while end != -1:  # BreakIterator.DONE
                out.append(text[idx[start]:idx[end]])
                start, end = end, bi.nextBoundary()
            return out
        return seg
    return factory


def make_fugashi():
    import fugashi
    tagger = fugashi.Tagger()
    return lambda t: [w.surface for w in tagger(t)]


def make_janome():
    from janome.tokenizer import Tokenizer
    tk = Tokenizer()
    return lambda t: list(tk.tokenize(t, wakati=True))


def make_sudachi(mode_letter):
    def factory():
        from sudachipy import Dictionary, SplitMode
        d, mode = Dictionary(), getattr(SplitMode, mode_letter)
        tok = d.tokenizer(mode=mode) if hasattr(d, "tokenizer") else d.create(mode=mode)

        def seg(text):
            out = []
            for piece in chunks(text, 40000):  # Sudachi's input limit is ~49k bytes
                out.extend(m.surface() for m in tok.tokenize(piece))
            return out
        return seg
    return factory


def make_ginza_bunsetsu():
    """Japanese BUNSETSU segmentation (content word + attached particles /
    auxiliaries / inflectional endings), via GiNZA. This is the unit Korean
    marks with spaces (eojeol), so it makes Japanese comparable to Korean,
    unlike the short-unit word segmenters, which split off every particle
    and ending."""
    import spacy
    import ginza
    nlp = spacy.load("ja_ginza")

    def seg(text):
        out = []
        for piece in chunks(text, 40000):  # GiNZA tokenizes with Sudachi (~49k-byte limit)
            doc = nlp(piece)
            for sent in doc.sents:
                out.extend(span.text for span in ginza.bunsetu_spans(sent))
        return out
    return seg


def make_nagisa():
    import nagisa
    return lambda t: nagisa.tagging(t).words


def make_pythainlp(engine):
    def factory():
        from pythainlp.tokenize import word_tokenize
        return lambda t: word_tokenize(t, engine=engine, keep_whitespace=False)
    return factory


def make_pyvi():
    from pyvi import ViTokenizer
    return lambda t: ViTokenizer.tokenize(t).split()


def make_underthesea():
    from underthesea import word_tokenize
    return lambda t: word_tokenize(t)


def make_stanza(lang):
    def factory():
        import stanza
        nlp = stanza.Pipeline(lang, processors="tokenize", verbose=False)
        return lambda t: [tok.text for s in nlp(t).sentences for tok in s.tokens]
    return factory


# name, factory, packages whose versions are recorded
SEGMENTERS = {
    "cmn_Hans": [
        ("jieba", make_jieba, ["jieba"]),
        ("pkuseg", make_pkuseg, ["spacy-pkuseg", "pkuseg"]),
        ("thulac", make_thulac, ["thulac"]),
        ("icu", make_icu("zh"), ["PyICU"]),
    ],
    "jpn_Jpan": [
        # Bunsetsu level, comparable to Korean eojeol (whitespace units).
        ("ginza_bunsetsu", make_ginza_bunsetsu, ["ginza", "ja-ginza"]),
        # Short-unit WORD segmenters, disabled for now: they split off every
        # particle, auxiliary and inflectional ending, which makes Japanese
        # incomparable to Korean. Re-enable to report them for comparison.
        # ("fugashi_unidic", make_fugashi, ["fugashi", "unidic-lite", "unidic"]),
        # ("janome_ipadic", make_janome, ["janome"]),
        # ("sudachi_A", make_sudachi("A"), ["SudachiPy", "SudachiDict-core"]),
        # ("sudachi_B", make_sudachi("B"), ["SudachiPy", "SudachiDict-core"]),
        # ("sudachi_C", make_sudachi("C"), ["SudachiPy", "SudachiDict-core"]),
        # ("nagisa", make_nagisa, ["nagisa"]),
        # ("icu", make_icu("ja"), ["PyICU"]),
    ],
    "tha_Thai": [
        ("pythainlp_newmm", make_pythainlp("newmm"), ["pythainlp"]),
        ("pythainlp_longest", make_pythainlp("longest"), ["pythainlp"]),
        ("pythainlp_attacut", make_pythainlp("attacut"), ["pythainlp", "attacut"]),
        ("pythainlp_deepcut", make_pythainlp("deepcut"), ["pythainlp", "deepcut"]),
        ("pythainlp_nlpo3", make_pythainlp("nlpo3"), ["pythainlp", "nlpo3"]),
        ("icu", make_icu("th"), ["PyICU"]),
    ],
    "vie_Latn": [
        ("pyvi", make_pyvi, ["pyvi"]),
        ("underthesea", make_underthesea, ["underthesea"]),
    ],
}
# Japanese is left out: stanza's tokens are short-unit words, which would be
# averaged together with the bunsetsu segmentation downstream.
STANZA_LANGS = {"cmn_Hans": "zh-hans", "vie_Latn": "vi"}  # , "jpn_Jpan": "ja"


def package_versions(pkgs):
    out = {}
    for p in pkgs:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def load_segmenters(lang, use_stanza):
    specs = list(SEGMENTERS.get(lang, []))
    if use_stanza and lang in STANZA_LANGS:
        specs.append(("stanza", make_stanza(STANZA_LANGS[lang]), ["stanza"]))
    loaded, failed = {}, {}
    for name, factory, pkgs in specs:
        try:
            fn = factory()
            if not clean_tokens(fn(PROBE[lang])):
                raise RuntimeError("probe sentence produced no words")
            loaded[name] = (fn, package_versions(pkgs))
        except Exception as e:
            failed[name] = f"{type(e).__name__}: {e}"
            print(f"  [skip] {lang} / {name}: {failed[name]}", file=sys.stderr)
    return loaded, failed


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_file(path, lang, use_stanza, normalize):
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    texts = [rec.get("text", "") for rec in records]
    if normalize:
        texts = [unicodedata.normalize(normalize, t) for t in texts]
    ids = [rec.get("id") for rec in records]

    n_char = sum(len(t) for t in texts)
    n_char_word = sum(1 for t in texts for ch in t if is_word_char(ch))
    n_whitespace = sum(1 for t in texts for ch in t if ch.isspace())

    methods = {"whitespace": (str.split, {})}
    segs, failed = load_segmenters(lang, use_stanza)
    methods.update(segs)

    n_words, avg_len, coverage, versions = {}, {}, {}, {}
    for name, (fn, vers) in methods.items():
        nw = nc = 0
        for t in texts:
            toks = clean_tokens(fn(t))
            nw += len(toks)
            nc += sum(word_length(x) for x in toks)
        n_words[name] = nw
        avg_len[name] = nc / nw if nw else None
        coverage[name] = nc / n_char_word if n_char_word else None
        if vers:
            versions[name] = vers

    print(f"=== {lang} ({len(texts)} docs, {n_char} chars, {n_whitespace} whitespace) ===")
    for name in methods:
        flag = ""
        if name == "whitespace" and lang in NO_WHITESPACE_WORDS:
            flag = "   (not a meaningful word count for this language)"
        al = avg_len[name]
        print(f"  {name:20s} n_words={n_words[name]:8d}  "
              f"avg_word_length={al:.3f}  coverage={coverage[name]:.3f}{flag}"
              if al is not None else f"  {name:20s} no words found")
    print()

    result = {
        "n_docs": len(texts),
        "n_char": n_char,
        "n_char_word": n_char_word,
        "n_whitespace": n_whitespace,
        "has_whitespace_words": lang not in NO_WHITESPACE_WORDS,
        "n_words": n_words,
        "avg_word_length": avg_len,
        "word_char_coverage": coverage,
        "segmenter_versions": versions,
        "segmenters_failed": failed,
    }
    return result, ids


def add_english_ratios(results, all_ids):
    if "eng_Latn" not in results:
        print("[word_length_stats] eng_Latn not processed -> no rel_to_eng ratios.",
              file=sys.stderr)
        return
    eng = results["eng_Latn"]
    eng_words = eng["n_words"]["whitespace"]
    for lang, res in results.items():
        if all_ids[lang] != all_ids["eng_Latn"]:
            print(f"[word_length_stats] {lang}: record ids differ from eng_Latn "
                  f"(not parallel?) -> no rel_to_eng ratios.", file=sys.stderr)
            continue
        res["rel_to_eng"] = {
            "n_char": res["n_char"] / eng["n_char"],
            "n_char_word": res["n_char_word"] / eng["n_char_word"],
            "n_words": {m: n / eng_words for m, n in res["n_words"].items()},
        }


def collect_files(inputs, only_20):
    files = []
    for inp in inputs:
        if os.path.isdir(inp):
            for fn in sorted(os.listdir(inp)):
                if not fn.endswith(".json"):
                    continue
                if only_20 and fn[:-5] not in OUR_20_LANGS:
                    continue
                files.append(os.path.join(inp, fn))
        else:
            files.append(inp)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Character counts, word counts and average word length per language.")
    parser.add_argument("inputs", nargs="+", help="JSON file(s) and/or a directory of them")
    parser.add_argument("--only-20", action="store_true",
                        help="For directories, only process the 20 project languages")
    parser.add_argument("--stanza", action="store_true",
                        help="Also run stanza for Chinese, Japanese and Vietnamese "
                             "(downloads models on first use)")
    parser.add_argument("--normalize", choices=["NFC", "NFD", "NFKC", "NFKD"], default=None,
                        help="Unicode-normalize texts before counting (default: count "
                             "codepoints exactly as stored, matching what the model saw)")
    parser.add_argument("--out-json", default="word_length_stats.json",
                        help="Output path (default: word_length_stats.json). "
                             "Pass an empty string to skip writing.")
    args = parser.parse_args()

    files = collect_files(args.inputs, args.only_20)
    if not files:
        print("No matching .json files found.", file=sys.stderr)
        sys.exit(1)

    results, all_ids = {}, {}
    for path in files:
        lang = os.path.splitext(os.path.basename(path))[0]
        try:
            results[lang], all_ids[lang] = analyze_file(path, lang, args.stanza, args.normalize)
        except Exception as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)

    add_english_ratios(results, all_ids)

    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[word_length_stats] Saved results -> {args.out_json}")


if __name__ == "__main__":
    main()