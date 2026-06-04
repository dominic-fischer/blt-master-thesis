import torch
from blt_patcher import load_patcher, patch_text

REPO = "facebook/blt-1b"
ENTROPY_REPO = "hf-weights/entropy_model"

TOP_K = 1  # set to 1-5

W_BITS   = 33
W_SCRIPT = 50

texts = {
    "English":  "The quick brown fox jumps over the lazy dog.",
    "Hindi":    "तेज़ लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
    "Chinese":  "敏捷的棕色狐狸跳过了懒狗。",
    "Arabic":   "الثعلب البني السريع يقفز فوق الكلب الكسول.",
    "Georgian": "სწრაფი მოყავისფერი მელა ზარმაც ძაღლს გადაახტა.",
    "Armenian": "Արագ շագանակագույն աղվեսը ցատկում է ծույլ շան վրայով։",
    "Tibetan":  "རྒྱང་མགྱོགས་པའི་བོང་བུ་གཉིད་ལོག་པའི་ཁྱི་ལ་མཆོང་།",
    "Lao":      "ຫມາກໄມ້ສີນ້ຳຕານໄວລອຍຢູ່ເທິງໝາຄ້ານ.",
    "Khmer":    "សត្វក្តាន់ពណ៌ត្នោតលឿនលោតឆ្លងពីលើឆ្កែខ្ជិល།",
    "Amharic":  "ፈጣኑ ቡናማ ቀበሮ ሰነፍ ውሻውን ዘለለ།",
}

SCRIPT_RANGES = [
    (0,     127,   "ASCII"),
    (128,   591,   "Latin-Ext"),
    (592,   687,   "IPA"),
    (688,   879,   "Other_1"),
    (880,   1023,  "Greek"),
    (1024,  1327,  "Cyrillic"),
    (1328,  1423,  "Armenian"),
    (1424,  1535,  "Hebrew"),
    (1536,  1791,  "Arabic"),
    (1792,  1871,  "Syriac"),
    (1872,  2303,  "Other_2"),
    (2304,  2431,  "Devanagari"),
    (2432,  2559,  "Bengali"),
    (2560,  2687,  "Gurmukhi"),
    (2688,  2815,  "Gujarati"),
    (2816,  2943,  "Oriya"),
    (2944,  3071,  "Tamil"),
    (3072,  3199,  "Telugu"),
    (3200,  3327,  "Kannada"),
    (3328,  3455,  "Malayalam"),
    (3456,  3583,  "Sinhala"),
    (3584,  3711,  "Thai"),
    (3712,  3839,  "Lao"),
    (3840,  4095,  "Tibetan"),
    (4096,  4255,  "Myanmar"),
    (4256,  4351,  "Georgian"),
    (4352,  4607,  "Hangul-Jamo"),
    (4608,  5119,  "Ethiopic"),
    (5120,  6015,  "Other_3"),
    (6016,  6143,  "Khmer"),
    (6144,  6319,  "Mongolian"),
    (6320,  11903, "Other_4"),
    (11904, 12031, "CJK-Rad"),
    (12032, 12287, "Other_5"),
    (12288, 12351, "CJK-Sym"),
    (12352, 12447, "Hiragana"),
    (12448, 12543, "Katakana"),
    (12544, 13311, "Other_6"),
    (13312, 19903, "CJK-ExtA"),
    (19904, 19967, "Other_7"),
    (19968, 40959, "CJK"),
    (40960, 44031, "Other_8"),
    (44032, 55215, "Hangul"),
    (55216, 63743, "Other_9"),
    (63744, 64255, "CJK-Compat"),
    (64256, 65535, "Other_10"),
]

SPECIAL_CHARS = {0x00: "\\0", 0x09: "\\t", 0x0A: "\\n", 0x0D: "\\r"}

def get_script(codepoint):
    for start, end, name in SCRIPT_RANGES:
        if start <= codepoint <= end:
            return name, start, end
    return "Other", None, None

def seq_length(b):
    if b < 0x80: return 1
    if b < 0xE0: return 2
    if b < 0xF0: return 3
    return 4

def format_byte_binary(b):
    s = f"{b:08b}"
    if b < 0x80:  return f"0|{s[1:]}"
    if b < 0xC0:  return f"10|{s[2:]}"
    if b < 0xE0:  return f"110|{s[3:]}"
    if b < 0xF0:  return f"1110|{s[4:]}"
    return        f"11110|{s[5:]}"

def format_char(b):
    if b in SPECIAL_CHARS: return SPECIAL_CHARS[b]
    if 0x20 <= b <= 0x7E:  return f"'{chr(b)}'"
    return f"0x{b:02x}"

def build_char_map(context_bytes):
    char_map = []
    i = 0
    while i < len(context_bytes):
        b = context_bytes[i]
        length = seq_length(b)
        for j in range(length):
            if i + j < len(context_bytes):
                char_map.append((j, length, b))
        i += length
    return char_map

def reconstruct_codepoint(off, total, lead, context_bytes, pos):
    if off != total - 1:
        return None
    if total == 1: return lead
    if total == 2:
        return ((lead & 0x1F) << 6) | (context_bytes[pos] & 0x3F)
    if total == 3:
        return ((lead & 0x0F) << 12) | ((context_bytes[pos-1] & 0x3F) << 6) | (context_bytes[pos] & 0x3F)
    if total == 4:
        return ((lead & 0x07) << 18) | ((context_bytes[pos-2] & 0x3F) << 12) | ((context_bytes[pos-1] & 0x3F) << 6) | (context_bytes[pos] & 0x3F)
    return None

def make_script_str(cp_low, cp_high=None):
    if cp_high is None:
        name, start, end = get_script(cp_low)
        range_str = f"({start}-{end})" if start is not None else ""
        return f"{name}{range_str}"
    scripts = []
    cp = cp_low
    while cp <= cp_high:
        name, start, end = get_script(cp)
        if not scripts or scripts[-1][0] != name:
            scripts.append((name, start, end))
        cp = (end + 1) if end is not None else cp_high + 1
    low   = scripts[0][1]  if scripts[0][1]  is not None else cp_low
    high  = scripts[-1][2] if scripts[-1][2] is not None else cp_high
    names = "/".join(s[0] for s in scripts)
    return f"{names}({low}-{high})"

def format_bits(byte_val, off, total, lead, context_bytes, pos):
    """Returns (bits_str, script_str) as a tuple."""
    pc = byte_val & 0x3F

    # ASCII
    if byte_val < 0x80:
        name, start, end = get_script(byte_val)
        range_str = f"({start}-{end})" if start is not None else ""
        return f"{byte_val:07b}={byte_val}", f"{name}{range_str}"

    # Leading byte
    if byte_val >= 0xC0:
        if byte_val < 0xE0:
            cp_low  = (byte_val & 0x1F) << 6
            cp_high = cp_low | 0x3F
            lb = f"{byte_val & 0x1F:05b}"
            return f"{lb}+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        if byte_val < 0xF0:
            cp_low  = (byte_val & 0x0F) << 12
            cp_high = cp_low | 0xFFF
            lb = f"{byte_val & 0x0F:04b}"
            return f"{lb}+bbbbbb+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        cp_low  = (byte_val & 0x07) << 18
        cp_high = cp_low | 0x3FFFF
        lb = f"{byte_val & 0x07:03b}"
        return f"{lb}+bbbbbb+bbbbbb+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)

    # Continuation byte
    if total == 2:
        if off == 0:
            lb = f"{lead & 0x1F:05b}"
            pb = f"{pc:06b}"
            cp = ((lead & 0x1F) << 6) | pc
            return f"{lb}+{pb}={cp}", make_script_str(cp)

    if total == 3:
        if off == 0:
            lb      = f"{lead & 0x0F:04b}"
            pb      = f"{pc:06b}"
            cp_low  = ((lead & 0x0F) << 12) | (pc << 6)
            cp_high = cp_low | 0x3F
            return f"{lb}+{pb}+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        if off == 1:
            lb  = f"{lead & 0x0F:04b}"
            c1  = context_bytes[pos] & 0x3F
            c1b = f"{c1:06b}"
            pb  = f"{pc:06b}"
            cp  = ((lead & 0x0F) << 12) | (c1 << 6) | pc
            return f"{lb}+{c1b}+{pb}={cp}", make_script_str(cp)

    if total == 4:
        if off == 0:
            lb      = f"{lead & 0x07:03b}"
            pb      = f"{pc:06b}"
            cp_low  = ((lead & 0x07) << 18) | (pc << 12)
            cp_high = cp_low | 0xFFF
            return f"{lb}+{pb}+bbbbbb+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        if off == 1:
            lb      = f"{lead & 0x07:03b}"
            c1      = context_bytes[pos] & 0x3F
            c1b     = f"{c1:06b}"
            pb      = f"{pc:06b}"
            cp_low  = ((lead & 0x07) << 18) | (c1 << 12) | (pc << 6)
            cp_high = cp_low | 0x3F
            return f"{lb}+{c1b}+{pb}+bbbbbb={cp_low}-{cp_high}", make_script_str(cp_low, cp_high)
        if off == 2:
            lb  = f"{lead & 0x07:03b}"
            c1  = context_bytes[pos-1] & 0x3F
            c2  = context_bytes[pos] & 0x3F
            c1b = f"{c1:06b}"
            c2b = f"{c2:06b}"
            pb  = f"{pc:06b}"
            cp  = ((lead & 0x07) << 18) | (c1 << 12) | (c2 << 6) | pc
            return f"{lb}+{c1b}+{c2b}+{pb}={cp}", make_script_str(cp)

    return "?", "?"

print("Loading patcher...")
tokenizer, patcher = load_patcher(repo=REPO, entropy_repo=ENTROPY_REPO)
offset = tokenizer.offsetting_special_char

output_lines = []

for lang, text in texts.items():
    output_lines.append("=" * 80)
    output_lines.append(f"  Language : {lang}")
    output_lines.append(f"  Text     : {text}")
    output_lines.append("")

    result = patch_text(text, tokenizer, patcher)
    context_bytes = result['text_bytes']
    char_map = build_char_map(context_bytes)

    for i, (byte_val, score, pred) in enumerate(
        zip(context_bytes, result['scores'], result['preds'])
    ):
        off, total, lead = char_map[i]
        char_disp = format_char(byte_val)
        byte_bin  = format_byte_binary(byte_val)
        cur_bits, cur_script = format_bits(byte_val, off - 1, total, lead, context_bytes, i - 1)

        output_lines.append(f"  [{i:4d}]  {char_disp:<6}  {byte_bin:<10}  entropy={score:.3f}  {cur_bits:<{W_BITS}}  {cur_script:<{W_SCRIPT}}")

        probs = torch.softmax(torch.tensor(pred), dim=-1)
        topk  = probs.topk(min(TOP_K, 5))

        for idx, p in zip(topk.indices, topk.values):
            b            = idx.item() - offset
            pchar        = format_char(b) if 0 <= b <= 255 else f"[{idx.item()}]"
            pbin         = format_byte_binary(b) if 0 <= b <= 255 else "?"
            cp_bits, cp_script = format_bits(b, off, total, lead, context_bytes, i)
            output_lines.append(f"       →  {pchar:<6}  {pbin:<10}  prob=   {p.item():.3f}  {cp_bits:<{W_BITS}}  {cp_script:<{W_SCRIPT}}")

        if off == total - 1:
            output_lines.append("  ---")
        else:
            output_lines.append("")

    output_lines.append("")

output = "\n".join(output_lines)
print(output)

with open("inspect_probabilities_output.txt", "w", encoding="utf-8") as f:
    f.write(output)

print("\nOutput written to inspect_probabilities_output.txt")