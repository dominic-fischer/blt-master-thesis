# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import logging
import re

from bytelatent.tokenizers.abstract_tokenizer import Tokenizer
from bytelatent.tokenizers.constants import (
    BOE_ID,
    BOS_ID,
    BPE_ID,
    BYTE_UNITS,
    EOS_ID,
    OFFSET,
    PAD_ID,
)
from bytelatent.tokenizers.sentence_piece_tokenizer import SentencePieceTokenizer

logger = logging.getLogger()


def convert_to_bytes(s):
    # check if the output is a bytes like object of the format <0x00>
    if re.match(r"<0x[0-9a-fA-F]+>", s):
        return bytes.fromhex(s[3:-1])
    else:
        return bytes(s, "utf-8", errors="ignore")


def text2bytes_bpe_delims(
    text: str,
    *,
    bpe_tokenizer,
    bpe_id: int,
    offsetting_special_char: int,
    add_bos: bool,
    add_eos: bool,
):
    cur_bpe = bpe_tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
    # merge the leading space tokens
    leading_space_tokens = []
    other_bpe_tokens = []
    leading = True
    for token in cur_bpe:
        bpe_str = bpe_tokenizer.sp_model.id_to_piece(token)
        if leading and all(c == "▁" for c in bpe_str):
            leading_space_tokens.append(bpe_str)
        else:
            leading = False
            other_bpe_tokens.append(bpe_str)
    cur_bpe_strs = ["".join(leading_space_tokens)] + other_bpe_tokens

    # Remove the '▁' characters
    bpe_strs = []
    for i, bpe_str in enumerate(cur_bpe_strs):
        if (
            len(bpe_strs) <= 1
            and all([c == " " for s in bpe_strs for c in s])
            and not all(c == "▁" for c in bpe_str)
        ):
            # Remove leading space for first non space token.
            bpe_str = bpe_str.replace("▁", "")
        elif i == 0 and all(c == "▁" for c in bpe_str):
            bpe_str = " " * (len(text) - len(text.lstrip(" ")))
        else:
            bpe_str = bpe_str.replace("▁", " ")
        if len(bpe_str) > 0:
            bpe_strs.append(bpe_str)
    ex_seq = []
    # Convert bpe tokens to bytes
    for s in bpe_strs:
        byte_chunk = convert_to_bytes(s)
        proc_chunk = [int(unit) for unit in byte_chunk]
        ex_seq.extend([bpe_id - offsetting_special_char] + proc_chunk)

    return ex_seq


class BltTokenizer(Tokenizer):
    def __init__(
        self,
        *,
        vocab_size_unit_1: int = BYTE_UNITS,
        bpe_delim: bool = False,
        bpe_tokenizer_path="/home/artidoro/tokenizers/llama_v2.tokenizer.model",
        add_bos: bool = True,
        add_eos: bool = True,
        custom_encoding_path: str | None = None,
    ):
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.boe_id = BOE_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.bpe_id = BPE_ID
        self.bpe_tokenizer_path = bpe_tokenizer_path
        if bpe_delim:
            self.bpe_tokenizer = SentencePieceTokenizer(
                model_path=self.bpe_tokenizer_path
            )
        else:
            self.bpe_tokenizer = None
        self.bpe_delim = bpe_delim
        self.offsetting_special_char = OFFSET
        self.vocab_size_unit_1 = vocab_size_unit_1
        self.n_words = vocab_size_unit_1 + self.offsetting_special_char

        # Custom (random-but-reproducible) per-character byte encoding, as an
        # alternative to raw UTF-8 -- see training_setup/build_custom_encoding.py.
        # Loaded once per tokenizer instance (cheap: one JSON file, tens of
        # thousands of entries at most). Vocab size/n_words are UNCHANGED --
        # each individual byte within a character's 2- or 3-byte code is
        # still just a value in 0-255, the same range UTF-8 bytes already
        # occupy, so nothing downstream needs to know this happened.
        self.custom_encoding_path = custom_encoding_path
        if custom_encoding_path is not None:
            with open(custom_encoding_path, encoding="utf-8") as f:
                self.custom_encoding = json.load(f)
            self.custom_bytes_per_char = self.custom_encoding["bytes_per_char"]
            logger.info(
                "BltTokenizer: using custom encoding from %s (%d chars, "
                "%d bytes/char, seed=%s)",
                custom_encoding_path,
                self.custom_encoding["num_chars"],
                self.custom_bytes_per_char,
                self.custom_encoding.get("seed"),
            )
        else:
            self.custom_encoding = None
            self.custom_bytes_per_char = None

    def get_vocab_size(self) -> int:
        return self.n_words

    def _custom_encode_to_bytes(self, text: str) -> bytes:
        """Encodes text using the custom per-character byte mapping instead
        of UTF-8. Raises on any character with no entry in the mapping --
        deliberately loud rather than silently dropping/substituting,
        since build_custom_encoding.py is meant to scan training + val +
        FLORES+ exhaustively, so a missing character here means the
        encoding needs to be rebuilt to include it, not that this text
        should be silently corrupted."""
        char_to_bytes_hex = self.custom_encoding["char_to_bytes_hex"]
        raw = bytearray()
        for ch in text:
            hex_key = char_to_bytes_hex.get(ch)
            if hex_key is None:
                raise ValueError(
                    f"Character {ch!r} (U+{ord(ch):04X}) has no entry in the "
                    f"custom encoding at {self.custom_encoding_path} -- rebuild "
                    f"it with training_setup/build_custom_encoding.py to include "
                    f"this character before encoding text that contains it."
                )
            raw.extend(bytes.fromhex(hex_key))
        return bytes(raw)

    def _custom_decode_from_bytes(self, raw_bytes: bytes) -> str:
        """Inverse of _custom_encode_to_bytes: groups raw bytes back into
        bytes_per_char-sized chunks and looks each one up. Any trailing
        partial chunk (fewer than bytes_per_char bytes left, e.g. from
        truncating mid-character) is silently dropped, same spirit as the
        UTF-8 path's errors="ignore"."""
        n = self.custom_bytes_per_char
        bytes_hex_to_char = self.custom_encoding["bytes_hex_to_char"]
        usable_len = len(raw_bytes) - (len(raw_bytes) % n)
        chars = []
        for i in range(0, usable_len, n):
            chunk_hex = raw_bytes[i : i + n].hex()
            chars.append(bytes_hex_to_char.get(chunk_hex, "\ufffd"))
        return "".join(chars)

    def encode(
        self, text: str, add_bos: bool | None = None, add_eos: bool | None = None
    ):
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        if self.bpe_delim:
            tokens = text2bytes_bpe_delims(
                text,
                bpe_tokenizer=self.bpe_tokenizer,
                bpe_id=self.bpe_id,
                offsetting_special_char=self.offsetting_special_char,
                add_bos=False,
                add_eos=False,
            )
        elif self.custom_encoding is not None:
            tokens = self._custom_encode_to_bytes(text)
        else:
            tokens = bytes(text, encoding="utf-8", errors="ignore")

        # Offsetting
        tokens = [int(unit) + self.offsetting_special_char for unit in tokens]

        if add_bos:
            tokens.insert(0, self.bos_id)
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: list[int], cut_at_eos: bool = False):
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        raw_bytes = bytes(
            [
                tok - self.offsetting_special_char
                for tok in tokens
                if tok - self.offsetting_special_char >= 0
            ]
        )
        if self.custom_encoding is not None:
            return self._custom_decode_from_bytes(raw_bytes)
        return raw_bytes.decode("utf-8", errors="ignore")

    def get_token_offsets(self, text: str, tokens: list[int] | None = None):
        # TODO: Figure out what this does
        raise NotImplementedError()