from __future__ import annotations

import re
import string
import zipfile
from functools import lru_cache
from typing import Callable, Mapping, Optional, Protocol, Sequence

import torch


ARPABET_PHONES = (
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)
_KEYWORD_CHARACTERS = set(string.ascii_lowercase + "'")
_NLTK_SETUP = (
    "python3 -m nltk.downloader cmudict "
    "averaged_perceptron_tagger averaged_perceptron_tagger_eng"
)


class CTCVocabulary(Protocol):
    symbols: tuple
    blank_id: int

    def encode(self, text: str) -> torch.Tensor:
        ...


def _normalize_keyword(text: str) -> str:
    normalized = text.strip().lower()
    if not normalized:
        raise ValueError("keyword text must not be empty")
    unsupported = sorted(set(normalized) - _KEYWORD_CHARACTERS)
    if unsupported:
        raise ValueError(
            f"unsupported keyword characters: {unsupported!r} in {text!r}")
    return normalized


class CharacterVocabulary:
    def __init__(self):
        self.symbols = ("<blank>", *string.ascii_lowercase, "'")
        self.blank_id = 0
        self._char_to_id = {
            symbol: index for index, symbol in enumerate(self.symbols)
            if index != self.blank_id
        }

    def __len__(self):
        return len(self.symbols)

    def normalize(self, text: str) -> str:
        return _normalize_keyword(text)

    def encode(self, text: str) -> torch.Tensor:
        normalized = self.normalize(text)
        return torch.tensor(
            [self._char_to_id[char] for char in normalized],
            dtype=torch.long,
        )


class PhonemeVocabulary:
    def __init__(
            self,
            converter: Optional[Callable[[str], Sequence[str]]] = None):
        self.symbols = ("<blank>", *ARPABET_PHONES)
        self.blank_id = 0
        self._phone_to_id = {
            symbol: index for index, symbol in enumerate(self.symbols)
            if index != self.blank_id
        }
        self._converter = converter if converter is not None else self._load_g2p()

    @staticmethod
    def _load_g2p():
        try:
            from g2p_en import G2p
        except ImportError as exc:
            raise RuntimeError(
                "phoneme CTC requires g2p_en; run: "
                "python3 -m pip install g2p-en==2.1.0") from exc
        except (LookupError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                f"g2p_en has missing or corrupt NLTK resources; run: "
                f"{_NLTK_SETUP}") from exc
        try:
            return G2p()
        except (LookupError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                f"g2p_en has missing or corrupt NLTK resources; run: "
                f"{_NLTK_SETUP}") from exc

    def __len__(self):
        return len(self.symbols)

    def normalize(self, text: str) -> str:
        return _normalize_keyword(text)

    @lru_cache(maxsize=None)
    def _phone_ids(self, normalized: str):
        try:
            raw_tokens = self._converter(normalized)
        except (LookupError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                f"g2p_en has missing or corrupt NLTK resources; run: "
                f"{_NLTK_SETUP}") from exc

        phone_ids = []
        for raw_token in raw_tokens:
            token = str(raw_token).strip().upper()
            if not token or token == "'":
                continue
            phone = re.sub(r"[012]$", "", token)
            if phone not in self._phone_to_id:
                raise ValueError(
                    f"unsupported G2P token {raw_token!r} for "
                    f"keyword {normalized!r}")
            phone_ids.append(self._phone_to_id[phone])
        if not phone_ids:
            raise ValueError(
                f"G2P produced an empty pronunciation for {normalized!r}")
        return tuple(phone_ids)

    def encode(self, text: str) -> torch.Tensor:
        normalized = self.normalize(text)
        return torch.tensor(self._phone_ids(normalized), dtype=torch.long)


def build_vocabulary(
        units: str,
        phoneme_converter: Optional[Callable[[str], Sequence[str]]] = None):
    if units == "char":
        return CharacterVocabulary()
    if units == "phoneme":
        return PhonemeVocabulary(phoneme_converter)
    raise ValueError(f"unsupported CTC units: {units!r}")


def checkpoint_units(checkpoint: Mapping) -> str:
    units = checkpoint.get("units", "char")
    if units not in {"char", "phoneme"}:
        raise ValueError(f"unsupported checkpoint CTC units: {units!r}")
    return units


def warm_vocabulary(vocabulary: CTCVocabulary, texts) -> int:
    normalized = sorted({text.strip().lower() for text in texts})
    for text in normalized:
        vocabulary.encode(text)
    return len(normalized)


def required_ctc_frames(targets: torch.Tensor,
                        target_lengths: torch.Tensor) -> torch.Tensor:
    if targets.ndim != 2:
        raise ValueError("targets must have shape (batch, max_target_length)")
    positions = torch.arange(targets.shape[1], device=targets.device)
    repeated = targets[:, 1:] == targets[:, :-1]
    valid_repeat = positions[1:].unsqueeze(0) < target_lengths.unsqueeze(1)
    return target_lengths + (repeated & valid_repeat).sum(dim=1)
