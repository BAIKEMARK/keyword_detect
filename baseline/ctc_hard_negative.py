from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

from ctc_diagnostics import sequence_edit_distance
from ctc_text import CTCVocabulary


def _phone_ngrams(sequence: Sequence[int]) -> set[tuple[int, ...]]:
    if len(sequence) < 2:
        return {(token,) for token in sequence}
    return {
        (sequence[index], sequence[index + 1])
        for index in range(len(sequence) - 1)
    }


def build_phoneme_hard_negatives(
        vocabulary: CTCVocabulary,
        anchor_texts: Iterable[str],
        candidate_texts: Iterable[str]) -> Mapping[str, str]:
    """Return the nearest distinct training word for each anchor word."""
    candidates = sorted({vocabulary.normalize(text) for text in candidate_texts})
    if len(candidates) < 2:
        raise ValueError("hard-negative mining requires at least two words")

    pronunciations = {
        text: tuple(vocabulary.encode(text).tolist())
        for text in candidates
    }
    ngram_index = defaultdict(set)
    length_index = defaultdict(set)
    for text, pronunciation in pronunciations.items():
        for ngram in _phone_ngrams(pronunciation):
            ngram_index[ngram].add(text)
        length_index[len(pronunciation)].add(text)

    neighbors = {}
    for anchor in sorted({vocabulary.normalize(text) for text in anchor_texts}):
        pronunciation = tuple(vocabulary.encode(anchor).tolist())
        pool = set()
        for ngram in _phone_ngrams(pronunciation):
            pool.update(ngram_index[ngram])
        pool.discard(anchor)
        pool = {
            candidate for candidate in pool
            if pronunciations[candidate] != pronunciation
        }
        if not pool:
            for difference in range(len(pronunciation) + 2):
                for length in {
                        len(pronunciation) - difference,
                        len(pronunciation) + difference}:
                    pool.update(length_index.get(length, ()))
                pool.discard(anchor)
                pool = {
                    candidate for candidate in pool
                    if pronunciations[candidate] != pronunciation
                }
                if pool:
                    break
        if not pool:
            pool = {
                candidate for candidate in candidates
                if candidate != anchor
                and pronunciations[candidate] != pronunciation
            }
        if not pool:
            raise ValueError(
                f"no distinguishable phoneme hard negative for {anchor!r}")

        def rank(candidate: str):
            candidate_pronunciation = pronunciations[candidate]
            distance = sequence_edit_distance(
                pronunciation, candidate_pronunciation)
            normalized_distance = distance / max(
                len(pronunciation), len(candidate_pronunciation), 1)
            return (
                normalized_distance,
                abs(len(pronunciation) - len(candidate_pronunciation)),
                candidate,
            )

        neighbors[anchor] = min(pool, key=rank)
    return neighbors
