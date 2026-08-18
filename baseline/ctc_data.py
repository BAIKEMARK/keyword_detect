from __future__ import annotations

import csv
from functools import partial
from typing import List, Mapping, Optional, Sequence

import torch
from torch.utils.data import Dataset

from config import AudioConfig
from ctc_text import CTCVocabulary
from data import (NoiseAugmenter, normalize_waveform, read_wav,
                  truncate_waveform)


def load_ctc_training_examples(csv_path: str) -> List[dict]:
    examples = []
    seen_waveforms = set()
    with open(csv_path, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            for side, text_field in (
                    ("enroll", "enroll_txt"), ("query", "query_txt")):
                wav_name = f"wav/{row['id']}_{side}.wav"
                if wav_name in seen_waveforms:
                    continue
                seen_waveforms.add(wav_name)
                examples.append({"wav_name": wav_name, "text": row[text_field]})
    return examples


def load_ctc_score_pairs(csv_path: str, with_label: bool) -> List[dict]:
    pairs = []
    with open(csv_path, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            item = {"id": row["id"], "enroll_text": row["enroll_txt"]}
            if with_label:
                item["label"] = int(row["label"])
            pairs.append(item)
    return pairs


def load_ctc_training_pairs(csv_path: str) -> List[dict]:
    pairs = []
    with open(csv_path, encoding="utf-8") as file:
        for row in csv.DictReader(file):
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"pair label must be 0 or 1: {row['id']!r}")
            pairs.append({
                "id": row["id"],
                "enroll_text": row["enroll_txt"],
                "query_text": row["query_txt"],
                "label": label,
            })
    return pairs


def _load_waveform(zip_path: str, wav_name: str, cfg: AudioConfig,
                   max_samples: int,
                   augment: Optional[NoiseAugmenter]) -> torch.Tensor:
    waveform = read_wav(zip_path, wav_name, cfg.sample_rate)
    if augment is not None:
        waveform = augment(waveform)
    return normalize_waveform(truncate_waveform(waveform, max_samples))


class CTCUtteranceDataset(Dataset):
    def __init__(self, examples: List[dict], zip_path: str, cfg: AudioConfig,
                 max_samples: int,
                 augment: Optional[NoiseAugmenter] = None):
        self.examples = examples
        self.zip_path = zip_path
        self.cfg = cfg
        self.max_samples = max_samples
        self.augment = augment

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        waveform = _load_waveform(
            self.zip_path, example["wav_name"], self.cfg,
            self.max_samples, self.augment)
        return waveform, example["text"], example["wav_name"], len(waveform)


class CTCScoreDataset(Dataset):
    def __init__(self, pairs: List[dict], zip_path: str, cfg: AudioConfig,
                 max_samples: int, inference: bool = False):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.max_samples = max_samples
        self.inference = inference

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        waveform = _load_waveform(
            self.zip_path, f"wav/{pair['id']}_query.wav", self.cfg,
            self.max_samples, augment=None)
        label = -1 if self.inference else pair["label"]
        return (waveform, pair["enroll_text"], label, pair["id"],
                len(waveform))


class CTCPairTrainingDataset(Dataset):
    def __init__(self, pairs: List[dict], zip_path: str, cfg: AudioConfig,
                 max_samples: int,
                 augment: Optional[NoiseAugmenter] = None):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.max_samples = max_samples
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        waveform = _load_waveform(
            self.zip_path, f"wav/{pair['id']}_query.wav", self.cfg,
            self.max_samples, self.augment)
        return (
            waveform, pair["enroll_text"], pair["query_text"],
            pair["label"], pair["id"], len(waveform),
        )


def _pad_waveforms(waveforms) -> torch.Tensor:
    max_length = max(len(waveform) for waveform in waveforms)
    output = torch.zeros(len(waveforms), max_length, dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        output[index, :len(waveform)] = waveform
    return output


def _encode_texts(texts, vocabulary: CTCVocabulary):
    encoded = [vocabulary.encode(text) for text in texts]
    lengths = torch.tensor([len(target) for target in encoded], dtype=torch.long)
    targets = torch.full(
        (len(encoded), int(lengths.max())),
        vocabulary.blank_id,
        dtype=torch.long,
    )
    for index, target in enumerate(encoded):
        targets[index, :len(target)] = target
    return targets, lengths


def collate_ctc_utterances(
        batch, vocabulary: CTCVocabulary,
        hard_negatives: Optional[Mapping[str, str]] = None):
    waveforms = _pad_waveforms([item[0] for item in batch])
    sample_lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
    targets, target_lengths = _encode_texts(
        [item[1] for item in batch], vocabulary)
    wav_names = [item[2] for item in batch]
    result = (waveforms, sample_lengths, targets, target_lengths, wav_names)
    if hard_negatives is None:
        return result
    negative_texts = [
        hard_negatives[vocabulary.normalize(item[1])] for item in batch
    ]
    negative_targets, negative_target_lengths = _encode_texts(
        negative_texts, vocabulary)
    return (*result, negative_targets, negative_target_lengths)


def collate_ctc_scores(batch, vocabulary: CTCVocabulary):
    waveforms = _pad_waveforms([item[0] for item in batch])
    sample_lengths = torch.tensor([item[4] for item in batch], dtype=torch.long)
    targets, target_lengths = _encode_texts(
        [item[1] for item in batch], vocabulary)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    pair_ids = [item[3] for item in batch]
    return (waveforms, sample_lengths, targets, target_lengths, labels,
            pair_ids)


def collate_ctc_training_pairs(
        batch, vocabulary: CTCVocabulary,
        hard_negative_candidates: Mapping[str, Sequence[str]]):
    waveforms = _pad_waveforms([item[0] for item in batch])
    sample_lengths = torch.tensor([item[5] for item in batch], dtype=torch.long)
    enroll_targets, enroll_target_lengths = _encode_texts(
        [item[1] for item in batch], vocabulary)
    query_targets, query_target_lengths = _encode_texts(
        [item[2] for item in batch], vocabulary)

    candidate_rows = [
        tuple(hard_negative_candidates[vocabulary.normalize(item[2])])
        for item in batch
    ]
    candidate_counts = {len(row) for row in candidate_rows}
    if len(candidate_counts) != 1 or not candidate_counts or 0 in candidate_counts:
        raise ValueError(
            "each training pair needs the same positive number of hard negatives")
    candidate_count = candidate_counts.pop()
    hard_targets, hard_target_lengths = _encode_texts(
        [text for row in candidate_rows for text in row], vocabulary)
    labels = torch.tensor([item[3] for item in batch], dtype=torch.float32)
    pair_ids = [item[4] for item in batch]
    return (
        waveforms, sample_lengths,
        enroll_targets, enroll_target_lengths,
        query_targets, query_target_lengths,
        hard_targets, hard_target_lengths, candidate_count,
        labels, pair_ids,
    )


def ctc_utterance_collate(
        vocabulary: CTCVocabulary,
        hard_negatives: Optional[Mapping[str, str]] = None):
    return partial(
        collate_ctc_utterances,
        vocabulary=vocabulary,
        hard_negatives=hard_negatives,
    )


def ctc_score_collate(vocabulary: CTCVocabulary):
    return partial(collate_ctc_scores, vocabulary=vocabulary)


def ctc_training_pair_collate(
        vocabulary: CTCVocabulary,
        hard_negative_candidates: Mapping[str, Sequence[str]]):
    return partial(
        collate_ctc_training_pairs,
        vocabulary=vocabulary,
        hard_negative_candidates=hard_negative_candidates,
    )
