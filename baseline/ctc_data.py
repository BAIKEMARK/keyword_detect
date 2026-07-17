from __future__ import annotations

import csv
from functools import partial
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from config import AudioConfig
from ctc_text import CharacterVocabulary
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


def _pad_waveforms(waveforms) -> torch.Tensor:
    max_length = max(len(waveform) for waveform in waveforms)
    output = torch.zeros(len(waveforms), max_length, dtype=torch.float32)
    for index, waveform in enumerate(waveforms):
        output[index, :len(waveform)] = waveform
    return output


def _encode_texts(texts, vocabulary: CharacterVocabulary):
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


def collate_ctc_utterances(batch, vocabulary: CharacterVocabulary):
    waveforms = _pad_waveforms([item[0] for item in batch])
    sample_lengths = torch.tensor([item[3] for item in batch], dtype=torch.long)
    targets, target_lengths = _encode_texts(
        [item[1] for item in batch], vocabulary)
    wav_names = [item[2] for item in batch]
    return waveforms, sample_lengths, targets, target_lengths, wav_names


def collate_ctc_scores(batch, vocabulary: CharacterVocabulary):
    waveforms = _pad_waveforms([item[0] for item in batch])
    sample_lengths = torch.tensor([item[4] for item in batch], dtype=torch.long)
    targets, target_lengths = _encode_texts(
        [item[1] for item in batch], vocabulary)
    labels = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    pair_ids = [item[3] for item in batch]
    return (waveforms, sample_lengths, targets, target_lengths, labels,
            pair_ids)


def ctc_utterance_collate(vocabulary: CharacterVocabulary):
    return partial(collate_ctc_utterances, vocabulary=vocabulary)


def ctc_score_collate(vocabulary: CharacterVocabulary):
    return partial(collate_ctc_scores, vocabulary=vocabulary)
