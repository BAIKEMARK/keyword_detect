from __future__ import annotations

import csv
import io
import os
import glob
import zipfile
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from config import AudioConfig

_ZIP_CACHE: dict = {}


def _get_zip(path: str) -> zipfile.ZipFile:
    key = (os.getpid(), path)
    if key not in _ZIP_CACHE:
        _ZIP_CACHE[key] = zipfile.ZipFile(path, "r")
    return _ZIP_CACHE[key]


def read_wav(zip_path: str, name: str, sr: int) -> np.ndarray:
    data = _get_zip(zip_path).read(name)
    wav, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        t = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), file_sr, sr)
        wav = t.squeeze(0).numpy()
    return wav.astype(np.float32)


def read_audio_file(path: str, sr: int) -> np.ndarray:
    wav, file_sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        t = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), file_sr, sr)
        wav = t.squeeze(0).numpy()
    return wav.astype(np.float32)


def load_pairs(csv_path: str, with_label: bool) -> List[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = {"id": r["id"]}
            if with_label:
                item["label"] = int(r["label"])
            rows.append(item)
    return rows


def mix_at_snr(wav: np.ndarray, noise: np.ndarray, snr_db: float,
               rng: np.random.Generator) -> np.ndarray:
    if len(wav) == 0 or len(noise) == 0:
        return wav
    if len(noise) < len(wav):
        repeats = int(np.ceil(len(wav) / len(noise)))
        noise = np.tile(noise, repeats)
    if len(noise) > len(wav):
        start = int(rng.integers(0, len(noise) - len(wav) + 1))
        noise = noise[start:start + len(wav)]

    sig_power = float(np.mean(wav ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return (wav + noise).astype(np.float32)


class NoiseAugmenter:
    def __init__(self, sample_rate: int, prob: float, snr_min: float,
                 snr_max: float, noise_dir: str = "", seed: int = 42):
        self.sample_rate = sample_rate
        self.prob = prob
        self.snr_min = snr_min
        self.snr_max = snr_max
        self.seed = seed
        self._rngs: dict[int, np.random.Generator] = {}
        self.noise_paths = self._find_noise_paths(noise_dir)

    @staticmethod
    def _find_noise_paths(noise_dir: str) -> List[str]:
        if not noise_dir:
            return []
        patterns = ["**/*.wav", "**/*.flac", "**/*.ogg"]
        paths: List[str] = []
        for pattern in patterns:
            paths.extend(glob.glob(os.path.join(noise_dir, pattern),
                                   recursive=True))
        return sorted(paths)

    def _rng(self) -> np.random.Generator:
        pid = os.getpid()
        if pid not in self._rngs:
            self._rngs[pid] = np.random.default_rng(self.seed + pid)
        return self._rngs[pid]

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        rng = self._rng()
        if self.prob <= 0 or rng.random() >= self.prob:
            return wav

        if self.noise_paths:
            noise_path = self.noise_paths[int(rng.integers(0, len(self.noise_paths)))]
            noise = read_audio_file(noise_path, self.sample_rate)
        else:
            noise = rng.standard_normal(len(wav)).astype(np.float32)

        snr_db = float(rng.uniform(self.snr_min, self.snr_max))
        return mix_at_snr(wav, noise, snr_db, rng)


class LogMel:
    def __init__(self, cfg: AudioConfig):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        )

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.log(self.mel(wav) + 1e-6)


def pad_spec(spec: torch.Tensor, max_frames: int) -> torch.Tensor:
    """(n_mels, T) -> (1, n_mels, max_frames)"""
    T = spec.shape[-1]
    if T < max_frames:
        spec = torch.nn.functional.pad(spec, (0, max_frames - T))
    else:
        spec = spec[:, :max_frames]
    return spec.unsqueeze(0)


class PairDataset(Dataset):
    def __init__(self, pairs: List[dict], zip_path: str, cfg: AudioConfig,
                 inference: bool = False,
                 augment: Optional[NoiseAugmenter] = None):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.inference = inference
        self.augment = augment
        self.logmel = LogMel(cfg)

    def __len__(self):
        return len(self.pairs)

    def _feat(self, wav_name: str) -> torch.Tensor:
        wav = read_wav(self.zip_path, wav_name, self.cfg.sample_rate)
        if self.augment is not None:
            wav = self.augment(wav)
        spec = self.logmel(torch.from_numpy(wav))
        return pad_spec(spec, self.cfg.max_frames)

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        pid = p["id"]
        e = self._feat(f"wav/{pid}_enroll.wav")
        q = self._feat(f"wav/{pid}_query.wav")
        label = -1 if self.inference else p["label"]
        return e, q, label, pid


def collate(batch):
    es = torch.stack([b[0] for b in batch])
    qs = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    ids = [b[3] for b in batch]
    return es, qs, labels, ids
