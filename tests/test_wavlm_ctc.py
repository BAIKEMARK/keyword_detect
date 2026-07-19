from __future__ import annotations

import csv
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "baseline"))

from config import AudioConfig  # noqa: E402
from ctc_data import (CTCScoreDataset, collate_ctc_scores,  # noqa: E402
                      collate_ctc_utterances, load_ctc_score_pairs,
                      load_ctc_training_examples)
from ctc_score import ctc_log_probability, normalized_ctc_score  # noqa: E402
from ctc_text import (CharacterVocabulary, PhonemeVocabulary,  # noqa: E402
                      build_vocabulary, checkpoint_units,
                      required_ctc_frames, warm_vocabulary)
from infer_wavlm_ctc import collect_scores  # noqa: E402
from train_wavlm_ctc import (ctc_valid_mask,  # noqa: E402
                             default_last_checkpoint_path, parse_args,
                             training_config,
                             validate_resume_checkpoint)
from wavlm_ctc_model import FrozenWavLMCTC  # noqa: E402


class CharacterVocabularyTest(unittest.TestCase):
    def setUp(self):
        self.vocabulary = CharacterVocabulary()

    def test_normalization_and_encoding(self):
        self.assertEqual(self.vocabulary.normalize(" London'S "), "london's")
        encoded = self.vocabulary.encode("a'z")
        torch.testing.assert_close(encoded, torch.tensor([1, 27, 26]))

    def test_rejects_empty_and_unsupported_text(self):
        for text in ("", "hello world", "中文", "word-"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.vocabulary.encode(text)

    def test_required_frames_counts_adjacent_repeats(self):
        targets = torch.stack([
            self.vocabulary.encode("letter"),
            self.vocabulary.encode("abcdef"),
        ])
        lengths = torch.tensor([6, 6])
        torch.testing.assert_close(
            required_ctc_frames(targets, lengths), torch.tensor([7, 6]))


class PhonemeVocabularyTest(unittest.TestCase):
    def test_fixed_inventory_and_stress_removal(self):
        vocabulary = PhonemeVocabulary(
            converter=lambda text: ["HH", "AH0", " ", "L", "OW1", "'"])
        self.assertEqual(len(vocabulary), 40)
        self.assertEqual(vocabulary.symbols[0], "<blank>")
        self.assertEqual(vocabulary.normalize(" Hello "), "hello")
        expected = torch.tensor([
            vocabulary.symbols.index(phone)
            for phone in ("HH", "AH", "L", "OW")
        ])
        torch.testing.assert_close(vocabulary.encode("Hello"), expected)

    def test_rejects_empty_and_unsupported_pronunciations(self):
        for output in ([" "], ["HH", "?"]):
            with self.subTest(output=output):
                vocabulary = PhonemeVocabulary(
                    converter=lambda text, result=output: result)
                with self.assertRaises(ValueError):
                    vocabulary.encode("hello")

    def test_build_vocabulary_and_old_checkpoint_default(self):
        self.assertIsInstance(build_vocabulary("char"), CharacterVocabulary)
        vocabulary = build_vocabulary(
            "phoneme", phoneme_converter=lambda text: ["K", "AE1", "T"])
        self.assertIsInstance(vocabulary, PhonemeVocabulary)
        self.assertEqual(checkpoint_units({}), "char")
        self.assertEqual(checkpoint_units({"units": "phoneme"}), "phoneme")
        with self.assertRaises(ValueError):
            build_vocabulary("word")
        with self.assertRaises(ValueError):
            checkpoint_units({"units": "word"})

    def test_warm_vocabulary_deduplicates_text(self):
        calls = []

        def convert(text):
            calls.append(text)
            return ["K", "AE1", "T"]

        vocabulary = PhonemeVocabulary(converter=convert)
        count = warm_vocabulary(vocabulary, ["cat", " Cat ", "cat"])
        self.assertEqual(count, 1)
        self.assertEqual(calls, ["cat"])


class CTCDataTest(unittest.TestCase):
    def setUp(self):
        self.vocabulary = CharacterVocabulary()

    def _csv(self, rows):
        temporary = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", newline="", encoding="utf-8",
            delete=False)
        with temporary:
            writer = csv.DictWriter(
                temporary,
                fieldnames=["id", "enroll_txt", "query_txt", "label"],
            )
            writer.writeheader()
            writer.writerows(rows)
        self.addCleanup(Path(temporary.name).unlink)
        return temporary.name

    def test_expands_both_sides_and_deduplicates_waveform_names(self):
        path = self._csv([
            {"id": "pair_1", "enroll_txt": "hello",
             "query_txt": "hello", "label": "1"},
            {"id": "pair_1", "enroll_txt": "hello",
             "query_txt": "hello", "label": "1"},
        ])
        examples = load_ctc_training_examples(path)
        self.assertEqual(examples, [
            {"wav_name": "wav/pair_1_enroll.wav", "text": "hello"},
            {"wav_name": "wav/pair_1_query.wav", "text": "hello"},
        ])

    def test_score_pairs_expose_no_query_text(self):
        path = self._csv([
            {"id": "pair_1", "enroll_txt": "target",
             "query_txt": "secret", "label": "0"},
        ])
        pair = load_ctc_score_pairs(path, with_label=True)[0]
        self.assertEqual(pair, {
            "id": "pair_1", "enroll_text": "target", "label": 0})
        self.assertNotIn("query_txt", pair)

    def test_training_and_score_collation(self):
        utterance_batch = [
            (torch.ones(5), "cat", "a.wav", 5),
            (torch.ones(3), "letter", "b.wav", 3),
        ]
        waves, lengths, targets, target_lengths, names = \
            collate_ctc_utterances(utterance_batch, self.vocabulary)
        self.assertEqual(waves.shape, (2, 5))
        torch.testing.assert_close(lengths, torch.tensor([5, 3]))
        torch.testing.assert_close(target_lengths, torch.tensor([3, 6]))
        self.assertEqual(names, ["a.wav", "b.wav"])

        score_batch = [
            (torch.ones(4), "cat", 1, "pair_1", 4),
            (torch.ones(2), "dog", 0, "pair_2", 2),
        ]
        result = collate_ctc_scores(score_batch, self.vocabulary)
        self.assertEqual(result[0].shape, (2, 4))
        torch.testing.assert_close(result[4], torch.tensor([1.0, 0.0]))
        self.assertEqual(result[5], ["pair_1", "pair_2"])

    @mock.patch("ctc_data._load_waveform")
    def test_score_dataset_reads_query_audio_only(self, load_waveform):
        load_waveform.return_value = torch.ones(8)
        dataset = CTCScoreDataset(
            [{"id": "pair_9", "enroll_text": "target", "label": 0}],
            "unused.zip", AudioConfig(), max_samples=10)
        item = dataset[0]
        self.assertEqual(item[1], "target")
        load_waveform.assert_called_once()
        self.assertEqual(load_waveform.call_args.args[1],
                         "wav/pair_9_query.wav")

    def test_training_cli_supports_full_data_paths(self):
        args = parse_args([
            "--train-zip", "train/wav.zip",
            "--train-csv", "train/train_label.csv",
            "--resume", "checkpoint.pt",
            "--last-out", "latest.pt",
        ])
        self.assertEqual(args.train_zip, "train/wav.zip")
        self.assertEqual(args.train_csv, "train/train_label.csv")
        self.assertIsNone(args.subset)
        self.assertEqual(args.resume, "checkpoint.pt")
        self.assertEqual(args.last_out, "latest.pt")

    def test_resume_checkpoint_paths_and_compatibility(self):
        self.assertEqual(
            default_last_checkpoint_path("checkpoints/model.pt"),
            "checkpoints/model.last.pt")
        self.assertEqual(
            default_last_checkpoint_path("checkpoints/model"),
            "checkpoints/model.last.pt")

        args = parse_args([
            "--model-id", "local/wavlm",
            "--units", "char",
            "--train-zip", "train/wav.zip",
            "--train-csv", "train/train_label.csv",
            "--bs", "128",
        ])
        args.workers = 0
        config = training_config(
            args, max_samples=40000, train_utterances=1000000,
            amp_enabled=False, device=torch.device("cpu"))
        vocabulary = CharacterVocabulary()

        legacy_checkpoint = {
            "model_id": "local/wavlm",
            "units": "char",
            "vocabulary": vocabulary.symbols,
            "train_zip": "train/wav.zip",
            "train_csv": "train/train_label.csv",
            "train_utterances": 1000000,
            "max_samples": 40000,
            "dropout": 0.1,
        }
        validate_resume_checkpoint(
            legacy_checkpoint, config, vocabulary)

        new_checkpoint = dict(legacy_checkpoint)
        new_checkpoint["training_config"] = dict(config)
        new_checkpoint["training_config"]["batch_size"] = 256
        with self.assertRaisesRegex(ValueError, "batch_size"):
            validate_resume_checkpoint(
                new_checkpoint, config, vocabulary)


class CTCScoreTest(unittest.TestCase):
    def test_matches_torch_ctc_loss_with_repeated_characters(self):
        torch.manual_seed(23)
        vocabulary = CharacterVocabulary()
        targets = torch.stack([
            vocabulary.encode("letter"),
            vocabulary.encode("better"),
        ])
        target_lengths = torch.tensor([6, 6])
        input_lengths = torch.tensor([12, 10])
        logits = torch.randn(2, 12, len(vocabulary), dtype=torch.float64)
        log_probs = logits.log_softmax(dim=-1)

        expected_loss = torch.nn.CTCLoss(
            blank=vocabulary.blank_id,
            reduction="none",
            zero_infinity=False,
        )(
            log_probs.transpose(0, 1),
            targets,
            input_lengths,
            target_lengths,
        )
        actual = ctc_log_probability(
            log_probs, input_lengths, targets, target_lengths,
            vocabulary.blank_id)
        torch.testing.assert_close(actual, -expected_loss)
        torch.testing.assert_close(
            normalized_ctc_score(
                log_probs, input_lengths, targets, target_lengths,
                vocabulary.blank_id),
            -expected_loss / target_lengths,
        )

    def test_valid_mask_rejects_impossible_repeated_target(self):
        vocabulary = CharacterVocabulary()
        targets = vocabulary.encode("letter").unsqueeze(0)
        target_lengths = torch.tensor([6])
        self.assertFalse(ctc_valid_mask(
            torch.tensor([6]), targets, target_lengths).item())
        self.assertTrue(ctc_valid_mask(
            torch.tensor([7]), targets, target_lengths).item())

    def test_collect_scores_preserves_ids_labels_and_raw_scores(self):
        vocabulary = CharacterVocabulary()
        torch.manual_seed(31)
        log_probs = torch.randn(2, 5, len(vocabulary)).log_softmax(dim=-1)
        output_lengths = torch.tensor([5, 4])

        class FakeModel:
            def eval(self):
                return self

            def log_probs(self, waveforms, sample_lengths):
                return log_probs, output_lengths

        targets = torch.stack([
            vocabulary.encode("a"), vocabulary.encode("b")])
        target_lengths = torch.tensor([1, 1])
        loader = [(
            torch.zeros(2, 8), torch.tensor([8, 7]), targets,
            target_lengths, torch.tensor([1.0, 0.0]), ["pair_1", "pair_2"],
        )]
        rows = collect_scores(
            FakeModel(), loader, torch.device("cpu"), False,
            vocabulary.blank_id)
        expected = normalized_ctc_score(
            log_probs, output_lengths, targets, target_lengths,
            vocabulary.blank_id)
        self.assertEqual([row[0] for row in rows], ["pair_1", "pair_2"])
        self.assertEqual([row[2] for row in rows], [1, 0])
        np.testing.assert_allclose(
            [row[1] for row in rows], expected.numpy(), rtol=1e-6)


class FrozenWavLMCTCTest(unittest.TestCase):
    def test_fake_backbone_forward_freeze_and_head_state(self):
        class FakeBackbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))
                self.config = types.SimpleNamespace(
                    hidden_size=8, num_hidden_layers=2)

            def forward(self, waveforms, attention_mask,
                        output_hidden_states, return_dict):
                base = waveforms[:, ::2].unsqueeze(-1).repeat(1, 1, 8)
                return types.SimpleNamespace(
                    hidden_states=tuple(base * self.weight + i
                                        for i in range(3)))

            def _get_feat_extract_output_lengths(self, lengths):
                return torch.div(lengths + 1, 2, rounding_mode="floor")

        backbone = FakeBackbone()

        class FakeAutoModel:
            @staticmethod
            def from_pretrained(model_id):
                self.assertEqual(model_id, "fake/wavlm")
                return backbone

        transformers = types.ModuleType("transformers")
        transformers.AutoModel = FakeAutoModel
        with mock.patch.dict(sys.modules, {"transformers": transformers}):
            model = FrozenWavLMCTC(28, "fake/wavlm", dropout=0.0)

        model.train()
        self.assertFalse(model.backbone.training)
        waveforms = torch.randn(2, 12)
        lengths = torch.tensor([12, 8])
        log_probs, output_lengths = model.log_probs(waveforms, lengths)
        self.assertEqual(log_probs.shape, (2, 6, 28))
        torch.testing.assert_close(output_lengths, torch.tensor([6, 4]))
        log_probs.sum().backward()
        self.assertIsNone(model.backbone.weight.grad)
        self.assertTrue(all(parameter.grad is not None
                            for parameter in model.head.parameters()))
        self.assertTrue(all(not key.startswith("backbone.")
                            for key in model.head_state_dict()))


if __name__ == "__main__":
    unittest.main()
