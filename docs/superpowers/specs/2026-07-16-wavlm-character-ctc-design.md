# Frozen WavLM Character CTC Design

## Goal

Test whether character-level supervision can recover unseen-keyword
generalization. Train a character CTC head from competition-provided
audio-text pairs, then score each dev pair using only the enrollment text and
query audio available at test time.

This stage reports CTC-only seen and unseen AUC. It does not fuse the existing
audio-pair matcher, use external pronunciation resources, fine-tune WavLM, or
use dev query text during scoring.

## Character Vocabulary

Use 28 output classes:

- Index 0: CTC blank.
- Indices 1-26: lowercase `a` through `z`.
- Index 27: apostrophe.

Reject unsupported or empty normalized text instead of silently dropping
characters. The complete visible train, dev, and eval enrollment text has been
verified to fit this vocabulary.

## Training Examples

Read the 50,000-row training CSV with both text columns preserved. Expand each
pair into two supervised utterances:

- `wav/{id}_enroll.wav` with `enroll_txt`.
- `wav/{id}_query.wav` with `query_txt`.

Deduplicate by waveform filename. This yields up to 100,000 distinct
audio-text examples while preventing duplicated rows from overweighting an
utterance.

Cap waveforms at 2.5 seconds, apply zero-mean/unit-variance normalization, and
dynamically pad each batch. During training, independently add real DEMAND
noise with probability 0.5 and SNR sampled uniformly from -10 to 5 dB. Dev and
eval audio remain unaugmented.

## Model And Loss

Load `microsoft/wavlm-base-plus` as a frozen feature extractor and keep it in
evaluation mode. Request all hidden states and combine them using trainable
softmax-normalized layer weights. Apply dropout followed by a trainable linear
classifier from WavLM hidden size 768 to 28 character logits.

Convert input sample lengths to WavLM output lengths with the backbone's
convolution length function. Train only the layer weights and character
classifier with PyTorch CTC loss:

- Blank index 0.
- Mean reduction.
- `zero_infinity=True`.

Reject a batch before loss calculation if any target needs more CTC frames than
its audio provides after accounting for adjacent repeated characters.

## CTC-Only Pair Scoring

For each dev or eval pair, encode only the query audio. Convert `enroll_txt` to
the character target sequence and calculate exact CTC log probability with the
forward dynamic program over blank and repeated-character transitions.

Use the following scalar as the pair score:

```text
ctc_score = log P(enroll_txt | query_audio) / len(enroll_txt)
```

This score is monotonic with the submitted posterior requirement for AUC. For
CSV output, fit a trainable or dev-fitted scalar calibration only after CTC-only
AUC is established; the initial experiment may apply sigmoid to the normalized
score because AUC depends only on ranking.

Development scoring must never read `query_txt`. The data loader for CTC pair
evaluation exposes only pair id, enrollment text, query waveform, label when
available, and query sample length.

## Training And Checkpoint

Add separate CTC training and inference entry points. The first A10 experiment
uses:

- Batch size 128, subject to measured memory.
- Three epochs over deduplicated training utterances.
- AdamW over CTC trainable parameters only.
- Initial learning rate 1e-3.
- CUDA automatic mixed precision.
- Eight data-loader workers.
- DEMAND noise probability 0.5 at -10 to 5 dB.

After every epoch, calculate CTC-only seen and unseen AUC. Select the checkpoint
by their arithmetic mean and log both values separately. Save only CTC layer
weights, vocabulary, WavLM model id, maximum samples, augmentation metadata,
and dev metrics; do not store WavLM weights.

## Verification

- Unit-test text normalization, vocabulary encoding, and unsupported text
  rejection.
- Unit-test training-pair expansion and waveform-filename deduplication.
- Unit-test dynamic waveform padding and target collation.
- Compare the custom CTC forward score against `torch.nn.CTCLoss` on synthetic
  examples, including repeated characters such as `letter`.
- Verify scores ignore padded audio frames and remain finite for valid targets.
- Verify dev pair scoring cannot access query text.
- Verify WavLM stays frozen and checkpoint state contains only the CTC head.
- Run existing regression tests and a fake-backbone end-to-end CTC smoke test
  locally without downloading WavLM.
- On A10, run 256 utterances for one epoch and complete seen/unseen scoring
  before launching the full three-epoch experiment.

## Decision Rule

If CTC-only unseen AUC clearly exceeds 0.50, freeze the CTC checkpoint and train
a small fusion head over CTC score and the existing audio-pair score. If unseen
remains random, inspect character error behavior and WavLM layer weighting
before adding an external phoneme dictionary.
