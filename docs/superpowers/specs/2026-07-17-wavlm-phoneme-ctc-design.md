# Frozen WavLM Phoneme CTC Design

## Goal

Add an independent phoneme-level CTC branch after the character CTC system
reached 0.81103 on the competition leaderboard. The new branch should improve
robustness to spelling-to-sound variation and provide a complementary score
for later fusion, while preserving the existing character checkpoint and
inference behavior.

This stage trains and evaluates phoneme CTC alone. It does not fuse scores,
fine-tune WavLM, inspect eval labels, or use query text during dev or eval
scoring.

## Pronunciation Front End

Use `g2p_en` to convert normalized English enrollment text into ARPAbet. The
package uses CMUdict for known words and its bundled G2P fallback for words not
found in the dictionary. This gives broader coverage than CMUdict alone while
avoiding a system-level eSpeak dependency.

Use a fixed inventory consisting of CTC blank plus the standard 39 English
ARPAbet phones. Remove lexical-stress suffixes `0`, `1`, and `2` so that, for
example, `AH0` and `AH1` both map to `AH`. Ignore whitespace separators and a
standalone apostrophe emitted for possessives such as `servants'`; reject empty
pronunciations and all other unexpected symbols. Do not derive the vocabulary
from train, dev, or eval text.

Cache pronunciations by normalized text within each process because the
50,000 training pairs contain many repeated keywords. The G2P output is an
input feature to the competition-trained CTC head and is never used directly
as the wake-up decision.

## Shared CTC Pipeline

Generalize the existing character vocabulary interface so both character and
phoneme vocabularies provide:

- `symbols` with blank at index 0.
- `blank_id`.
- `normalize(text)`.
- `encode(text)` returning a one-dimensional integer tensor.

Add `--units {char,phoneme}` to the existing WavLM CTC training entry point,
defaulting to `char`. Both modes continue to use the same training example
expansion, DEMAND augmentation, frozen WavLM backbone, trainable layer mixture,
linear CTC head, exact CTC score, and seen/unseen AUC calculation.

The phoneme experiment uses the same initial settings as the successful
character run: all 100,000 utterances, ten epochs, batch size 128, AdamW at
1e-3, 0.5 DEMAND probability at -10 to 5 dB, and best checkpoint selection by
mean dev AUC.

## Checkpoint And Inference Compatibility

New checkpoints store `units` as either `char` or `phoneme` and retain the
exact output vocabulary. Inference constructs the corresponding vocabulary,
then verifies its symbols against the checkpoint before loading the CTC head.

Existing character checkpoints do not contain `units`. Treat a missing value
as `char`, so the leaderboard-validated `wavlm_char_ctc_100k_e10.pt` remains
loadable without conversion. Keep existing command defaults and filenames for
character mode. Use explicit phoneme filenames for phoneme checkpoints and
submissions.

## Dependencies And Reproducibility

Add a pinned-compatible `g2p_en` dependency to `requirements.txt` and document
it as an external pronunciation resource in the README. Persist CMUdict and
the old and English-specific averaged perceptron taggers under
`/mnt/workspace/nltk_data`. If these resources are absent, fail before training
with a concrete setup message rather than failing inside a data-loader worker.
Pin NumPy below 2 because `g2p_en 2.1.0` produces numerical-overflow warnings
for almost every OOV prediction under NumPy 2.x.

The server smoke test must initialize the phoneme vocabulary in the main
process before workers start. The model path remains the persistent local
WavLM directory under `/mnt/workspace/models/wavlm-base-plus`.

## Verification

- Unit-test the fixed ARPAbet inventory and blank index.
- Unit-test stress removal and whitespace filtering with an injected fake G2P
  converter, without downloading external resources.
- Unit-test rejection of empty and unsupported G2P output.
- Verify the existing character vocabulary tests remain unchanged.
- Verify old character checkpoints default to character mode.
- Verify new phoneme checkpoints select the phoneme vocabulary and reject a
  mismatched vocabulary.
- Run the complete local unit-test suite without loading WavLM weights.
- On A10, run a 256-utterance, one-epoch phoneme smoke test through both dev
  subsets before launching the full ten-epoch run.

## Decision Rule

First submit phoneme CTC independently. Retain it for fusion if its unseen AUC
is meaningfully above 0.5 and its ranking errors are complementary to character
CTC. Do not replace the 0.81103 character system merely because phoneme CTC is
conceptually appealing. Fusion is a separate experiment selected on dev AUC
and submitted only after character-only and phoneme-only baselines are known.
