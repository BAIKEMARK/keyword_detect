# Online Frozen WavLM Design

## Goal

Build the first foundation-model experiment on the 50,000-pair training subset.
Use `microsoft/wavlm-base-plus` only as a frozen speech feature extractor and
train the competition decision layers from the provided labeled pairs. The
primary success signal is unseen dev AUC moving clearly above 0.50.

This version does not cache features, fine-tune WavLM, use enrollment text, mine
hard negatives, or switch to the 500,000-pair dataset.

## Data

Add a raw-waveform pair dataset beside the existing log-mel dataset. Existing
CNN commands and checkpoints remain unchanged.

- Read 16 kHz enrollment and query waveforms from the existing zip archives.
- Cap each waveform at 2.5 seconds (40,000 samples). This covers the observed
  dev maximum of 2.34 seconds.
- Dynamically pad enrollment and query waveforms to the longest waveform in the
  batch, with an upper bound of 2.5 seconds.
- Return sample lengths so WavLM output padding can be excluded from matching.
- Keep enrollment audio clean. Apply DEMAND noise to query audio with
  probability 0.5 and sample SNR uniformly from -10 to 5 dB.
- Fail before training when `--noise-dir` is supplied but contains no supported
  audio files. Do not silently fall back to Gaussian noise in the WavLM path.

## Model

Load `microsoft/wavlm-base-plus` through Transformers. Freeze every backbone
parameter, keep the backbone in evaluation mode, and run it without gradient
tracking. Concatenate enrollment and query waveforms along the batch dimension
so one WavLM call encodes both sides.

Request all hidden layers and combine them with trainable softmax-normalized
layer weights. Apply a trainable projection from 768 to 128 dimensions, then L2
normalize each output frame.

Convert input sample lengths to WavLM feature lengths using the backbone feature
extractor's convolution length calculation. Build boolean masks from these
lengths.

The trainable matcher computes:

1. Mean over valid enrollment frames of their maximum similarity to valid query
   frames.
2. The symmetric query-to-enrollment value.
3. Cosine similarity between masked mean-pooled enrollment and query vectors.

A small MLP receives these three values and outputs one logit. This learned
layer weighting, projection, and matcher are the competition decision model;
WavLM output is not used directly as the final decision.

## Training

Add a separate WavLM training entry point so the baseline remains reproducible.
The first A10 configuration is:

- Batch size 16.
- Three epochs on the 50,000-pair subset.
- AdamW over trainable parameters only.
- Learning rate 1e-3 and positive loss weight 4.0.
- CUDA automatic mixed precision.
- Eight data-loader workers.
- Query noise probability 0.5 with DEMAND at -10 to 5 dB.

Evaluate seen and unseen dev AUC after each epoch and select the checkpoint by
their arithmetic mean. Log WavLM model id, trainable and frozen parameter
counts, real-noise file count, per-epoch AUC, elapsed time, and peak CUDA memory.

## Checkpoint And Inference

Save only trainable model state, configuration, augmentation metadata, and best
dev AUC. Do not store frozen WavLM parameters in the checkpoint. Inference
reloads the named public backbone, loads the trained decision state, and writes
the existing `id,posterior` submission format for seen and unseen evaluation
sets.

The loader must reject mismatched or missing trainable keys. A missing local
WavLM download should fail with a message naming the required model id rather
than continuing with random weights.

## Verification

- Unit-test waveform truncation, dynamic padding, and returned sample lengths.
- Unit-test that query-only augmentation leaves enrollment unchanged.
- Unit-test that padded embedding values cannot change matching scores.
- Unit-test symmetric matching and finite backward gradients for every
  trainable component using synthetic embeddings, without downloading WavLM.
- Verify frozen backbone parameters are excluded from saved trainable state.
- On the A10 server, download WavLM once, run a small-batch forward smoke test,
  then train a small subset through one complete seen/unseen evaluation.
- Start the 50,000-pair, three-epoch experiment only after the smoke run passes
  and the log reports 144 real DEMAND files.

## Decision After The First Run

If unseen AUC remains near 0.50, inspect layer choice, augmentation, and hard
negatives before scaling data. If unseen AUC improves clearly, move next to
hard-negative construction and the 500,000-pair dataset; feature caching can be
introduced only when repeated online encoding becomes the measured bottleneck.
