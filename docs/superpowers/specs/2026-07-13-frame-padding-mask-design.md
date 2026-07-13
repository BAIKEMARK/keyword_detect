# Frame Padding Mask Design

## Goal

Exclude padded log-mel frames from `frame_maxmean` matching so scores depend
only on frames originating from each audio sample. Keep the `global` model's
behavior unchanged.

## Data Flow

`PairDataset` records the valid spectrogram length before padding or truncation.
Each sample returns the enrollment and query lengths with their fixed-size
features. `collate` batches these lengths, and training, evaluation, and
inference pass them to the model.

The encoder has two time-axis max-pooling layers with kernel and stride 2.
Valid lengths are therefore downsampled twice with floor division and clamped
to at least one output frame.

## Matching

`FrameMaxMeanKWS` builds boolean masks from the downsampled lengths. For each
valid enrollment frame, its maximum similarity is taken only over valid query
frames; the result is averaged only over valid enrollment frames. The reverse
query-to-enrollment score is computed symmetrically, then the two scores are
averaged as before.

The `global` model accepts the length arguments but ignores them. Model layers
and state-dict keys do not change, so existing checkpoints remain loadable.
Their scores may change under masked inference and they are not substitutes for
retraining with the corrected matcher.

## Verification

- A synthetic unit test changes only padded values and verifies that masked
  frame matching produces the same score.
- A synthetic unit test verifies that changing a valid frame can change the
  score.
- A data test verifies that pre-padding lengths are capped at `max_frames`.
- Global and frame models complete forward and backward passes with the new
  batch interface.
- Existing frame checkpoints load, and a small train/evaluation smoke run
  completes.
