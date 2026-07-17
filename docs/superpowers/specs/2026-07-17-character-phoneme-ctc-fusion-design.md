# Character And Phoneme CTC Fusion Design

## Goal

Fuse the leaderboard-validated character CTC system (`0.81103`) and phoneme
CTC system (`0.83939`) without retraining WavLM. Use labeled dev data to choose
one reproducible fusion weight, then apply that fixed weight to the existing
character and phoneme eval prediction CSV files.

This experiment must preserve the phoneme-only result as an available
baseline. A fusion submission is recommended only if its dev mean AUC exceeds
the phoneme-only dev mean AUC of `0.8406`.

## Dev Score Export

Add a CTC score-export entry point that accepts one character or phoneme
checkpoint and a split. The first version needs the labeled dev split and
writes one row per pair with:

```text
id,subset,score,label
```

`score` is the raw length-normalized CTC log probability, before sigmoid. The
exporter uses only `enroll_txt` and query audio. It must not expose or read dev
query text. Score seen and unseen independently but write one combined CSV with
the same globally unique ids used by submission files.

The exporter reuses the existing frozen WavLM loader, checkpoint unit metadata,
CTC-valid target handling, and query-only score data loader. It verifies that
the checkpoint vocabulary matches the selected character or phoneme units.

## Rank Fusion

The two CTC branches have different target lengths and score distributions, so
their raw values and sigmoid posteriors are not directly comparable. Convert
each branch to percentile ranks independently within seen and unseen:

```text
rank = average_rank(score) / (number_of_rows - 1)
```

Use average ranks for tied values, including impossible CTC targets assigned
the shared score floor. The transform stays in `[0, 1]` and preserves each
branch's ordering.

For phoneme weight `w`, calculate:

```text
fused = (1 - w) * character_rank + w * phoneme_rank
```

Search one global `w` from `0.000` through `1.000` in increments of `0.001`.
For every candidate, compute seen AUC, unseen AUC, and their arithmetic mean.
Choose the highest mean AUC. On an exact tie, choose the weight closest to
`1.0`, retaining more of the stronger phoneme model and reducing unnecessary
dependence on the weaker branch.

Do not search separate seen and unseen weights in the first experiment. One
global parameter has lower dev-overfitting risk and matches the competition's
mean-of-subsets objective.

## Eval Submission

The fusion script accepts:

- Character dev score CSV.
- Phoneme dev score CSV.
- Existing character eval submission CSV.
- Existing phoneme eval submission CSV.

It validates exact id, subset, and label alignment for dev; exact id alignment
for eval; finite scores; unique ids; and identical row sets regardless of CSV
order. Eval files may call the numeric column `posterior` while dev files call
it `score`.

Apply the same per-subset rank transform to eval character and phoneme scores,
then use the single dev-selected weight. Write only:

```text
id,posterior
```

Preserve the official seen rows followed by unseen rows and verify all fused
posteriors lie in `[0, 1]`. Also write a small JSON report beside the submission
containing the selected weight, dev branch AUCs, fused AUCs, source paths, and
row count.

## Verification

- Unit-test average ranks with ties and constant inputs.
- Unit-test id-based alignment when input CSV order differs.
- Reject duplicate ids, missing ids, label disagreement, non-finite scores,
  and unknown id prefixes.
- Unit-test weight search on synthetic complementary predictions and verify
  the phoneme-favoring tie break.
- Unit-test fused posteriors stay within `[0, 1]` and output order is official.
- Run the complete existing test suite.
- On A10, export character and phoneme dev scores, run fusion against the two
  existing eval submissions, and inspect the JSON report before submission.

## Later Epoch Comparison

The requested 10-epoch versus 20-epoch comparison is a later experiment. It
must keep data, seed, batch size, augmentation, model, and learning-rate policy
identical, and compare best dev AUC through epoch 10 against best dev AUC
through epoch 20. It is not part of this fusion implementation.
