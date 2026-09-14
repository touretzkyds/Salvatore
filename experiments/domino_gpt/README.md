# Domino GPT pip-recognition experiment

This experiment keeps the existing local perception split:

1. `weights/bestieee.pt` detects standing dominoes.
2. `weights/fallen.pt` detects fallen dominoes.
3. Python keeps each detection's bounding quadrilateral, posture, center, and
   directed axis endpoints.
4. GPT-5.6 Sol receives rectified crops and supplies only the two pip counts.

`bestieee.pt` is the local copy of the feature branch's `standing.pt`; their
file hashes match. The local `fallen.pt` likewise matches that branch.

The old `different.pt` / `fallenhalf.pt` CNN classifiers are unchanged and
remain the default behavior of `aim_fsm.domino.DominoWorldDetector`. The GPT
path explicitly starts that detector with `label_mode="none"`.

## Run one image

Set `OPENAI_API_KEY` in the shell, then run:

```bash
.venv/bin/python -m experiments.domino_gpt.run_experiment /absolute/path/to/frame.jpg \
  --save-crops /absolute/path/to/crops
```

The command writes these beside the input unless output paths are supplied:

- `frame.domino-gpt.png`: boxes, posture, directed ends, and GPT labels
- `frame.domino-gpt.json`: geometry and ordered `end_0_pips` / `end_1_pips`

To inspect only local standing/fallen detection and rectified crops without an
API call:

```bash
.venv/bin/python -m experiments.domino_gpt.run_experiment /absolute/path/to/frame.jpg \
  --detection-only --save-crops /absolute/path/to/crops
```

For a more conservative comparison, `--samples 3` makes three independent API
calls and returns a label only when the same ordered pair has a strict majority.
This costs roughly three times as much and is intended for evaluation, not the
live camera loop.

## End-order contract

The red endpoint marked `0` in the output image is `half_counts[0]`; the green
endpoint marked `1` is `half_counts[1]`. Crops are perspective-rectified so END
0 is always on the left and END 1 on the right. Values are never numerically
sorted. This matches `DominoBridge.end_values()` and lets world-map geometry
attach each count to the correct physical end.

## Current limitation

This first version is deliberately a single-frame experiment. It must not be
called on every live camera frame: a network request would block perception and
incur repeated cost. The next integration step, after measuring accuracy on
real robot frames, is a background label worker triggered only for stable,
unlabelled world-map objects, with a crop cache and retry/consensus policy.
