# OpenCV pip-counting experiment

This directory is intentionally separate from `experiments/`, which contains
the earlier GPT-vision work.

## Question being tested

Can the existing local standing/fallen models provide domino geometry while a
fully local OpenCV pipeline reads the pip counts accurately enough for the
robot?  The two stages must be scored separately:

1. `weights/bestieee.pt` and `weights/fallen.pt` propose tile boxes and posture.
2. Each box is perspective-rectified and split at its geometric midpoint.
3. The existing classical OpenCV blob/layout code counts each half from 0–6.

No OpenAI API call is made and no new model is trained by this experiment.

## Run one image

From the repository root:

```bash
.venv/bin/python -m experiment_OpenCVpip.run_experiment \
  experiments/test_images/IMG_9520.jpg
```

## Run a directory

```bash
.venv/bin/python -m experiment_OpenCVpip.run_experiment \
  experiments/test_images \
  --output-dir experiment_OpenCVpip/results/mixed
```

Useful options:

```text
--detector-conf 0.35       confidence threshold for standing/fallen detection
--pip-conf 0.22            minimum confidence required on each half
--save-crops               save full/end crops and OpenCV debug overlays (default)
--no-save-crops            write only JSON and annotated full images
--standing-weights FILE    defaults to bestieee.pt in weights/
--fallen-weights FILE      defaults to fallen.pt in weights/
```

Outputs include:

- `<image>.opencv-pip.json`: per-candidate geometry, posture, counts, and confidence;
- `<image>.opencv-pip.png`: detector boxes with OpenCV labels;
- `crops/<image>/`: rectified crops and accepted/rejected pip overlays;
- `summary.json`: batch execution summary (not an accuracy score).

## Important baseline behavior

This first version deliberately preserves all detector candidates.  It does
not perform additional cross-model IoU deduplication or split merged boxes.
That makes duplicate, missed, and merged detections measurable instead of
hiding them inside pip-recognition post-processing.

The first benchmark should record these metrics independently:

1. real-tile recall;
2. duplicate and false-positive candidates;
3. merged candidates spanning more than one physical tile;
4. half-pip accuracy when a candidate isolates exactly one tile;
5. exact full-tile accuracy;
6. exact final image-set accuracy.

## Current assumptions and next step

The midpoint split assumes the detector quad tightly encloses one complete
domino and that the divider is near the geometric center.  Debug crops make
violations visible.  If correct detector boxes frequently split off-center,
the next change should detect the black center divider in the rectified crop.
If duplicate standing/fallen boxes dominate, add rotated-IoU deduplication as a
separate, measured experiment rather than changing this baseline.

## Tests

```bash
.venv/bin/python -m unittest experiment_OpenCVpip.test_opencv_pip
```

## Score the existing 41-image benchmark

After running both source directories, generate a reproducible report:

```bash
.venv/bin/python -m experiment_OpenCVpip.evaluate_results \
  experiment_OpenCVpip/results/single-baseline \
  experiment_OpenCVpip/results/mixed-baseline \
  --output experiment_OpenCVpip/results/REPORT.md
```

`ground_truth.json` contains the manually inferred values carried over from the
earlier reports. Confirm them with the photographer before using the numbers as
a formal benchmark.

## Run the manual oracle-crop experiment

After completing the four-corner annotations, bypass the detector and score
OpenCV directly on the rectified crops:

```bash
.venv/bin/python -m experiment_OpenCVpip.run_oracle_experiment
```

The report, structured results, and per-image debug panels are written to
`experiment_OpenCVpip/results/oracle-single/` by default.
