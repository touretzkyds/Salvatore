# Detector-Free Full-Frame GPT-5.6 Experiment Report

## Experiment Configuration

- Model: `gpt-5.6-sol`.
- Input mode: one complete original-resolution photo per request.
- Image detail: `original`; reasoning effort: `medium`.
- No `Standing.pt`, `Fallen.pt`, CNN classifier, local bounding box, crop, or posture hint was used.
- GPT was asked to enumerate every physical domino exactly once and return its two pip counts, posture, and an approximate normalized image-space bounding box.
- Domino orientation is ignored during face-value scoring.

## Main Results

| Dataset | Images | Real tiles | GPT tiles | Correct tile values | Exact image sets | Correct posture |
|---|---:|---:|---:|---:|---:|---:|
| Single-domino | 10 | 10 | 10 | 10 / 10 (100%) | 10 / 10 (100%) | 10 / 10 (100%) |
| Original mixed set | 31 | 55 | 55 | 48 / 55 (87.3%) | 24 / 31 (77.4%) | 52 / 55 (94.5%) |
| Combined | 41 | 65 | 65 | 58 / 65 (89.2%) | 34 / 41 (82.9%) | 62 / 65 (95.4%) |

Across all 41 images, direct full-frame GPT returned exactly the correct number of physical tiles. There were no duplicate domino results, false-positive objects, unreadable results, or count-level misses in this dataset.

## Comparison with the Detector-Assisted Pipeline

| Metric | Existing `.pt` boxes + GPT crops | Direct full-frame GPT |
|---|---:|---:|
| Single-domino exact images | 4 / 10 (40%) | 10 / 10 (100%) |
| Mixed-set candidates for 55 real tiles | 75 | 55 |
| Mixed-set correct tile values | 31 / 55 (56.4%) | 48 / 55 (87.3%) |
| Mixed-set exact image sets | 3 / 31 (9.7%) | 24 / 31 (77.4%) |

The direct method eliminated the previous pipeline's main failure modes: duplicate standing/fallen boxes and crops spanning two adjacent dominoes.

## Single-Domino Results

| Image | Ground truth | Direct GPT | Posture |
|---|---|---|---|
| IMG_9540 | 4-4 | 4-4 | standing |
| IMG_9541 | 0-4 | 0-4 | standing |
| IMG_9542 | 5-3 | 5-3 | standing |
| IMG_9543 | 6-5 | 6-5 | standing |
| IMG_9544 | 5-5 | 5-5 | standing |
| IMG_9545 | 4-3 | 4-3 | standing |
| IMG_9546 | 6-1 | 6-1 | standing |
| IMG_9547 | 5-2 | 5-2 | standing |
| IMG_9548 | 6-6 | 6-6 | standing |
| IMG_9549 | 1-5 | 1-5 | standing |

## Multi-Domino Pip Errors

The other 24 mixed-set images were completely correct. The seven images below each contained one incorrectly counted tile; the number of tiles and tile separation were still correct.

| Image | Ground truth | Direct GPT result | Error |
|---|---|---|---|
| IMG_9516 | 3-4, 4-4 | 3-6, 4-4 | 3-4 read as 3-6 |
| IMG_9520 | 1-3, 6-5 | 1-3, 6-6 | 6-5 read as 6-6 |
| IMG_9521 | 1-3, 6-5 | 1-3, 6-6 | 6-5 read as 6-6 |
| IMG_9524 | 5-5, 4-6 | 5-6, 4-6 | 5-5 read as 5-6 |
| IMG_9527 | 5-5, 4-6 | 5-4, 4-6 | 5-5 read as 5-4 |
| IMG_9529 | 2-2, 1-5 | 2-2, 1-6 | 1-5 read as 1-6 |
| IMG_9531 | 2-2, 1-5 | 2-2, 1-4 | 1-5 read as 1-4 |

The pip errors are all single-half, off-by-one or off-by-two mistakes in angled views. GPT did not merge adjacent tiles in any of these cases.

## Posture Errors

- IMG_9514 contains one fallen `6-6`, but GPT labeled it standing.
- IMG_9531 contains two fallen tiles, but GPT labeled both standing.
- All other 62 tile postures were correct.

## Bounding-Box Limitation

GPT's normalized boxes are useful for visual association and avoided merging the two physical tiles, but they are approximate and sometimes include substantial background or overlap. This experiment does not establish that GPT-generated boxes are precise enough for calibrated pixel-to-world projection.

If the robot already has trustworthy world-map positions from a separate Python geometry process, direct GPT can provide the observed tile count, face values, posture, and approximate image association. If all `.pt` localization is removed, a separate evaluation is still needed to determine whether GPT box centers can be matched reliably to world-map objects.

## Recommended Next Step

The strongest next experiment is repeated direct inference on the seven failed multi-domino images. Run three independent calls and accept a tile value only when at least two calls agree. This will measure whether the remaining pip mistakes are random and can be reduced through consensus.

After that, test direct GPT bounding-box centers against manually labeled pixel centers or known world-map projections. Pip recognition and geometric localization should be reported as separate metrics.

## Output Locations

- Multi-domino JSON, logs, and annotated images: `experiments/test_results_direct/multi/`
- Single-domino JSON, logs, and annotated images: `experiments/test_results_direct/single/`
