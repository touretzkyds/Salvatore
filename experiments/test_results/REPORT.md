# Domino GPT-5.6 Real-Image Experiment Report

## Experiment Configuration

- Input: 31 iPhone photos from `experiments/test_images`.
- Stage 1: The existing `Standing.pt` and `Fallen.pt` models detect domino bounding boxes and posture.
- Stage 2: `gpt-5.6-sol` reads the full, end-0, and end-1 crops for each candidate and returns the pip counts on both halves.
- Image detail: `original`; reasoning effort: `medium`.
- Domino orientation is ignored during scoring. For example, `4-3` and `3-4` are treated as the same tile.
- Ground truth was inferred through manual visual inspection. It should be confirmed by the person who took the photos before this is used as a formal benchmark.

## Summary

| Metric | Result |
|---|---:|
| Input images | 31 |
| Ground-truth domino instances | 55 |
| Candidates produced by the existing detector | 75 |
| GPT-readable candidates | 65 |
| Candidates returned as unknown | 10 |
| Correct domino instances recovered end to end | 31 / 55 (56.4%) |
| Candidate boxes matching a ground-truth tile | 31 / 75 (41.3%) |
| Images with all ground-truth tiles recovered, allowing extra detections | 15 / 31 |
| Images with an exactly correct output set and no extra candidates | 3 / 31 |

The 56.4% result is not a standalone measure of GPT's visual-recognition accuracy. The main bottleneck is currently the first-stage bounding boxes. Adjacent dominoes are often merged into one candidate, and the same domino can receive duplicate standing and fallen detections. GPT then reads the visible ends inside the incorrect crop, producing a cross-tile combination that does not correspond to a real domino.

The seven single-domino images (IMG_9504–9506 and IMG_9511–9514) provide a cleaner comparison. The detector produced 15 candidates for these images. GPT marked 12 as readable, and all 12 readable candidates had the correct face value. The other three were returned as `unknown`; there were no readable but incorrectly classified candidates in this subset. This suggests that GPT pip recognition is promising when a crop isolates one domino. However, the 12 readings include duplicate boxes produced by the detector, so they should not be presented as 12 independent test tiles or as a formal 100% accuracy result.

## Bounding Box and Posture Diagnosis

- The 55 real domino instances produced 75 candidate detections, creating 20 extra candidates while still missing some tiles.
- The 24 all-standing images contain 42 real dominoes but produced 67 candidates: 41 were labeled standing and 26 were incorrectly labeled fallen.
- The seven all-fallen images contain 13 real dominoes but produced only eight candidates. All eight detected candidates were labeled fallen, but five real tiles were missed.
- If posture is scored at the candidate level using the posture shared by every tile in each image, 49 of 75 candidate labels are correct (65.3%). This figure is affected by duplicate detections and is useful only for diagnosis; it is not instance-level posture accuracy.

Typical merged-box failures include:

- IMG_9520 merges parts of `1-3` and `6-5`, causing GPT to return `1-5`.
- IMG_9524 merges `5-5` and `4-6`, causing GPT to return `5-6`.
- IMG_9531 merges `2-2` and `1-5`, causing GPT to return `3-6`.
- IMG_9533 merges `0-0` and `1-1`, causing GPT to return `0-1`.

## Per-Image Results

The “GPT candidates” column retains duplicate detections and `?` (`unknown`) results. It is therefore not a deduplicated final domino set.

| Image | Manually inferred ground truth | GPT candidates | Recovered |
|---|---|---|---:|
| IMG_9504 | 4-0 | 4-0, 4-0 | 1/1 |
| IMG_9505 | 4-0 | 4-0, 4-0, ? | 1/1 |
| IMG_9506 | 4-0 | 4-0, 4-0 | 1/1 |
| IMG_9507 | 4-0, 5-0 | 5-0, 4-5, 4-5, 4-0 | 2/2 |
| IMG_9508 | 4-0, 5-0 | 4-5, 4-5 | 0/2 |
| IMG_9509 | 4-0, 5-0 | 4-5, 4-5, 4-3 | 0/2 |
| IMG_9510 | 4-0, 5-0 | 5-0 | 1/2 |
| IMG_9511 | 6-6 | 6-6, 6-6, ? | 1/1 |
| IMG_9512 | 6-6 | 6-6, ? | 1/1 |
| IMG_9513 | 6-6 | 6-6, 6-6 | 1/1 |
| IMG_9514 | 6-6 | 6-6 | 1/1 |
| IMG_9515 | 3-4, 4-4 | ?, 4-4, 3-4, ?, ? | 2/2 |
| IMG_9516 | 3-4, 4-4 | 3-4, 3-4 | 1/2 |
| IMG_9517 | 3-4, 4-4 | 3-4, 4-4 | 2/2 |
| IMG_9518 | 3-4, 4-4 | 4-4 | 1/2 |
| IMG_9519 | 1-3, 6-5 | 1-3, 1-3, 6-5, 1-3 | 2/2 |
| IMG_9520 | 1-3, 6-5 | 1-5 | 0/2 |
| IMG_9521 | 1-3, 6-5 | 1-5, ?, 1-5 | 0/2 |
| IMG_9522 | 1-3, 6-5 | 6-5 | 1/2 |
| IMG_9523 | 5-5, 4-6 | ?, 4-6, ?, 5-5, ? | 2/2 |
| IMG_9524 | 5-5, 4-6 | 5-6, 5-6 | 0/2 |
| IMG_9526 | 5-5, 4-6 | 5-5, 4-6 | 2/2 |
| IMG_9527 | 5-5, 4-6 | 4-6 | 1/2 |
| IMG_9528 | 2-2, 1-5 | 1-5, 4-6, 2-2, 4-6 | 2/2 |
| IMG_9529 | 2-2, 1-5 | 2-5, 2-5, 1-5 | 1/2 |
| IMG_9530 | 2-2, 1-5 | 4-6, 4-6, 2-2 | 1/2 |
| IMG_9531 | 2-2, 1-5 | 3-6 | 0/2 |
| IMG_9532 | 0-0, 1-1 | 0-2, 1-1, 0-2, 0-0 | 2/2 |
| IMG_9533 | 0-0, 1-1 | 0-1, 0-1 | 0/2 |
| IMG_9534 | 0-0, 1-1 | 0-2, 0-1 | 0/2 |
| IMG_9535 | 0-0, 1-1 | 1-1, 1-1 | 1/2 |

## Recommended Next Experiment

1. Keep the current GPT configuration and evaluate bounding boxes separately from pip recognition. Manually crop a small labeled set of individual dominoes and pass those crops directly to the GPT recognizer to measure true pip-classification accuracy.
2. Apply cross-model IoU deduplication to the standing and fallen detections instead of relying only on center distance. However, the incorrect fallen box in IMG_9504 has slightly higher confidence than the correct standing box, so keeping only the highest-confidence box will not be sufficient; posture or shape constraints are also needed.
3. Reject or split candidate boxes that span two dominoes. Potential signals include the center divider, contour aspect ratio, and expected spacing between tiles in world coordinates.
4. Preserve the existing CNN recognition route as a baseline and compare CNN and GPT against the same confirmed ground-truth dataset.

## Output Locations

- Detections and crops: `experiments/test_results/detection/`
- GPT JSON and annotated images: `experiments/test_results/gpt56/`
- Contact sheet: `experiments/test_results/contact_sheet.jpg`
