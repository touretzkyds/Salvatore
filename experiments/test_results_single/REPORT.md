# Single-Domino GPT-5.6 Experiment Report

## Experiment Configuration

- Input: 10 iPhone photos from `experiments/test_images_single`.
- Every photo contains exactly one standing domino.
- Bounding boxes and posture: existing `Standing.pt` and `Fallen.pt` models.
- Pip classification: `gpt-5.6-sol`, image detail `original`, reasoning effort `medium`.
- Each GPT request received the full, end-0, and end-1 crop for every detector candidate.
- Domino orientation is ignored during scoring.

## Results

| Metric | Result |
|---|---:|
| Input images / real dominoes | 10 |
| Images with at least one detected candidate | 10 / 10 |
| Detector candidates | 17 |
| Standing candidates | 11 / 17 |
| Incorrect fallen candidates | 6 / 17 |
| GPT-readable candidates | 15 / 17 |
| GPT unknown candidates | 2 / 17 |
| Correct values among readable candidates | 15 / 15 (100%) |
| Images containing at least one correct GPT result | 10 / 10 (100%) |
| Raw output exactly correct with one candidate only | 4 / 10 (40%) |
| Highest-confidence candidate correct | 10 / 10 (100%) |

## Per-Image Results

| Image | Ground truth | Candidate posture and GPT result | Highest-confidence result |
|---|---|---|---|
| IMG_9540 | 4-4 | standing 4-4; fallen 4-4 | 4-4 standing |
| IMG_9541 | 0-4 | standing 0-4; fallen 0-4; standing unknown false positive | 0-4 standing |
| IMG_9542 | 5-3 | standing 5-3 | 5-3 standing |
| IMG_9543 | 6-5 | standing 6-5 | 6-5 standing |
| IMG_9544 | 5-5 | standing 5-5; fallen 5-5 | 5-5 standing |
| IMG_9545 | 4-3 | standing 4-3 | 4-3 standing |
| IMG_9546 | 6-1 | standing 6-1 | 6-1 standing |
| IMG_9547 | 5-2 | standing 5-2; fallen 5-2 | 5-2 standing |
| IMG_9548 | 6-6 | standing 6-6; fallen 6-6 | 6-6 standing |
| IMG_9549 | 1-5 | standing 1-5; fallen unknown | 1-5 standing |

## Interpretation

This experiment strongly supports the hypothesis that GPT pip classification works well when the bounding box contains one complete domino. Every readable domino crop was classified correctly, covering blank, non-double, and double tiles with pip counts from zero through six.

The remaining problem is detector post-processing rather than pip recognition. Six images received duplicate standing/fallen boxes for the same tile, and IMG_9541 also received a false-positive standing box over the monitor in the background. Consequently, only four images had an exactly correct raw output set even though every image contained a correct GPT result.

For this single-standing-domino batch, choosing the highest-confidence detector candidate would produce the correct value and posture on all ten images. This is encouraging but should not yet be generalized: in the earlier mixed dataset, IMG_9504 had an incorrect fallen detection with slightly higher confidence than the correct standing detection.

## Recommended Next Step

Use IoU-based cross-model deduplication to merge standing and fallen boxes that cover the same physical tile, while retaining GPT results and detector confidence. Then rerun both datasets and compare:

1. tile recall;
2. duplicate detections per image;
3. posture accuracy;
4. pip accuracy on valid isolated crops;
5. exact final domino-set accuracy.

The original detector and CNN recognition path should remain unchanged as a baseline. The deduplication should be implemented only in the GPT experiment layer first.

## Output Locations

- Per-image JSON, annotated images, logs, and crops: `experiments/test_results_single/gpt56/`
