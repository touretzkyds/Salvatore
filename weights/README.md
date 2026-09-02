# Model weights

Model weights are **not** stored in git (`*.pt` is gitignored). Drop the files
listed below into this directory before running with `domino=True`.

## Required for the default perception pipeline

`StateMachineProgram(domino=True)` loads `lab8.domino_world_detector`, which
needs exactly one file:

| File | Model | Purpose |
|---|---|---|
| `domino_segment.pt` | YOLO26-seg | Segments domino tiles in the camera image |

Pip counting (`domino_labeling=True`) uses `ClassicalDominoLabelProvider`, a
pure OpenCV pipeline, so it needs no extra weights.

## Optional: the dual standing/fallen pipeline

`aim_fsm/domino.py` is an alternate detector that also classifies whether a tile
is standing or fallen. It is not wired into `program.py`; use it directly if you
need fallen-tile handling.

| File | Model | Purpose |
|---|---|---|
| `bestieee.pt` | YOLO26-seg | Standing tile segmentation |
| `fallen.pt` | YOLO26-seg | Fallen tile segmentation |
| `different.pt` | EfficientNetB0 | Pip count on a standing tile's half face |
| `fallenhalf.pt` | EfficientNetB0 | Pip count on a fallen tile's half face |

## Optional: retraining

| File | Purpose |
|---|---|
| `yolo26s-seg.pt` | Pretrained COCO checkpoint to fine-tune from |

## Where these came from

All of the above were trained in the `vision-salvatore` project.
`domino_segment.pt` is that repo's `runs/segment/train5/weights/best.pt`; the
rest were at its `vex-aim-tools/` top level. Ask the vision team for a copy, or
retrain the segmenter with `lab8/train_segmenter.py`.

## Overriding paths

Pass an explicit path if your weights live elsewhere:

```python
StateMachineProgram(domino=True, domino_weights_path='/abs/path/to/weights.pt')
```

Relative paths are resolved against this directory.
