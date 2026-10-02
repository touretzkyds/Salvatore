# Manual oracle-crop annotations

This directory stores human-drawn domino-face quadrilaterals for experiments
that evaluate pip recognition independently from standing/fallen detection.

## `single_domino/`

- Source images: `experiments/test_images_single/IMG_9540.jpg` through
  `IMG_9549.jpg`.
- `annotations.json`: one four-corner face annotation, ordered pip pair, and
  posture per image.
- `crops/`: 600 x 300 perspective-rectified crops exported by the annotation
  tool.

The files were copied from
`/Users/yuyuanzhengjack/Desktop/Touretzky_Dev/Domino_OpenCV_Annotator/output/`.
The source copy was preserved.
