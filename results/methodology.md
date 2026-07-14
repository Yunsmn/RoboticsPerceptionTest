# Methodology — how these numbers are produced

## The question
Can a **single fixed camera** plus **one monocular metric-depth model** locate a 3 cm cube
accurately enough to grasp it, with no triangulation, no RGB-D sensor, no support-plane prior,
and no colour thresholding? "Accurately enough" means a localisation error at or below roughly
2 mm — the point where the grasp becomes reliable in our earlier pick-and-place work.

## The data
- **48 frames** rendered from the SO-101 MuJoCo simulation, `640 × 480`, focal length `f = 579.4 px`.
- Each frame carries the cube's **exact** centre in three forms: `gt_xyz` (world/base metres),
  `gt_uv` (the exact pixel that centre projects to), and `gt_depth_m` (its depth along the camera
  optical axis).
- Ground truth is used **only** to (a) know which pixel to read the model at and (b) score the
  error. It never enters an estimate.

## The scoring pipeline
For each frame: read the model's depth map at the known cube pixel → back-project that pixel and
depth to a 3D point via `camera_math.point_at_depth` → measure the Euclidean distance (mm) to the
true `gt_xyz`. We report the **median** and **95th percentile (p95)** over the 48 frames. Median is
the typical case; p95 is the worst realistic case, which is what actually breaks a grasp.

## Why the numbers can be trusted (model-free self-checks)
Before any model runs, two checks prove the scoring geometry is exact:

| Check | What it proves | Result |
|---|---|---|
| A: project ↔ deproject | The back-projection math is the exact inverse of the projection | max error **2.96e-16 m** |
| B: stored `gt_uv` vs reprojection | The manifest's stored pixels really are the projected centres | max error **0.0006 px** |

So any error we report is the **depth model's**, not the harness's.

## The three ways we report each model
- **raw** — no correction; the model's native metric output.
- **oracle-scaled** — one global scale, computed by peeking at *all* the ground-truth depths
  (`median(gt / pred)`). This **cheats** and is only an upper bound: "if your scale were perfect,
  this is the best the model could do."
- **anchored (BASE / FLOOR)** — the **honest** version. Instead of the oracle's all-knowing scale,
  recover the scale from a point we legitimately know:
  - *base anchor*: the robot base is the origin `(0,0,0)`; it projects to pixel `(320, 256)` with a
    true depth of `1.173 m`. Scale `= 1.173 / model_depth_at_base_pixel`.
  - *floor anchor*: the floor is `z = 0` (known from the camera-to-base calibration), so a grid of
    **476** floor pixels each has an analytic true depth. The scale is the robust median over all of
    them. Denser than one point, so steadier.
  Both use only robot self-knowledge — no cube truth, no assumed plane height.

## Environment
Google Colab, **Tesla T4** GPU, Python 3.12. One depth model per session (their dependencies
conflict). Full raw logs are in [`../logs/`](../logs/); depth-map images in [`../viz/`](../viz/).
