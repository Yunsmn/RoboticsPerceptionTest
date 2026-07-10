# SO-101 Perception Benchmark

Standalone benchmark for the SO-101 perceptual pick-and-place pipeline. It runs the
**`find_object`** components — **SAM 3** (open-vocab detection) and a **monocular metric-depth
model** (Depth Pro / UniDepthV2 / Metric3D-v2) — on simulation frames whose object centre is
known **exactly**, and scores each against ground truth so the model is chosen on measured
error, not intuition.

Meant to be cloned and run on **Google Colab** (GPU). You run it; when a cell breaks, tell the
author which cell + the error, they push a fix, you `git pull`.

## What it measures

Two things, kept separate so we know where error comes from:

1. **Detection (SAM 3)** — a text prompt (`"red cube"`) → predicted centre pixel, scored against
   the exact projected pixel (pixel error + detection rate).
2. **Metric depth** — depth at a pixel → back-project with `camera_math.point_at_depth` → world
   `(x, y, z)`, scored against the exact world centre, both:
   - at the **GT pixel** (isolates depth-model accuracy), and
   - at the **SAM pixel** (the full detection + depth pipeline).

The bar is the **~2 mm grasp basin** at the ~1.15 m side-camera standoff. `benchmark.ipynb`
prints the new contenders next to the pipelines we already measured (HSV+plane ~2 mm,
SAM+plane ~1.7 mm, mono Depth-Anything ~12 mm, 3-cam triangulation ~1.1 mm).

## Honesty

Ground truth is used **only** to know which pixel is the true centre and to score error — it
never drives an estimate. Every depth backend must return depth in **metres along the camera
optical/forward axis** (the convention `camera_math.point_at_depth` consumes).

## Layout

```
benchmark.ipynb     # the Colab notebook — run top to bottom
camera_math.py      # pure-numpy camera model (projection + point_at_depth), standalone
data/
  manifest.json     # camera intrinsics/extrinsics + per-frame {png, gt_xyz, gt_uv, gt_depth_m}
  frames/pose_*.png # rendered side-camera frames (cube at known poses)
reference_results.md  # our earlier measured pipeline numbers, to compare against
requirements.txt    # per-backend install notes (they conflict — one at a time)
```

## Run on Colab

1. `!git clone https://github.com/Yunsmn/RoboticsPerceptionTest.git` and open `benchmark.ipynb`.
2. Runtime → GPU.
3. Run the setup + data cells, then install **one** backend (cell 1) and run its adapter + eval.
4. Repeat per backend; read the comparison table.

## The data

Generated once from the main SO-101 repo (needs the sim):

```
MUJOCO_GL=egl venv/bin/python experiments/export_depth_benchmark_data.py \
    --out depth_bench --nx 5 --ny 9 --raised
```

The projection of each `gt_xyz` to its `gt_uv` self-checks to ~1e-16 m, so the centre labels
are exact. `--raised` adds off-plane-height cases.
