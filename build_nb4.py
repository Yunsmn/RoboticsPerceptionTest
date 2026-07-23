#!/usr/bin/env python3
"""Generator for benchmark4.ipynb — DA3 + RynnBrain iterative trajectory benchmark.

Local generator kept in the benchmark1-3 style (see project memory
perceptual_pipeline_scaffold.md: "Notebook generator ... regenerate/keep if needed").
Run: python3 build_nb4.py   -> writes benchmark4.ipynb in this directory.

Do NOT run this on a machine without re-reading it first if you change cell content by
hand elsewhere -- this script is the source of truth for benchmark4.ipynb's structure.
"""
import json

CELLS = []


def md(text):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text):
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────────────────────────────────────
md(r"""# SO-101 Perception Benchmark 4 — calibrated DA3 point-cloud + RynnBrain multi-view triangulation

**v2 — rewritten after a real Colab run exposed 3 concrete failures. See "What changed" below.**

Single frame, two heavy models that cannot share a Python process (conflicting pip pins), so this
notebook runs in **two stages, each its own Colab kernel**:

- **Stage A** (§1A install → Restart → §0,§2,§3A,§4A): **Depth Anything 3** (`DA3METRIC-LARGE`)
  predicts depth on the ORIGINAL RGB → **calibrated against robot self-knowledge** (floor grid +
  base origin + the known gripper pixel, robust inverse-depth affine fit — DA3's raw output on
  these clean renders is ~2.7x too far, the same "reads the scene as smooth floor" failure mode
  bench1 already measured) → builds a point cloud from the CALIBRATED depth → renders 3
  purpose-framed viewpoints with **Open3D** (workspace-centred + elevation-lifted, not a blind
  azimuth spin) → saves RGB + depth viz + cloud (`.ply`) + 3 renders + each view's exact camera
  pose to `viz/`, then frees DA3 from memory.
- **Stage B** (§1B install → Restart → §0,§2,§3B,§4B): **RynnBrain** (VLM) predicts a
  gripper→target trajectory **independently on each of the 4 views** (original + 3 renders,
  told that view's own projected gripper pixel as the start — never a "refine the previous
  answer" turn). The 4 views' 2D endpoints are **triangulated** into one 3D grasp point
  (`camera_math.triangulate_nrays`) — this is what makes 2D pixel picks into a metric 3D
  estimate, since a single view is fundamentally one ray, not a point. DA3 is never imported in
  this kernel; it only reads the artifacts Stage A left on disk.
- **§5/§6/§7** run in the same kernel as Stage B (no third restart): evaluate against held-out
  ground truth, write a Markdown report, push.

## What changed from v1 (the first real run, logs/run4_20260723_132657.md, was a clear negative)
1. **Depth calibration added (§4A).** v1 fed DA3's raw `focal*net/300` straight into the cloud;
   it read the cube at 2.59m vs a true 0.96m (1639mm 3D error). Now calibrated with a robust
   inverse-depth affine fit against 3 honest anchor families: a floor grid, the robot base
   origin, and the known gripper pixel — no ground truth, same "up-to-affine monocular depth"
   fact bench1-3 already exploit, just done properly this time.
2. **Render framing fixed (§4A).** v1 spun ±90°/±45° azimuth around the scene origin; two of
   three renders came back near-empty (arm+cube swung to the frame edge — confirmed by eye on
   the pushed PNGs). Now uses workspace-centred lookat + elevation lift (empirically tuned to
   keep gripper/cube/base in frame), which also gives real vertical baseline for triangulation
   (a pure azimuth spin barely moves the camera in z).
3. **Triangulate instead of "refine" (§4B) — the core redesign.** v1 asked RynnBrain to *edit*
   its own previous 2D answer while looking at a second image; the model just echoed the same
   points back (see v1's raw outputs — refine 1/2/3 are near-identical), making the endpoint
   *worse* (20.8→26.1px). A 2D trajectory is a ray, not a 3D point, and no amount of "please
   reconsider" turns one ray into geometry. Now each view gets an **independent** fresh
   prediction (never shown a previous answer), and the 4 endpoints are triangulated. The known
   gripper pixel, projected into each view, both anchors that view's start point AND gives a
   free self-consistency check: triangulating what RynnBrain *actually drew* as each view's
   start and comparing to the known `GRIPPER_TCP_XYZ` measures the whole pipeline's geometric
   accuracy independent of the target itself.
4. **3D eval now comes from triangulation, not a depth lookup (§5).** v1's only 3D number was
   "look up DA3's raw depth at the predicted pixel" — a single, uncalibrated read. Now it's
   `endpoint_err_3d_mm` from `triangulate_nrays` over 4 views; the depth model's job shrinks to
   what it's actually good for (an honest point cloud for framing renders), not carrying the
   whole 3D estimate alone.

## Models (pin exactly, see the assumptions boxes in §3A/§3B for how these were chosen)
- Depth: `depth-anything/DA3METRIC-LARGE` (ByteDance-Seed, Apache-2.0, not gated).
- Trajectory VLM: `Alibaba-DAMO-Academy/RynnBrain1.1-2B` (Alibaba-DAMO-Academy, Apache-2.0).
  **Not gated as far as the model card shows, but §3B still gates on an optional `HF_TOKEN`
  Colab secret and gives an actionable error if loading 401s/403s — same pattern as benchmark3's
  SAM 3 gate.** `RynnBrain-Plan-2B` **does not exist** (confirmed against the live HF listing —
  the "Plan" post-trained line only ships at 8B/30B-A3B); §3B explains the substitution.

## Honesty discipline (same as bench1-3)
`GT_UV` / `GT_XYZ` / `GT_DEPTH_M` are loaded in §2 for the assert self-check and are used **only**
in §5 (Evaluate). They are never passed into a DA3 or RynnBrain call. The gripper start point is
**measured FK of the parked home pose**, not cube ground truth — see the provenance comment in §2.
Depth calibration anchors (floor grid, base origin, gripper pixel) are all robot/rig facts, never
the target object's own ground truth.

## Run order (follow literally — do not "Run all" across a restart)
1. Run §0 (2 cells), then §1A (1 cell). **Runtime → Restart runtime.**
2. Run §0 (2 cells) → §2 → §3A → §4A (Stage A). This writes everything Stage B needs to `viz/`,
   then frees DA3. **Do not run §1B yet.**
3. Run §1B (1 cell). **Runtime → Restart runtime.**
4. Run §0 (2 cells) → §2 → §3B → §4B (Stage B) → §5 → §6 → §7 (Push), all in this same kernel.
""")

# ─────────────────────────────────────────────────────────────────────────────
# 0. Setup + logger
# ─────────────────────────────────────────────────────────────────────────────
md("## 0. Setup + logger\n\nRe-run this section (both cells) after EVERY restart — kernel state does not survive a restart, only files on disk do.")

code(r"""import os, sys
CLONE_URL = 'https://github.com/Yunsmn/RoboticsPerceptionTest.git'
if not os.path.exists('camera_math.py'):
    os.system('git clone ' + CLONE_URL)
    os.chdir('RoboticsPerceptionTest')
sys.path.insert(0, '.')
import glob
import numpy as np, json
from pathlib import Path
from PIL import Image, ImageDraw
import camera_math as CM
print('setup OK')
""")

code(r"""# Tees every subsequent cell (source + stdout/stderr + tracebacks) to run_log.md.
# NOTE vs bench1-3: this notebook restarts the kernel TWICE (Stage A, Stage B), and each restart
# wipes Python state -- but NOT the file on disk. Opening in 'a' (append), not bench1-3's 'w',
# lets one run_log.md accumulate Stage A's install/depth/render log AND Stage B's model/eval log,
# so the single pushed logs/run4_<ts>.md (see 7.) has the whole run, not just the last stage.
import sys, io, datetime, traceback, subprocess
from IPython import get_ipython
_LOG = 'run_log.md'; _f = open(_LOG, 'a')
def _w(s=''):
    _f.write(str(s) + '\n'); _f.flush()
_w(); _w('# Run log (benchmark4) — ' + datetime.datetime.now().isoformat(timespec='seconds'))
try:
    _gpu = subprocess.run(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],
                          capture_output=True, text=True).stdout.strip()
except Exception:
    _gpu = ''
_w('- GPU: ' + (_gpu or 'none / CPU')); _w('- Python: ' + sys.version.split()[0])
class _Tee:
    _is_tee = True
    def __init__(self, real): self._real = real
    def write(self, s):
        n = self._real.write(s)
        try: _f.write(s); _f.flush()
        except Exception: pass
        return n
    def flush(self):
        try: self._real.flush()
        except Exception: pass
    def __getattr__(self, k):
        return getattr(self.__dict__['_real'], k)
if not getattr(sys.stdout, '_is_tee', False): sys.stdout = _Tee(sys.stdout)
if not getattr(sys.stderr, '_is_tee', False): sys.stderr = _Tee(sys.stderr)
_ip = get_ipython(); _n = {'i': 0}
def _pre(info):
    _n['i'] += 1
    _w(); _w('## Cell ' + str(_n['i'])); _w('```python')
    _w((info.raw_cell or '').rstrip()); _w('```'); _w('**output:**'); _w('```text')
def _post(res):
    _w('```')
    err = getattr(res,'error_in_exec',None) or getattr(res,'error_before_exec',None)
    if err is not None:
        _w('**ERROR:**'); _w('```text')
        _w(''.join(traceback.format_exception(type(err), err, err.__traceback__))); _w('```')
if not globals().get('_LOGGER_ON'):
    _ip.events.register('pre_run_cell', _pre); _ip.events.register('post_run_cell', _post); _LOGGER_ON = True
print('run logger active -> run_log.md (append mode -- spans both restarts)')
""")

# ─────────────────────────────────────────────────────────────────────────────
# 1A. Install (Stage A) -> Restart
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 1A. Install — Stage A (Depth Anything 3 + Open3D) — then Runtime → Restart

DA3 and RynnBrain (installed in §1B) are never imported in the same kernel — this is the same
"one model per Colab session, restart between" rule bench1 used for competing depth models,
applied once per stage here because the two pip stacks (DA3's torch/xformers pins vs RynnBrain's
`transformers==5.2.0`) conflict.

**After this cell: Runtime → Restart runtime. Then run §0 (2 cells) → §2 → §3A → §4A only. Do
NOT run §1B in this pass — that happens in a later, separate kernel (step 3 in the run order
above).**""")

code(r"""# Stage A deps: Depth Anything 3 (metric depth) + Open3D (point cloud + rendering).
# RESTART after this cell (torch/numpy ABI, same reason bench1-3 restart after installing
# Depth Pro). Do NOT also run the §1B cell in this kernel.
get_ipython().system('pip -q install "torch>=2" torchvision xformers')
get_ipython().system('pip -q install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git')
get_ipython().system('pip -q install open3d')
""")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 2. Data — one frame + calibration + honest gripper origin

Model-free. Re-run after EVERY restart (both Stage A and Stage B need it). Uses `data/manifest.json`
(the single-object dataset — same calibration `camera_math.side_cam_params()` already matches, see
the self-check below) and `frames[0]` (`pose_000.png`): the cube sits well clear of the parked
gripper (~100px in-frame separation) so the trajectory task is non-trivial.""")

code(r"""FRAME_IDX = 0   # data/manifest.json frames[0] = pose_000.png
man = json.loads(Path('data/manifest.json').read_text()); cam = man['camera']
f, cx, cy = cam['f'], cam['cx'], cam['cy']
cam_pos = np.array(cam['cam_pos'], float); R_cw = np.array(cam['R_cw'], float)
W, H = cam['width'], cam['height']
FR = man['frames'][FRAME_IDX]
IMG_PATH = 'data/' + FR['png']
TARGET_LABEL = 'red cube'                    # a grounding PROMPT, not a label (same convention as bench3)
DEPTH_MODEL_ID = 'depth-anything/DA3METRIC-LARGE'

# Held out for §5 ONLY -- never passed to DA3 or RynnBrain.
GT_XYZ = np.array(FR['gt_xyz'], float)
GT_UV = np.array(FR['gt_uv'], float)
GT_DEPTH_M = FR['gt_depth_m']
print('frame:', IMG_PATH, '| f=%.1f cx=%.1f cy=%.1f | %dx%d' % (f, cx, cy, W, H))

def project(P):
    xc, yc, zc = R_cw.T @ (np.asarray(P, float) - cam_pos)
    return cx + f * xc / -zc, cy - f * yc / -zc, -zc

# --- Honest gripper origin: measured FK of the PARKED home pose all data/ frames were rendered
# at -- NOT cube ground truth, never derived from GT_XYZ. Provenance: project memory
# perceptual_pipeline_scaffold.md (SO-101 controller.py home pose, forward kinematics).
# TCP = tool-centre-point (wrist mount). grasp = midpoint between the open fingers.
GRIPPER_TCP_XYZ = np.array([0.2321, -0.0002, 0.0750])    # base frame, m
GRIPPER_TCP_UV = (415.4, 278.1)                            # side-cam pixel
GRIPPER_TCP_DEPTH_M = 0.995                                # side-cam forward-axis depth, m
GRASP_POINT_XYZ = np.array([0.2322, 0.0022, 0.1353])      # base frame, m (between fingers)
GRASP_POINT_UV = (419.8, 249.2)
GRASP_POINT_DEPTH_M = 0.962
# The trajectory START (§4B) uses GRIPPER_TCP_* per the task spec; GRASP_POINT_* kept for reference.

_chk_tcp = project(GRIPPER_TCP_XYZ); _chk_grasp = project(GRASP_POINT_XYZ)
print('self-check TCP   proj=(%.1f,%.1f,%.3f) vs stated (%.1f,%.1f,%.3f)'
      % (_chk_tcp[0], _chk_tcp[1], _chk_tcp[2], GRIPPER_TCP_UV[0], GRIPPER_TCP_UV[1], GRIPPER_TCP_DEPTH_M))
print('self-check GRASP proj=(%.1f,%.1f,%.3f) vs stated (%.1f,%.1f,%.3f)'
      % (_chk_grasp[0], _chk_grasp[1], _chk_grasp[2], GRASP_POINT_UV[0], GRASP_POINT_UV[1], GRASP_POINT_DEPTH_M))
assert abs(_chk_tcp[0] - GRIPPER_TCP_UV[0]) < 1.0 and abs(_chk_tcp[1] - GRIPPER_TCP_UV[1]) < 1.0, \
    'gripper TCP constant does not match the camera calibration -- check provenance before trusting it'

os.makedirs('viz', exist_ok=True)
img0 = np.array(Image.open(IMG_PATH).convert('RGB'))
Image.fromarray(img0).save('viz/bench4_00_original_rgb.png')
print('saved viz/bench4_00_original_rgb.png', img0.shape)
""")

# ─────────────────────────────────────────────────────────────────────────────
# 3A. Depth adapter
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 3A. Depth adapter — Depth Anything 3 (`DA3METRIC-LARGE`)

Behind a 2-method interface (`load()` + `infer_metric_depth(rgb, focal_px)`) so another
metric-depth model (Depth Pro, UniDepthV2, ...) can be swapped in by editing only this cell —
nothing in §4A references DA3 by name.""")

code(r"""import torch, gc

class DepthAdapter:
    '''Metric depth adapter. Swap models by writing a class with these two methods and
    pointing DEPTH at it -- section 4A only calls .load()/.infer_metric_depth()/.free().'''
    MODEL_ID = DEPTH_MODEL_ID

    def __init__(self):
        self._model = None

    def load(self):
        from depth_anything_3.api import DepthAnything3
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('loading', self.MODEL_ID, 'on', device, '...')
        self._model = DepthAnything3.from_pretrained(self.MODEL_ID).to(device=device)
        return self

    def infer_metric_depth(self, rgb, focal_px):
        # DA3METRIC-LARGE outputs a CANONICAL (non-metric) depth; the model card's FAQ gives the
        # exact conversion: metric_depth[m] = focal_px * net_output / 300. We pass OUR OWN known
        # calibration focal (data/manifest.json), not DA3's self-estimated single-view intrinsics
        # -- the same convention this repo's Depth Pro adapter uses (bench1-3: f_px is always an
        # argument, never trusted from the model).
        pred = self._model.inference([rgb])
        raw = np.asarray(pred.depth[0], dtype=np.float32)
        metric = (float(focal_px) * raw / 300.0).astype(np.float32)
        # DA3 returns depth at the model's OWN internal (patch-aligned) resolution, not the
        # input's -- e.g. 378x504 for a 480x640 frame. The intrinsics (focal_px, cx, cy) and every
        # downstream deprojection assume the INPUT resolution, so resize back to it here. Keeping
        # this in the adapter means any swapped-in depth model also honours the "(H,W) of the
        # input" contract, instead of the point-cloud cell silently broadcasting a mismatched grid.
        Hh, Ww = rgb.shape[:2]
        if metric.shape[:2] != (Hh, Ww):
            import cv2
            metric = cv2.resize(metric, (Ww, Hh), interpolation=cv2.INTER_LINEAR)
        return metric

    def free(self):
        del self._model; self._model = None
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

DEPTH = DepthAdapter().load()
print('DA3 ready.')
""")

# ─────────────────────────────────────────────────────────────────────────────
# 4A. Stage A execution
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 4A. Stage A — calibrated depth, point cloud, 3 framed viewpoints (writes `viz/`, then frees DA3)

Only `img0` (the original RGB) and this frame's own DA3 depth go into the cloud — no ground
truth anywhere. The 3 renders (+ their exact camera poses) are saved to disk so Stage B (a
separate kernel, after §1B + restart) can read them without ever importing DA3 or torch/xformers
built for it.

**Depth calibration (new in v2):** DA3's raw `focal*net/300` output measured **2.59m at the cube
vs a true 0.96m** on the first real run — the model reads this clean synthetic scene as a smooth
floor (the same OOD failure bench1 already found for Depth Pro/UniDepth on this dataset). Rather
than trust that number, we fit a robust affine in **inverse depth** (`1/d_true = a*(1/d_raw)+b` —
monocular nets are affine in inverse depth, not depth itself) against three honest,
object-independent anchor families: a **floor grid** (z=0, `project()` from §2 gives the exact
forward depth analytically), the **robot base origin** (0,0,0), and the **known gripper pixel**
(`GRIPPER_TCP_UV` / `GRIPPER_TCP_DEPTH_M` — the same FK constant used as the trajectory start).
The fit is robust (drop the worst 20% residual, refit, 3 rounds) because some floor-grid pixels
are occluded by the arm itself and would otherwise poison the fit.""")

code(r"""def calibrate_depth_inverse_affine(depth_raw, anchors, robust_iters=3, drop_frac=0.2,
                                    seed=0, max_pairs=20000):
    '''Fit depth_true^-1 = a*depth_raw^-1 + b from (u,v,d_true) anchors sampled in depth_raw,
    robust to occlusion outliers (e.g. floor pixels blocked by the arm). Monocular metric depth
    nets are approximately affine in INVERSE depth -- the same "up-to-affine" fact bench1-3
    already exploit with a single floor/base anchor, generalised here to a robust multi-anchor fit.

    SEEDED WITH THEIL-SEN, not plain least squares: a local synthetic test (14pct contaminated
    anchors, occlusion-sized errors) showed that seeding the very first round with ordinary
    least squares on the FULL contaminated set can drag the fit so far off that "drop the worst
    20pct residual" starts discarding GOOD points instead of bad ones -- OLS is not
    breakdown-robust to a handful of high-leverage outliers. The median of ALL pairwise slopes
    (Theil-Sen) has a ~29pct breakdown point and needs no threshold, so it seeds a trustworthy
    starting line; the iterative trim-and-refit (3 rounds, worst 20pct dropped each round) then
    polishes it. Verified locally to recover the injected (a,b) exactly at 14pct AND 25pct
    contamination; plain-OLS seeding failed even at 14pct (recovered a=0.21 vs true a=0.42).
    MODEL-AGNOSTIC: takes a plain depth array + anchor list, not the adapter -- swapping
    DepthAdapter for another model still gets calibrated by this same function.
    Returns (a, b, n_used, n_total).'''
    uu = np.array([p[0] for p in anchors], float); vv = np.array([p[1] for p in anchors], float)
    d_true = np.array([p[2] for p in anchors], float)
    ui = np.clip(np.round(uu).astype(int), 0, depth_raw.shape[1] - 1)
    vi = np.clip(np.round(vv).astype(int), 0, depth_raw.shape[0] - 1)
    d_raw = depth_raw[vi, ui].astype(float)
    keep = np.isfinite(d_raw) & (d_raw > 1e-6) & np.isfinite(d_true) & (d_true > 1e-6)
    inv_raw = 1.0 / d_raw[keep]; inv_true = 1.0 / d_true[keep]
    n_total = len(inv_raw)
    idx = np.arange(n_total)
    if n_total < 3:
        return 1.0, 0.0, 0, n_total

    rng = np.random.default_rng(seed)
    ii, jj = np.triu_indices(n_total, k=1)
    if len(ii) > max_pairs:
        sel = rng.choice(len(ii), size=max_pairs, replace=False)
        ii, jj = ii[sel], jj[sel]
    dx = inv_raw[ii] - inv_raw[jj]
    ok = np.abs(dx) > 1e-9
    slopes = (inv_true[ii[ok]] - inv_true[jj[ok]]) / dx[ok]
    a = float(np.median(slopes))
    b = float(np.median(inv_true - a * inv_raw))

    for _ in range(robust_iters):
        resid = np.abs(a * inv_raw[idx] + b - inv_true[idx])
        n_drop = int(round(len(idx) * drop_frac))
        if n_drop == 0 or len(idx) - n_drop < 3:
            break
        order = np.argsort(resid)
        idx = idx[order[:len(idx) - n_drop]]
        A = np.vstack([inv_raw[idx], np.ones(len(idx))]).T      # OLS refit is fine once seeded
        a, b = np.linalg.lstsq(A, inv_true[idx], rcond=None)[0]  # robustly and outliers are gone
    return float(a), float(b), int(len(idx)), int(n_total)

# --- anchors: all robot/rig facts, never the target object's own ground truth ---
FLOOR_ANCHORS = []
for xw in np.arange(-0.20, 0.55 + 1e-9, 0.04):
    for yw in np.arange(-0.40, 0.42 + 1e-9, 0.04):
        u_a, v_a, d_a = project([float(xw), float(yw), 0.0])
        if d_a > 0.05 and 0 <= u_a < W and 0 <= v_a < H:
            FLOOR_ANCHORS.append((u_a, v_a, d_a))
u_base, v_base, d_base = project([0.0, 0.0, 0.0])
BASE_ANCHOR = [(u_base, v_base, d_base)] if (d_base > 0.05 and 0 <= u_base < W and 0 <= v_base < H) else []
GRIPPER_ANCHOR = [(GRIPPER_TCP_UV[0], GRIPPER_TCP_UV[1], GRIPPER_TCP_DEPTH_M)]
DEPTH_CALIB_ANCHORS = FLOOR_ANCHORS + BASE_ANCHOR + GRIPPER_ANCHOR
print('depth calibration anchors: %d floor + %d base + %d gripper = %d total'
      % (len(FLOOR_ANCHORS), len(BASE_ANCHOR), len(GRIPPER_ANCHOR), len(DEPTH_CALIB_ANCHORS)))

depth_raw = DEPTH.infer_metric_depth(img0, f)
DEPTH_CALIB_A, DEPTH_CALIB_B, DEPTH_CALIB_N_USED, DEPTH_CALIB_N_TOTAL = calibrate_depth_inverse_affine(
    depth_raw, DEPTH_CALIB_ANCHORS)
print('calibration fit: a=%.6f b=%.6f (used %d/%d anchors after robust rejection)'
      % (DEPTH_CALIB_A, DEPTH_CALIB_B, DEPTH_CALIB_N_USED, DEPTH_CALIB_N_TOTAL))
depth_m = 1.0 / np.clip(DEPTH_CALIB_A / np.clip(depth_raw, 1e-6, None) + DEPTH_CALIB_B, 1e-6, None)

_gu, _gv = int(round(GRIPPER_TCP_UV[0])), int(round(GRIPPER_TCP_UV[1]))
print('gripper-pixel check: raw=%.3fm -> calibrated=%.3fm  (true=%.3fm)'
      % (float(depth_raw[_gv, _gu]), float(depth_m[_gv, _gu]), GRIPPER_TCP_DEPTH_M))
print('depth range after calibration: %.3f - %.3f m' % (float(depth_m.min()), float(depth_m.max())))
np.save('viz/bench4_depth_m.npy', depth_m)
with open('viz/bench4_depth_calib.json', 'w') as _fjs:
    json.dump({'a': DEPTH_CALIB_A, 'b': DEPTH_CALIB_B, 'n_anchors_used': DEPTH_CALIB_N_USED,
               'n_anchors_total': DEPTH_CALIB_N_TOTAL,
               'gripper_check_raw_m': float(depth_raw[_gv, _gu]),
               'gripper_check_calibrated_m': float(depth_m[_gv, _gu]),
               'gripper_check_true_m': GRIPPER_TCP_DEPTH_M}, _fjs, indent=2)

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(depth_m, cmap='turbo'); ax.set_title('DA3 depth, calibrated (m)'); ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)
plt.savefig('viz/bench4_01_depth_da3.png', dpi=110, bbox_inches='tight'); plt.close()
print('saved viz/bench4_01_depth_da3.png, viz/bench4_depth_m.npy, viz/bench4_depth_calib.json')
""")

code(r"""# Open3D CPU/software-rendering fallback flags -- MUST be set before the first `import open3d`
# (setting them after import is a documented no-op). EGL_PLATFORM=surfaceless is correct for
# Ubuntu 20.04+/Mesa>=20.2 (current Colab images); OPEN3D_CPU_RENDERING=true is the older-Mesa
# fallback. Harmless to set both when a real GPU/EGL context IS available (Colab T4 normally has
# one) -- Open3D only reads them if it needs software rendering.
os.environ.setdefault('EGL_PLATFORM', 'surfaceless')
os.environ.setdefault('OPEN3D_CPU_RENDERING', 'true')
import open3d as o3d

# --- honest metric point cloud from ONE view: back-project every (downsampled) pixel through
# DA3's own depth using the SAME camera_math geometry the rest of the repo trusts (this is the
# vectorised form of camera_math.point_at_depth -- no plane, no object prior, no ground truth).
STRIDE = 2
vv, uu = np.mgrid[0:H:STRIDE, 0:W:STRIDE]
uu = uu.astype(float); vv = vv.astype(float)
d_local = np.stack([(uu - cx) / f, -(vv - cy) / f, -np.ones_like(uu)], axis=-1)  # camera-frame rays
d_world = d_local @ R_cw.T
d_unit = d_world / np.linalg.norm(d_world, axis=-1, keepdims=True)
fwd = -R_cw[:, 2]
ddot = d_unit @ fwd
depth_ds = depth_m[0:H:STRIDE, 0:W:STRIDE]
t = depth_ds / np.clip(ddot, 1e-6, None)
pts = cam_pos[None, None, :] + t[..., None] * d_unit
valid = np.isfinite(pts).all(axis=-1) & (depth_ds > 0.05) & (depth_ds < 5.0)
pts_flat = pts[valid].reshape(-1, 3)
cols_flat = (img0[0:H:STRIDE, 0:W:STRIDE][valid].reshape(-1, 3).astype(np.float64) / 255.0)
print('point cloud:', pts_flat.shape[0], 'points')

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts_flat)
pcd.colors = o3d.utility.Vector3dVector(cols_flat)
o3d.io.write_point_cloud('viz/bench4_cloud.ply', pcd)
print('saved viz/bench4_cloud.ply')
""")

code(r"""# --- 3 purpose-framed re-rendered viewpoints of the SAME monocular (now calibrated) cloud (see
# the honest caveat in §6). v1 orbited the SCENE ORIGIN by +-45/90deg azimuth around the side
# camera's own look-at point and threw 2 of 3 renders nearly empty (arm+cube swung to the frame
# edge -- confirmed by eye on the pushed PNGs). The fix is a TRANSLATION, not just a rotation:
# aim at the WORKSPACE centre (raised to arm height, not the floor-level scene origin) and lift
# the elevation well below the side camera's own -35deg. The elevation lift matters as much as
# the translation -- a pure azimuth spin barely moves the camera in z, which starves triangulation
# of vertical baseline (ill-conditioned in z even with 4 views). These (az,el) pairs are centred
# on the side camera's own az=135 +-25deg, at el -50/-55 (empirically tuned so gripper + cube +
# base + container all stay in frame -- see §6 for the actual renders).
WORK_LOOKAT = np.array([0.13, -0.07, 0.08])   # workspace centre, raised to include arm height
VIEW_DIST = 0.70
VIEWS = [
    ('A_az110_el50', 110.0, -50.0),
    ('B_az135_el55', 135.0, -55.0),
    ('C_az160_el50', 160.0, -50.0),
]

def _cv_extrinsic(cam_pos_v, R_cw_v):
    # world->camera, OpenCV/Open3D convention (+X right, +Y down, +Z forward). camera_math's
    # R_cw is MuJoCo convention (+X right, +Y up, -Z forward) -- flip Y,Z to match. Verified by
    # hand against camera_math.project()'s own pixel formula (see build_nb4.py commit notes).
    flip = np.diag([1.0, -1.0, -1.0])
    Rwc = flip @ R_cw_v.T
    twc = -Rwc @ cam_pos_v
    Tcv = np.eye(4); Tcv[:3, :3] = Rwc; Tcv[:3, 3] = twc
    return Tcv

def _fallback_render(view_cam_pos, view_R_cw, out_path, title):
    # Robust Plan-B if Open3D's offscreen GL context errors on this particular Colab runtime:
    # splat the SAME points with our own pinhole model + painter's-algorithm depth sort. Zero
    # Open3D dependency, so it cannot fail for the same reason as the primary path.
    def proj(P):
        xc, yc, zc = view_R_cw.T @ (P - view_cam_pos)
        return cx + f * xc / -zc, cy - f * yc / -zc, -zc
    xc_, yc_, zc_ = proj(pts_flat.T)
    keep = zc_ > 0.05
    order = np.argsort(-zc_[keep])          # far first, near last (painter's algorithm)
    fig, ax = plt.subplots(figsize=(W / 100, H / 100), dpi=100)
    ax.set_facecolor('white')
    ax.scatter(xc_[keep][order], yc_[keep][order], c=cols_flat[keep][order], s=2, marker='.')
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off'); ax.set_title(title)
    plt.savefig(out_path, dpi=100, bbox_inches='tight'); plt.close()

# Clean up any renders from a PREVIOUS notebook version/run before writing this run's set. v1
# named these 'oblique45'/'left'/'right'; v2 uses 'A_az110_el50' etc -- different filenames, so
# without this cleanup the old ones just sit in viz/ alongside the new ones (they were pushed to
# the repo by the earlier run) and Stage B's glob('viz/bench4_02_render_*.png') would pick up
# BOTH sets, including stale ones with no matching entry in this run's VIEW_POSES. Also drop v1's
# dead trajectory filenames (bench4_03_traj_initial.png / bench4_04_traj_refine_*.png) for hygiene
# -- v2 writes bench4_05_traj_<name>.png instead.
for _stale in glob.glob('viz/bench4_02_render_*.png') + glob.glob('viz/bench4_03_traj_*.png') + \
              glob.glob('viz/bench4_04_traj_*.png'):
    os.remove(_stale)

RENDER_PATHS = []
VIEW_POSES = {}    # render_path -> {f,cx,cy,cam_pos,R_cw,az,el,dist,lookat} -- Stage B (a
                    # SEPARATE kernel after restart) needs the exact pose each render was taken
                    # from to project the gripper into it and to build triangulation rays; Python
                    # variables don't survive the restart, so this is persisted to disk.
for name, az, el in VIEWS:
    vf, vcx, vcy, vcam_pos, vR_cw = CM.free_cam_params(az, el, VIEW_DIST, WORK_LOOKAT, W=W, H=H)
    out_path = 'viz/bench4_02_render_%s.png' % name
    ok = False
    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(W, H)
        renderer.scene.set_background([1, 1, 1, 1])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'      # flat/textureless -- matches the point colours as-is
        mat.point_size = 3.0
        renderer.scene.add_geometry('cloud', pcd, mat)
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, vf, vf, vcx, vcy)
        extrinsic = _cv_extrinsic(vcam_pos, vR_cw)
        renderer.setup_camera(intrinsic, extrinsic)
        img_o3d = renderer.render_to_image()
        o3d.io.write_image(out_path, img_o3d)
        del renderer
        ok = True
    except Exception as e:
        print('  Open3D OffscreenRenderer failed for %s (%s) -> matplotlib point-splat fallback' % (name, e))
    if not ok:
        _fallback_render(vcam_pos, vR_cw, out_path, name)
    RENDER_PATHS.append(out_path)
    VIEW_POSES[out_path] = {'f': float(vf), 'cx': float(vcx), 'cy': float(vcy),
                             'cam_pos': vcam_pos.tolist(), 'R_cw': vR_cw.tolist(),
                             'az': az, 'el': el, 'dist': VIEW_DIST, 'lookat': WORK_LOOKAT.tolist()}
    print('saved', out_path, '(open3d)' if ok else '(fallback splat)')

with open('viz/bench4_view_poses.json', 'w') as _fjs:
    json.dump(VIEW_POSES, _fjs, indent=2)
print('saved viz/bench4_view_poses.json (%d view poses)' % len(VIEW_POSES))
""")

code(r"""DEPTH.free()
print('DA3 freed from memory. Stage A complete -- proceed to 1B (install RynnBrain) + Restart.')
""")

# ─────────────────────────────────────────────────────────────────────────────
# 1B. Install (Stage B) -> Restart
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 1B. Install — Stage B (RynnBrain) — then Runtime → Restart

**After this cell: Runtime → Restart runtime. Then run §0 (2 cells) → §2 → §3B → §4B → §5 → §6 →
§7, all in this same kernel (no third restart).**""")

code(r"""# Stage B deps: RynnBrain (transformers-native VLM). Version pin per the model's own README.
# RESTART after this cell. Do NOT run this in the same kernel as §1A/§3A/§4A -- DA3 and RynnBrain
# must never be imported in the same process (conflicting torch/transformers pins).
get_ipython().system('pip -q install -U "transformers==5.2.0" accelerate')
""")

# ─────────────────────────────────────────────────────────────────────────────
# 3B. RynnBrain adapter
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 3B. RynnBrain adapter — per-view trajectory prediction

**Model choice (verified against the live HF listing, not guessed):** the task asked for the 2B
*manipulation-planning* checkpoint ("RynnBrain-Plan"). **`RynnBrain-Plan-2B` does not exist** —
the "Plan" post-trained line ships only at 8B/30B-A3B (`Alibaba-DAMO-Academy/RynnBrain-Plan-8B`,
confirmed from its own model card's "Model Zoo" table). An 8B checkpoint in bf16 is too tight for
a free T4's 16GB (weights alone ≈16GB before activations/vision tower), so this adapter defaults
to **`Alibaba-DAMO-Academy/RynnBrain1.1-2B`** — the current (2026-07-16) 2B *base* model, which
ships its own "Trajectory Location" and "Contact Point Prediction" cookbooks (i.e. exactly
low-level pixel-trajectory / grasp-point prediction), arguably a closer fit for THIS experiment
than the Plan checkpoint's high-level multi-step task-decomposition training anyway. Flip
`RYNNBRAIN_MODEL_ID` below to `RynnBrain-Plan-8B` to test the literal "Plan" checkpoint on a
bigger GPU (A100/L4) — verify VRAM before doing so.

**Prompt format is copied verbatim from RynnBrain's own cookbook**
(`cookbooks/6_trajectory_location.ipynb`, single-image cell): `<trajectory> (x1,y1), (x2,y2), ...
</trajectory>`, coordinates normalised to **[0,1000]** over the image's own width/height. That
cookbook example is free-form (the model decides its own start point, single image, no history).

**v2 redesign (§4B): independent per-view prediction + triangulation, not "refine".** v1 fed the
model its own previous `<trajectory>` answer plus a second image and asked it to "refine" — the
real run showed it just echoing the same points back (see the v1 raw outputs preserved in
`logs/run4_20260723_132657.md`), which made the endpoint worse. A 2D trajectory from one image is
a ray, not a 3D point; asking a VLM to introspect and edit its own prior answer is not what turns
one ray into geometry — triangulating **independent** predictions from views with known,
different poses is. So §4B now calls `RYNN.predict()` **once per view, fresh, with no history**
(the model never sees a previous trajectory), and 3D geometry is recovered afterward by
`camera_math.triangulate_nrays` over the 4 views' endpoints.

**Assumptions this adapter makes beyond what the cookbook demonstrates — verify on the first
Colab run, not asserted as fact:**
1. Telling the model the start point in the instruction text ("its current position in THIS
   image is pixel (x,y)... the FIRST point of your trajectory must be...") makes it anchor there,
   in EVERY view (not just the original). If it doesn't, §4B numerically re-anchors the drawn
   trajectory at that view's own projected gripper pixel (logged, not silent) for the path/viz —
   but keeps the model's own UNMODIFIED first point separately for the gripper-anchor
   triangulation check (§4B/§5), since that check exists specifically to measure whether the
   model actually complies, not to paper over it.
2. Each per-view prediction is assumed independent enough that triangulating across them is
   meaningful — i.e. RynnBrain is not so anchored to one canonical "reading" of the scene that
   all 4 views collapse to the same pixel-ratio answer regardless of the actual rendered content.
   The gripper-anchor residual (§5) is partly a check on this: if the 4 views' drawn start points
   don't triangulate close to the known `GRIPPER_TCP_XYZ`, the per-view geometry (pose and/or
   model behaviour) is suspect and the endpoint triangulation should be read with that in mind.
3. Gating: the model card shows no gated-repo banner, so `load()` tries anonymously first and
   only asks for `HF_TOKEN` if the download actually 401s/403s (same UX as bench3's SAM 3 gate,
   but not a hard requirement up front since we could not confirm gating is actually needed).""")

code(r"""import re, json as _json, torch
from transformers import AutoModelForImageTextToText, AutoProcessor

RYNNBRAIN_MODEL_ID = 'Alibaba-DAMO-Academy/RynnBrain1.1-2B'
# Swappable alternative -- the literal "Plan" checkpoint the task asked for, only at 8B (no 2B
# Plan variant exists). Heavier: verify free-T4 VRAM before switching.
# RYNNBRAIN_MODEL_ID = 'Alibaba-DAMO-Academy/RynnBrain-Plan-8B'

TRAJ_RE = re.compile(r'\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)')

def parse_trajectory(text, w, h):
    '''<trajectory> (x1,y1), (x2,y2), ... </trajectory>, coords normalised to [0,1000] -- the
    exact format demonstrated in RynnBrain's own cookbooks/6_trajectory_location.ipynb. Returns
    a list of (u,v) PIXEL tuples for THIS frame's own (w,h). Falls back to scanning the whole
    response for point-like tuples if the <trajectory> tags themselves are missing.'''
    m = re.search(r'<trajectory>(.*?)</trajectory>', text, re.S)
    body = m.group(1) if m else text
    return [(float(a) / 1000.0 * w, float(b) / 1000.0 * h) for a, b in TRAJ_RE.findall(body)]

class RynnBrainAdapter:
    '''Trajectory VLM adapter: load() + predict(image_paths, prompt)->raw_text. Deliberately
    thin (no parsing here) so section 4B controls prompt construction / start-anchoring explicitly.'''

    def __init__(self, model_id=RYNNBRAIN_MODEL_ID):
        self.model_id = model_id
        self._model = None; self._proc = None; self._device = None

    def load(self):
        try:
            from google.colab import userdata
            hf_token = userdata.get('HF_TOKEN')
        except Exception:
            hf_token = None
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print('WARNING: no GPU visible -- RynnBrain on CPU will be very slow.')
        kwargs = dict(token=hf_token) if hf_token else {}
        try:
            self._proc = AutoProcessor.from_pretrained(self.model_id, **kwargs)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, dtype=torch.bfloat16, **kwargs)
        except OSError as e:
            raise RuntimeError(
                "Could not load %s (%s). If this is a gated-repo 401/403, add an HF_TOKEN Colab "
                "secret (key icon, Notebook access ON) and re-run this cell." % (self.model_id, e)) from e
        try:
            self._model.to(device)
        except torch.cuda.OutOfMemoryError as e:
            raise RuntimeError(
                'OOM loading %s on this GPU. Try Runtime -> Change runtime type -> a bigger GPU, '
                'or keep RYNNBRAIN_MODEL_ID at the 2B default.' % self.model_id) from e
        self._device = device
        return self

    def predict(self, image_paths, prompt, max_new_tokens=256):
        content = [{'type': 'image', 'image': p} for p in image_paths]
        content.append({'type': 'text', 'text': prompt})
        conversation = [{'role': 'user', 'content': content}]
        inputs = self._proc.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt').to(self._device)
        gen = self._model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = gen[:, inputs['input_ids'].shape[1]:]
        return self._proc.batch_decode(gen, skip_special_tokens=True)[0]

    def free(self):
        del self._model; self._model = None
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

RYNN = RynnBrainAdapter().load()
print('RynnBrain ready:', RYNNBRAIN_MODEL_ID)
""")

# ─────────────────────────────────────────────────────────────────────────────
# 4B. Stage B execution
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 4B. Stage B — independent per-view trajectories, then triangulate to 3D

4 views total: the **original** RGB (the true side-cam pose from §2) plus the **3 rendered**
views from §4A. Each gets its own **fresh, independent** `RYNN.predict()` call — no view is ever
told a previous answer, so there is nothing to just echo back. `RENDER_PATHS` and
`VIEW_POSES` are rebuilt/reloaded from disk because Stage A's Python variables did not survive
the restart — only its files did.

Per view: project the known `GRIPPER_TCP_XYZ` through THAT view's own camera pose to get the
start pixel to tell RynnBrain (for the original view this is exactly `GRIPPER_TCP_UV`; for the
renders it's wherever the gripper actually falls in that reprojection). After all 4 predictions:
triangulate the 4 endpoints -> the 3D grasp point, and SEPARATELY triangulate the 4
(unmodified) start points RynnBrain actually drew -> compare to the known gripper position as a
self-consistency check on the whole pipeline's geometry.""")

code(r"""RENDER_PATHS = sorted(glob.glob('viz/bench4_02_render_*.png'))
assert RENDER_PATHS, 'No renders found in viz/ -- run Stage A (1A/2/3A/4A) first, in an earlier kernel.'
with open('viz/bench4_view_poses.json') as _fjs:
    VIEW_POSES = json.load(_fjs)
print('found', len(RENDER_PATHS), 'rendered views:', RENDER_PATHS)

def project_pt(P, f_, cx_, cy_, cam_pos_, R_cw_):
    # same math as §2's project(), parametrised per-view instead of closed over the original
    # side-cam pose -- used to find where the KNOWN gripper falls in each rendered view.
    xc, yc, zc = np.asarray(R_cw_, float).T @ (np.asarray(P, float) - np.asarray(cam_pos_, float))
    return cx_ + f_ * xc / -zc, cy_ - f_ * yc / -zc, -zc

FORMAT_INSTR = ('Return up to 10 key trajectory points as a list of tuples in the format: '
                '<trajectory> (x1, y1), (x2, y2), ... </trajectory>. All coordinates normalised '
                'to the [0, 1000] range, matching this image\'s own width and height.')

def build_traj_prompt(start_u1000, start_v1000, target_label):
    return (
        'You are controlling a robot gripper. Its current position in THIS image is the pixel '
        '(%.0f, %.0f) (normalised 0-1000). Predict a collision-aware manipulation trajectory '
        'that moves the gripper from its current position to a grasp pose on the %s, as seen in '
        'THIS image. The FIRST point of your trajectory must be the gripper\'s given start '
        'position. %s' % (start_u1000, start_v1000, target_label, FORMAT_INSTR)
    )

def draw_traj(img_path, traj, out_path, title):
    im = Image.open(img_path).convert('RGB'); draw = ImageDraw.Draw(im)
    for i in range(len(traj) - 1):
        draw.line([traj[i], traj[i + 1]], fill=(255, 0, 0), width=3)
    for i, (px, py) in enumerate(traj):
        r = 6
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 220, 255) if i else (0, 255, 0))
    im.save(out_path)
    return out_path

# --- the 4 views: the TRUE original camera + the 3 rendered/re-projected ones, each with its own
# exact pose. The original is included as a view like any other (not special-cased) so the
# triangulation treats it uniformly with the renders.
ALL_VIEWS = [{'name': 'original', 'image': IMG_PATH, 'f': f, 'cx': cx, 'cy': cy,
              'cam_pos': cam_pos.tolist(), 'R_cw': R_cw.tolist()}]
for render_path in RENDER_PATHS:
    pose = VIEW_POSES[render_path]
    ALL_VIEWS.append({'name': os.path.splitext(os.path.basename(render_path))[0].replace('bench4_02_render_', ''),
                       'image': render_path, **pose})
print('views for trajectory prediction:', [v['name'] for v in ALL_VIEWS])
""")

code(r"""# --- one FRESH, INDEPENDENT prediction per view -- no view is ever shown a previous answer, so
# there is nothing for the model to just echo back (the v1 failure mode). Each view's own
# GRIPPER_TCP_XYZ projection gives the start pixel to tell it.
PER_VIEW = []
for vi, v in enumerate(ALL_VIEWS):
    cam_pos_v = np.array(v['cam_pos'], float); R_cw_v = np.array(v['R_cw'], float)
    su, sv, sdepth = project_pt(GRIPPER_TCP_XYZ, v['f'], v['cx'], v['cy'], cam_pos_v, R_cw_v)
    given_start_px = (float(su), float(sv))
    prompt = build_traj_prompt(su / W * 1000.0, sv / H * 1000.0, TARGET_LABEL)
    print('--- view %s: given start px (%.1f, %.1f), depth %.3fm ---' % (v['name'], su, sv, sdepth))
    raw = RYNN.predict([v['image']], prompt)
    print(raw)
    traj_px = parse_trajectory(raw, W, H)
    if not traj_px:
        print('  could not parse -- skipping this view (fewer views -> triangulation still works with >=2)')
        PER_VIEW.append({'name': v['name'], 'image': v['image'], 'f': v['f'], 'cx': v['cx'], 'cy': v['cy'],
                          'cam_pos': cam_pos_v, 'R_cw': R_cw_v, 'given_start_px': given_start_px,
                          'raw': raw, 'raw_first_px': None, 'traj_px': [], 'endpoint_px': None})
        continue
    raw_first_px = traj_px[0]                     # UNMODIFIED model output -- used for the
                                                    # gripper-anchor self-consistency check below,
                                                    # never silently replaced.
    _start_err_px = float(np.hypot(raw_first_px[0] - given_start_px[0], raw_first_px[1] - given_start_px[1]))
    print('  model start vs given gripper pixel in this view: %.1f px' % _start_err_px)
    # Numeric safety net for the DRAWN/path-length trajectory only (never for the gripper-anchor
    # check, which deliberately uses the model's raw, unmodified first point instead). Same 20px
    # drift threshold as v1's re-anchoring.
    traj_anchored = [given_start_px] + traj_px if _start_err_px > 20.0 else traj_px
    PER_VIEW.append({'name': v['name'], 'image': v['image'], 'f': v['f'], 'cx': v['cx'], 'cy': v['cy'],
                      'cam_pos': cam_pos_v, 'R_cw': R_cw_v, 'given_start_px': given_start_px,
                      'raw': raw, 'raw_first_px': raw_first_px, 'traj_px': traj_anchored,
                      'endpoint_px': traj_anchored[-1]})
    out_png = 'viz/bench4_05_traj_%s.png' % v['name']
    draw_traj(v['image'], traj_anchored, out_png, v['name'])
    print('  saved', out_png)
""")

code(r"""# --- triangulate: (1) the 4 views' ENDPOINTS -> the 3D grasp target (this is the number §5
# scores against GT_XYZ); (2) the 4 views' RAW (unmodified) START points RynnBrain actually drew
# -> compared against the KNOWN GRIPPER_TCP_XYZ as a self-consistency check on the whole
# pipeline's geometry (pose accuracy + whether the model actually complied with the given start).
def rays_from_views(pixel_key):
    rays, used = [], []
    for v in PER_VIEW:
        px = v.get(pixel_key)
        if px is None:
            continue
        o, d = CM.back_project(px[0], px[1], v['f'], v['cx'], v['cy'], v['cam_pos'], v['R_cw'])
        rays.append((o, d)); used.append(v['name'])
    return rays, used

endpoint_rays, endpoint_views_used = rays_from_views('endpoint_px')
tri_endpoint, tri_endpoint_resid, tri_endpoint_n, tri_endpoint_cond = CM.triangulate_nrays(endpoint_rays)
print('endpoint triangulation: %s views -> %s' % (endpoint_views_used, tri_endpoint))
if tri_endpoint is not None:
    print('  residual %.4fm | n_inliers %d/%d | cond %.3f'
          % (tri_endpoint_resid, tri_endpoint_n, len(endpoint_rays), tri_endpoint_cond))

start_rays, start_views_used = rays_from_views('raw_first_px')
tri_gripper, tri_gripper_resid, tri_gripper_n, tri_gripper_cond = CM.triangulate_nrays(start_rays)
gripper_anchor_residual_mm = None
if tri_gripper is not None:
    gripper_anchor_residual_mm = float(np.linalg.norm(tri_gripper - GRIPPER_TCP_XYZ) * 1000.0)
    print('gripper-anchor triangulation: %s views -> %s | residual %.4fm | n_inliers %d/%d | cond %.3f'
          % (start_views_used, tri_gripper, tri_gripper_resid, tri_gripper_n, len(start_rays), tri_gripper_cond))
    print('  vs KNOWN gripper %s -> pipeline self-consistency error: %.1f mm'
          % (GRIPPER_TCP_XYZ.tolist(), gripper_anchor_residual_mm))
else:
    print('gripper-anchor triangulation FAILED (<2 usable views) -- treat the endpoint triangulation with caution')

# --- best-effort/secondary: resample each view's own 2D path by arc-length to K points and
# triangulate index-by-index -> an approximate 3D path (NOT the primary metric -- endpoint-only
# triangulation above is what §5 scores).
def resample_by_arclength(pts, k):
    pts = np.array(pts, float)
    if len(pts) < 2:
        return np.repeat(pts if len(pts) else np.zeros((1, 2)), k, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-9:
        return np.repeat(pts[:1], k, axis=0)
    out = []
    for t in np.linspace(0.0, total, k):
        j = int(np.clip(np.searchsorted(cum, t) - 1, 0, len(pts) - 2))
        seg_t = (t - cum[j]) / max(seg[j], 1e-9)
        out.append(pts[j] + seg_t * (pts[j + 1] - pts[j]))
    return np.array(out)

K_PATH = 5
TRAJ_3D_PATH = []
_usable_views = [v for v in PER_VIEW if v['traj_px']]
if len(_usable_views) >= 2:
    _resampled = {v['name']: resample_by_arclength(v['traj_px'], K_PATH) for v in _usable_views}
    for k in range(K_PATH):
        rays_k = []
        for v in _usable_views:
            uk, vk = _resampled[v['name']][k]
            o, d = CM.back_project(float(uk), float(vk), v['f'], v['cx'], v['cy'], v['cam_pos'], v['R_cw'])
            rays_k.append((o, d))
        p, resid, n, cond = CM.triangulate_nrays(rays_k)
        TRAJ_3D_PATH.append({'k': k, 'xyz': None if p is None else p.tolist(),
                              'resid_m': resid, 'n': n, 'cond': cond})
    print('resampled 3D path (%d pts, secondary/best-effort):' % K_PATH,
          [r['xyz'] for r in TRAJ_3D_PATH])
else:
    print('fewer than 2 usable views -- skipping the secondary resampled 3D path')

with open('trajectory.json', 'w') as fjs:
    _json.dump({
        'per_view': [{'name': v['name'], 'image': v['image'],
                      'given_start_px': list(v['given_start_px']),
                      'raw_first_px': None if v['raw_first_px'] is None else list(v['raw_first_px']),
                      'traj_px': [list(p) for p in v['traj_px']],
                      'endpoint_px': None if v['endpoint_px'] is None else list(v['endpoint_px']),
                      'raw': v['raw']} for v in PER_VIEW],
        'gripper_xyz_known': GRIPPER_TCP_XYZ.tolist(),
        'triangulated_endpoint_xyz': None if tri_endpoint is None else tri_endpoint.tolist(),
        'triangulated_endpoint_resid_m': tri_endpoint_resid,
        'triangulated_endpoint_n_inliers': tri_endpoint_n,
        'triangulated_endpoint_cond': tri_endpoint_cond,
        'triangulated_endpoint_views_used': endpoint_views_used,
        'gripper_anchor_triangulated_xyz': None if tri_gripper is None else tri_gripper.tolist(),
        'gripper_anchor_resid_m': tri_gripper_resid,
        'gripper_anchor_n_inliers': tri_gripper_n,
        'gripper_anchor_cond': tri_gripper_cond,
        'gripper_anchor_residual_mm': gripper_anchor_residual_mm,
        'gripper_anchor_views_used': start_views_used,
        'trajectory_3d_resampled': TRAJ_3D_PATH,
    }, fjs, indent=2)
print('saved trajectory.json (%d views)' % len(PER_VIEW))
RYNN.free()
print('RynnBrain freed. Stage B trajectory prediction + triangulation complete -- proceed to 5. Evaluate.')
""")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 5. Evaluate — 2D pixel error (original view), 3D error from TRIANGULATION, gripper-anchor check

**v2 change:** the 3D error no longer comes from reading DA3's depth at the predicted pixel (v1's
only 3D number, and the source of the 1639mm miss) — it comes from `triangulate_nrays` over the
4 views' endpoints (§4B). The depth model's role has shrunk to producing an honest point cloud for
framing the renders, not carrying the 3D estimate by itself. `GT_UV`/`GT_XYZ`/`GT_DEPTH_M` are
used for the first time in this notebook, here.""")

code(r"""# 2D: the ORIGINAL view's own endpoint vs the true pixel (depth-model- and triangulation-independent).
_orig = next(v for v in PER_VIEW if v['name'] == 'original')
endpoint_err_2d_px = None
if _orig['endpoint_px'] is not None:
    endpoint_err_2d_px = float(np.hypot(_orig['endpoint_px'][0] - GT_UV[0], _orig['endpoint_px'][1] - GT_UV[1]))

# 3D: triangulated endpoint (from §4B) vs GT_XYZ -- REPLACES the old depth-lookup 3D estimate.
endpoint_err_3d_mm = None
if tri_endpoint is not None:
    endpoint_err_3d_mm = float(np.linalg.norm(tri_endpoint - GT_XYZ) * 1000.0)

# Load Stage A's calibration diagnostics (best-effort -- purely informational in this metrics
# dict, not used to compute any of the above).
_depth_calib = {}
try:
    with open('viz/bench4_depth_calib.json') as _fjs:
        _depth_calib = json.load(_fjs)
except Exception as _e:
    print('no viz/bench4_depth_calib.json found (%s) -- Stage A calibration diagnostics omitted' % _e)

metrics = {
    'frame': IMG_PATH,
    'endpoint_err_2d_px': None if endpoint_err_2d_px is None else round(endpoint_err_2d_px, 1),
    'endpoint_err_3d_mm': None if endpoint_err_3d_mm is None else round(endpoint_err_3d_mm, 1),
    'gripper_anchor_residual_mm': None if gripper_anchor_residual_mm is None else round(gripper_anchor_residual_mm, 1),
    'tri_residual_mm': None if tri_endpoint_resid is None else round(tri_endpoint_resid * 1000.0, 2),
    'tri_cond': tri_endpoint_cond,
    'tri_n_inliers': tri_endpoint_n,
    'tri_views_used': endpoint_views_used,
    'gripper_tri_residual_mm': None if tri_gripper_resid is None else round(tri_gripper_resid * 1000.0, 2),
    'gripper_tri_cond': tri_gripper_cond,
    'gripper_tri_views_used': start_views_used,
    'per_view_endpoint_px': {v['name']: (None if v['endpoint_px'] is None
                                          else [round(v['endpoint_px'][0], 1), round(v['endpoint_px'][1], 1)])
                             for v in PER_VIEW},
    'depth_calibration': _depth_calib,
    'rynnbrain_model': RYNNBRAIN_MODEL_ID,
    'depth_model': DEPTH_MODEL_ID,
}
print(json.dumps(metrics, indent=2))
with open('metrics.json', 'w') as fjs:
    json.dump(metrics, fjs, indent=2)
print('saved metrics.json')
""")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Report
# ─────────────────────────────────────────────────────────────────────────────
md("## 6. Report — Markdown, with every image + the honest caveat")

code(r"""_report_lines = []
def R(s=''): _report_lines.append(s)

R('# Benchmark 4 report (v2) — calibrated DA3 point-cloud + RynnBrain multi-view triangulation')
R('')
R('Frame: `%s` | gripper start px (%.1f, %.1f) | target prompt: "%s"'
  % (IMG_PATH, GRIPPER_TCP_UV[0], GRIPPER_TCP_UV[1], TARGET_LABEL))
R('')
R('## Depth calibration (§4A)')
R('')
if metrics.get('depth_calibration'):
    _dc = metrics['depth_calibration']
    R('Robust inverse-depth affine fit: a=%.6g b=%.6g (%d/%d anchors used after outlier rejection).'
      % (_dc.get('a', float('nan')), _dc.get('b', float('nan')),
         _dc.get('n_anchors_used', 0), _dc.get('n_anchors_total', 0)))
    R('Gripper-pixel check: raw %.3fm -> calibrated %.3fm (true %.3fm).'
      % (_dc.get('gripper_check_raw_m', float('nan')), _dc.get('gripper_check_calibrated_m', float('nan')),
         _dc.get('gripper_check_true_m', float('nan'))))
else:
    R('(no viz/bench4_depth_calib.json found -- Stage A calibration diagnostics unavailable)')
R('')
R('## Images')
R('')
R('Original RGB (the true side-cam view):')
R('')
R('![original](bench4_00_original_rgb.png)')
R('')
R('DA3 depth, calibrated:')
R('')
R('![depth](bench4_01_depth_da3.png)')
R('')
R('Rendered point-cloud viewpoints (re-renders of the SAME calibrated monocular cloud, not new geometry -- see the caveat below):')
R('')
for p in RENDER_PATHS:
    R('![%s](%s)' % (os.path.basename(p), os.path.basename(p)))
R('')
R('## Per-view trajectories (each an INDEPENDENT prediction — no view was shown a previous answer)')
R('')
for v in PER_VIEW:
    _png = 'bench4_05_traj_%s.png' % v['name']
    if os.path.exists('viz/' + _png):
        R('**%s** (given start px %s, endpoint px %s):' % (v['name'], v['given_start_px'], v['endpoint_px']))
        R('')
        R('![%s](%s)' % (v['name'], _png))
        R('')
    else:
        R('**%s**: no trajectory drawn (parse failed) — raw: `%s`' % (v['name'], v['raw'][:200].replace(chr(10), ' ')))
        R('')
R('## Triangulation')
R('')
R('- Endpoint (grasp target) triangulated from views %s: `%s`, residual %s m, cond %s, n_inliers %s/%d'
  % (metrics['tri_views_used'],
     None if tri_endpoint is None else [round(c, 4) for c in tri_endpoint.tolist()],
     None if tri_endpoint_resid is None else round(tri_endpoint_resid, 5), metrics['tri_cond'],
     metrics['tri_n_inliers'], len(PER_VIEW)))
R('- Gripper-anchor self-consistency: triangulated the RAW start points RynnBrain actually drew '
  '(views %s) -> `%s`, vs the KNOWN gripper `%s` -> **residual %s mm**'
  % (metrics['gripper_tri_views_used'],
     None if tri_gripper is None else [round(c, 4) for c in tri_gripper.tolist()],
     GRIPPER_TCP_XYZ.tolist(), metrics['gripper_anchor_residual_mm']))
R('')
R('## Metrics')
R('')
R('```json')
R(json.dumps(metrics, indent=2))
R('```')
R('')
R('## Conclusion')
R('')
if metrics['endpoint_err_3d_mm'] is not None:
    R('Triangulated 3D endpoint error on this single frame: **%.1f mm** (vs v1\'s depth-lookup '
      '1639.3mm). 2D endpoint error in the original view: **%s px**. Gripper-anchor '
      'self-consistency residual: **%s mm** — a LOW value here means the pose geometry and '
      'RynnBrain\'s per-view compliance are both trustworthy, so a large endpoint error would '
      'point at the target localisation itself rather than the pipeline plumbing; a HIGH value '
      'means the endpoint number is confounded by pipeline geometry error and should not be read '
      'at face value. **n=1 -- directional signal only, not a statistically powered claim; do '
      'not generalise from one frame.**'
      % (metrics['endpoint_err_3d_mm'], metrics['endpoint_err_2d_px'], metrics['gripper_anchor_residual_mm']))
else:
    R('Triangulation FAILED (fewer than 2 views produced a parseable endpoint) -- see the raw '
      'per-view outputs above for why.')
R('')
R('**Honest caveat (novel views of a monocular, now-calibrated cloud):** the 3 rendered '
  'viewpoints above are re-renders of ONE monocular point cloud -- they do not reveal any surface '
  'the original camera could not already see (front-facing points only, with holes at '
  'silhouettes and behind occluders). Calibrating the depth (§4A) fixes the cloud\'s ABSOLUTE '
  'SCALE, not this fundamental single-view coverage limit. So triangulating RynnBrain\'s per-view '
  'picks converts its 2D reasoning into a 3D estimate and AVERAGES OUT its per-view localisation '
  'noise/pose error — it cannot invent depth or geometry the calibrated cloud does not contain. '
  'This is exactly why the depth calibration in §4A is a prerequisite for this experiment to mean '
  'anything: triangulating over badly-scaled renders would still triangulate confidently to the '
  'wrong place. Read the gripper-anchor residual above as the honest bound on what this pipeline '
  '(pose geometry + calibration + RynnBrain compliance) can currently promise, independent of '
  'whether RynnBrain correctly identifies the target itself.')

_report_md = '\n'.join(_report_lines)
with open('viz/bench4_report.md', 'w') as fmd:
    fmd.write(_report_md)
print(_report_md)
""")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Push
# ─────────────────────────────────────────────────────────────────────────────
md("## 7. Push the run log")

code(r"""import subprocess, datetime, os, shutil
from google.colab import userdata
print('=== results snapshot ===')
print('metrics:', globals().get('metrics', '(eval cell not run yet)'))
def _sh(args, secret=None):
    r = subprocess.run(args, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if secret and out: out = out.replace(secret, '***')
    if out: print(out)
    return r.returncode
try: _TOKEN = userdata.get('GH_TOKEN')
except Exception as _e: _TOKEN = None; print('No GH_TOKEN secret:', _e)
if not _TOKEN:
    print('Set GH_TOKEN in the Colab Secrets panel, then re-run.')
else:
    _REPO = 'Yunsmn/RoboticsPerceptionTest'
    try: _f.flush()
    except Exception: pass
    os.makedirs('logs', exist_ok=True)
    _stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S'); _dest = 'logs/run4_%s.md' % _stamp
    shutil.copy('run_log.md', _dest)
    _url = 'https://%s@github.com/%s.git' % (_TOKEN, _REPO)
    _sh(['git','config','user.email','younes.menfalouti@um6p.ma']); _sh(['git','config','user.name','Yunsmn'])
    _sh(['git','pull','--rebase','--autostash', _url, 'main'], secret=_TOKEN)
    _sh(['git','add', _dest]); _sh(['git','add', 'viz'])
    _sh(['git','add', 'metrics.json']); _sh(['git','add', 'trajectory.json'])
    _sh(['git','commit','-m','log: benchmark4 run %s' % _stamp])
    _rc = _sh(['git','push', _url, 'HEAD:main'], secret=_TOKEN)
    print(('PUSHED ' if _rc == 0 else 'PUSH FAILED (rc=%d) ' % _rc) + _dest)
    print('-> tell the author: pull and read ' + _dest)
""")

# ─────────────────────────────────────────────────────────────────────────────
NB = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

if __name__ == "__main__":
    out_path = "benchmark4.ipynb"
    with open(out_path, "w") as fh:
        json.dump(NB, fh, indent=1)
    n_code = sum(1 for c in CELLS if c["cell_type"] == "code")
    n_md = sum(1 for c in CELLS if c["cell_type"] == "markdown")
    print("wrote %s: %d cells (%d code, %d markdown)" % (out_path, len(CELLS), n_code, n_md))
