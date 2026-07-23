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
md(r"""# SO-101 Perception Benchmark 4 — DA3 point-cloud + RynnBrain iterative trajectory (proof-of-concept)

Single frame, two heavy models that cannot share a Python process (conflicting pip pins), so this
notebook runs in **two stages, each its own Colab kernel**:

- **Stage A** (§1A install → Restart → §0,§2,§3A,§4A): **Depth Anything 3** (`DA3METRIC-LARGE`)
  predicts metric depth on the ORIGINAL RGB only → builds a point cloud (`camera_math` calibration,
  no ground truth) → renders 3 re-projected viewpoints with **Open3D** → saves RGB + depth viz +
  cloud (`.ply`) + 3 renders to `viz/`, then frees DA3 from memory.
- **Stage B** (§1B install → Restart → §0,§2,§3B,§4B): **RynnBrain** (VLM) predicts a manipulation
  trajectory from the gripper's known start to a grasp on the target, seeing ONLY the original RGB
  first, then refining that SAME trajectory once per rendered view (never generating a fresh one).
  DA3 is never imported in this kernel; it only reads the artifacts Stage A left on disk.
- **§5/§6/§7** run in the same kernel as Stage B (no third restart): evaluate against held-out
  ground truth, write a Markdown report, push.

**Read `docs/archive` conventions aside — this is new territory for the repo (two restarts, not
one), so the run order is spelled out explicitly at every step below. Follow it literally; do not
just click "Run all" — see the warning in §1A.**

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
        return (float(focal_px) * raw / 300.0).astype(np.float32)

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
md(r"""## 4A. Stage A — depth, point cloud, 3 rendered viewpoints (writes `viz/`, then frees DA3)

Only `img0` (the original RGB) and this frame's own DA3 depth go into the cloud — no ground
truth anywhere. The 3 renders are saved to disk so Stage B (a separate kernel, after §1B +
restart) can read them without ever importing DA3 or torch/xformers built for it.""")

code(r"""depth_m = DEPTH.infer_metric_depth(img0, f)
np.save('viz/bench4_depth_m.npy', depth_m)
_gu, _gv = int(round(GRIPPER_TCP_UV[0])), int(round(GRIPPER_TCP_UV[1]))
print('depth range: %.3f - %.3f m | at gripper-start px: %.3f m (FK says %.3f m)'
      % (float(depth_m.min()), float(depth_m.max()), float(depth_m[_gv, _gu]), GRIPPER_TCP_DEPTH_M))

import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 4.5))
im = ax.imshow(depth_m, cmap='turbo'); ax.set_title('DA3 metric depth (m)'); ax.axis('off')
plt.colorbar(im, ax=ax, fraction=0.046)
plt.savefig('viz/bench4_01_depth_da3.png', dpi=110, bbox_inches='tight'); plt.close()
print('saved viz/bench4_01_depth_da3.png')
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

code(r"""# --- 3 complementary re-rendered viewpoints of the SAME monocular cloud (see the honest caveat
# in §6). Views rotate the ORIGINAL side-camera azimuth by ~45deg increments around the scene's
# own look-at point (CM.SIDE_LOOKAT -- a known SCENE/rig constant the side camera itself already
# uses to render every data/ frame, not anything derived from an object's ground truth).
VIEWS = [
    ('oblique45', CM.SIDE_AZ - 45.0, CM.SIDE_EL),
    ('left', CM.SIDE_AZ - 90.0, CM.SIDE_EL),
    ('right', CM.SIDE_AZ + 90.0, CM.SIDE_EL),
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

RENDER_PATHS = []
for name, az, el in VIEWS:
    vf, vcx, vcy, vcam_pos, vR_cw = CM.free_cam_params(az, el, CM.SIDE_DIST, CM.SIDE_LOOKAT, W=W, H=H)
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
    print('saved', out_path, '(open3d)' if ok else '(fallback splat)')
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
md(r"""## 3B. RynnBrain adapter — trajectory prediction + refinement

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

**Assumptions this adapter makes beyond what the cookbook demonstrates — verify on the first
Colab run, not asserted as fact:**
1. Telling the model the start point in the instruction text ("its current position is pixel
   (x,y)... the FIRST point of your trajectory must be...") makes it anchor there. If it doesn't,
   §4B numerically re-anchors the trajectory at the known `GRIPPER_TCP_UV` (logged, not silent).
2. The "refine, don't regenerate" instruction (§4B, feeding the original + a rendered view + the
   previous `<trajectory>` back in) is a construction for this experiment — the cookbook only
   shows single-shot prediction over independent inputs, never an explicit refine-given-history
   turn. If the model ignores history and free-generates instead, that will show up directly as
   low trajectory consistency across iterations (§5 reports this) rather than failing silently.
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
md(r"""## 4B. Stage B — initial trajectory, then iterative refinement over the 3 rendered views

The FIRST image RynnBrain ever sees is the original RGB (never a render). Each refinement step
adds exactly one rendered view on top of the original + the previous trajectory text, and is
told to refine, not regenerate. `RENDER_PATHS` is rebuilt from disk (`glob`) because Stage A's
Python variables did not survive the restart — only its files did.""")

code(r"""RENDER_PATHS = sorted(glob.glob('viz/bench4_02_render_*.png'))
assert RENDER_PATHS, 'No renders found in viz/ -- run Stage A (1A/2/3A/4A) first, in an earlier kernel.'
print('found', len(RENDER_PATHS), 'rendered views:', RENDER_PATHS)

START_U01000 = GRIPPER_TCP_UV[0] / W * 1000.0
START_V01000 = GRIPPER_TCP_UV[1] / H * 1000.0

FORMAT_INSTR = ('Return up to 10 key trajectory points as a list of tuples in the format: '
                '<trajectory> (x1, y1), (x2, y2), ... </trajectory>. All coordinates normalised '
                'to the [0, 1000] range, matching this image\'s own width and height.')

initial_prompt = (
    'You are controlling a robot gripper. Its current position is the pixel (%.0f, %.0f) '
    '(normalised 0-1000). Predict a collision-aware manipulation trajectory that moves the '
    'gripper from its current position to a grasp pose on the %s. The FIRST point of your '
    'trajectory must be the gripper\'s given start position. %s'
    % (START_U01000, START_V01000, TARGET_LABEL, FORMAT_INSTR)
)
print(initial_prompt)

raw0 = RYNN.predict([IMG_PATH], initial_prompt)
print('--- raw RynnBrain output (initial) ---'); print(raw0)
traj0 = parse_trajectory(raw0, W, H)
if not traj0:
    raise RuntimeError('Could not parse any (x,y) points from the initial RynnBrain response -- '
                        'the prompt/format assumption in 3B may not hold for this checkpoint; '
                        'inspect raw0 above and adjust FORMAT_INSTR / TRAJ_RE.')
_start_err_px = float(np.hypot(traj0[0][0] - GRIPPER_TCP_UV[0], traj0[0][1] - GRIPPER_TCP_UV[1]))
print('model start vs given gripper origin: %.1f px' % _start_err_px)
# Numeric safety net (logged, never silent): the trajectory must NUMERICALLY start at the known
# gripper origin regardless of what the model's own first token said.
traj0 = [GRIPPER_TCP_UV] + traj0
""")

code(r"""def draw_traj(img_path, traj, out_path, title):
    im = Image.open(img_path).convert('RGB'); draw = ImageDraw.Draw(im)
    for i in range(len(traj) - 1):
        draw.line([traj[i], traj[i + 1]], fill=(255, 0, 0), width=3)
    for i, (px, py) in enumerate(traj):
        r = 6
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 220, 255) if i else (0, 255, 0))
    im.save(out_path)
    return out_path

TRAJ_HISTORY = [{'stage': 'initial', 'points_px': [list(p) for p in traj0], 'raw': raw0}]
draw_traj(IMG_PATH, traj0, 'viz/bench4_03_traj_initial.png', 'initial trajectory')
print('saved viz/bench4_03_traj_initial.png')
""")

code(r"""traj_prev = traj0
for i, render_path in enumerate(RENDER_PATHS):
    prev_str = ', '.join('(%.0f, %.0f)' % (p[0] / W * 1000.0, p[1] / H * 1000.0) for p in traj_prev)
    refine_prompt = (
        'Image 1 is the original view. Image 2 is the SAME scene re-rendered from a different '
        'viewpoint (built from a monocular depth point cloud -- it may have holes or artifacts). '
        'Your previous predicted trajectory, in Image 1\'s own pixel coordinates (normalised '
        '0-1000), for moving the gripper from its start position to grasp the %s was: '
        '<trajectory> %s </trajectory>. Using Image 2 only to better judge depth and possible '
        'collisions, REFINE this trajectory -- do not generate an unrelated new one, and keep it '
        'anchored at the same start position. %s'
        % (TARGET_LABEL, prev_str, FORMAT_INSTR)
    )
    raw_i = RYNN.predict([IMG_PATH, render_path], refine_prompt)
    print('--- raw RynnBrain output (refine %d, %s) ---' % (i + 1, os.path.basename(render_path)))
    print(raw_i)
    traj_i = parse_trajectory(raw_i, W, H)
    if not traj_i:
        print('  could not parse -- keeping the previous trajectory for this iteration')
        traj_i = traj_prev
    else:
        _s_err = float(np.hypot(traj_i[0][0] - GRIPPER_TCP_UV[0], traj_i[0][1] - GRIPPER_TCP_UV[1]))
        if _s_err > 20.0:               # model drifted the start -- re-anchor, don't silently trust it
            traj_i = [GRIPPER_TCP_UV] + traj_i
    TRAJ_HISTORY.append({'stage': 'refine_%d_%s' % (i + 1, os.path.basename(render_path)),
                          'points_px': [list(p) for p in traj_i], 'raw': raw_i})
    draw_traj(IMG_PATH, traj_i, 'viz/bench4_04_traj_refine_%d.png' % (i + 1), 'refine %d' % (i + 1))
    traj_prev = traj_i

TRAJ_FINAL = traj_prev
with open('trajectory.json', 'w') as fjs:
    _json.dump({'history': TRAJ_HISTORY, 'final_px': [list(p) for p in TRAJ_FINAL],
                'gripper_start_px': list(GRIPPER_TCP_UV), 'image_wh': [int(W), int(H)]}, fjs, indent=2)
print('saved trajectory.json with', len(TRAJ_HISTORY), 'stages; final has', len(TRAJ_FINAL), 'points')
RYNN.free()
print('RynnBrain freed. Stage B trajectory prediction complete -- proceed to 5. Evaluate.')
""")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Evaluate
# ─────────────────────────────────────────────────────────────────────────────
md(r"""## 5. Evaluate — 2D pixel error, 3D world error (DA3-depth confound), improvement, consistency

Reports the 2D pixel endpoint error (depth-model-independent — pure "did it point at the cube")
and the 3D world error (lifts the predicted pixel through DA3's OWN saved depth, never
`GT_DEPTH_M`) **separately**, so trajectory accuracy and depth-model quality don't get conflated
into one number. `GT_UV`/`GT_XYZ`/`GT_DEPTH_M` are used for the first time in this notebook, here.""")

code(r"""depth_m_saved = np.load('viz/bench4_depth_m.npy')     # from Stage A, persisted across the restart

pred_uv = TRAJ_FINAL[-1]                                  # final trajectory endpoint = predicted grasp pixel
err_2d_px_final = float(np.hypot(pred_uv[0] - GT_UV[0], pred_uv[1] - GT_UV[1]))
init_uv = traj0[-1]
err_2d_px_initial = float(np.hypot(init_uv[0] - GT_UV[0], init_uv[1] - GT_UV[1]))

pu = int(round(np.clip(pred_uv[0], 0, W - 1))); pv = int(round(np.clip(pred_uv[1], 0, H - 1)))
pred_depth_m = float(depth_m_saved[pv, pu])
pred_xyz = CM.point_at_depth(pred_uv[0], pred_uv[1], f, cx, cy, cam_pos, R_cw, pred_depth_m)
err_3d_mm_final = float(np.linalg.norm(pred_xyz - GT_XYZ) * 1000.0)

def path_len(traj):
    return float(sum(np.hypot(traj[i + 1][0] - traj[i][0], traj[i + 1][1] - traj[i][1])
                      for i in range(len(traj) - 1)))

path_lengths = [round(path_len(h['points_px']), 1) for h in TRAJ_HISTORY]
endpoints = np.array([h['points_px'][-1] for h in TRAJ_HISTORY])
consistency_px = float(np.mean(np.linalg.norm(endpoints[1:] - endpoints[:-1], axis=1))) if len(endpoints) > 1 else 0.0

metrics = {
    'frame': IMG_PATH,
    'endpoint_err_2d_px_initial': round(err_2d_px_initial, 1),
    'endpoint_err_2d_px_final': round(err_2d_px_final, 1),
    'improvement_2d_px': round(err_2d_px_initial - err_2d_px_final, 1),
    'endpoint_err_3d_mm_final': round(err_3d_mm_final, 1),
    'pred_depth_m_at_endpoint': round(pred_depth_m, 4),
    'gt_depth_m_at_endpoint': round(GT_DEPTH_M, 4),
    'path_length_px_per_stage': path_lengths,
    'endpoint_consistency_px_mean_step': round(consistency_px, 1),
    'n_refinement_stages': len(TRAJ_HISTORY) - 1,
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

R('# Benchmark 4 report — DA3 point-cloud + RynnBrain iterative trajectory')
R('')
R('Frame: `%s` | gripper start px (%.1f, %.1f) | target prompt: "%s"'
  % (IMG_PATH, GRIPPER_TCP_UV[0], GRIPPER_TCP_UV[1], TARGET_LABEL))
R('')
R('## Images')
R('')
R('Original RGB (the only image RynnBrain sees first):')
R('')
R('![original](bench4_00_original_rgb.png)')
R('')
R('DA3 predicted metric depth:')
R('')
R('![depth](bench4_01_depth_da3.png)')
R('')
R('Rendered point-cloud viewpoints (re-renders of the SAME monocular cloud, not new geometry -- see the caveat below):')
R('')
for p in RENDER_PATHS:
    R('![%s](%s)' % (os.path.basename(p), os.path.basename(p)))
R('')
R('## Trajectory refinement')
R('')
R('![initial trajectory](bench4_03_traj_initial.png)')
R('')
for i in range(len(RENDER_PATHS)):
    R('![refine %d](bench4_04_traj_refine_%d.png)' % (i + 1, i + 1))
R('')
R('## Metrics')
R('')
R('```json')
R(json.dumps(metrics, indent=2))
R('```')
R('')
R('## Raw trajectory per stage')
R('')
for h in TRAJ_HISTORY:
    _raw_1line = h['raw'].replace(chr(10), ' ')[:200]
    R('- **%s**: %d points, raw model text: `%s`' % (h['stage'], len(h['points_px']), _raw_1line))
R('')
R('## Conclusion')
R('')
did_improve = metrics['improvement_2d_px'] > 0
R('Iterative multi-view point-cloud refinement %s the endpoint on this single frame (initial '
  '%.1f px -> final %.1f px from the true pixel; %+.1f px). **n=1 -- directional signal only, '
  'not a statistically powered claim; do not generalise from one frame.**'
  % ('IMPROVED' if did_improve else 'did NOT improve',
     metrics['endpoint_err_2d_px_initial'], metrics['endpoint_err_2d_px_final'], metrics['improvement_2d_px']))
R('')
R('**Honest caveat (novel views of a monocular cloud):** the 3 "novel" viewpoints above are '
  're-renders of ONE monocular point cloud -- they do not reveal any surface the original camera '
  'could not already see (front-facing points only, with holes at silhouettes and behind '
  'occluders). So this experiment tests whether re-rendered depth CUES from a different angle '
  'help RynnBrain reason about a fixed, already-known set of 3D points -- not whether genuinely '
  'new geometry (a second real camera, a wrist-mounted sensor) would help. A positive result '
  'here says "showing the same facts differently helps a VLA reason"; it is not evidence that '
  'multi-view re-rendering could substitute for real multi-camera triangulation.')
R('')
R('**3D error confound:** the 3D world error lifts the predicted pixel through DA3\'s OWN metric '
  'depth (never ground truth) -- it is therefore a joint score of (trajectory endpoint accuracy) '
  '× (DA3 depth accuracy at that pixel), not trajectory accuracy alone. The 2D pixel error '
  'above isolates the trajectory-only signal; read the two together, not the 3D number in isolation.')

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
    _sh(['git','config','user.email','younesosf@gmail.com']); _sh(['git','config','user.name','Yunsmn'])
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
