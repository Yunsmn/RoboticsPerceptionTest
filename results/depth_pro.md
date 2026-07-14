# Depth Pro (Apple `ml-depth-pro`, full model)

Monocular metric depth. We pass the known focal length so it skips its own focal estimation.
Depth map is `640 × 480` optical-axis depth in metres.

## Headline numbers (48 frames)

| Correction | median | p95 | notes |
|---|---|---|---|
| raw (native metric) | 603.0 mm | 662.5 mm | absolute scale is badly off |
| oracle-scaled (cheats) | 9.4 mm | 30.4 mm | upper bound with a perfect scale |
| **base-anchored (honest)** | 12.8 mm | 26.1 mm | scale from one known point |
| **floor-anchored (honest)** | **11.2 mm** | **24.7 mm** | scale from 476 known floor pixels |

Runs: `logs/run_20260713_125520.md` (base anchor), `logs/run_20260713_132856.md` (floor anchor).

## What we learned, in order

**1. The raw error is huge but not random.** 603 mm median, and the distribution is *tight*
(mean 594, p95 662 — only ~60 mm spread). Random failure would scatter; a tight cluster means a
single systematic offset.

**2. It's a pure global scale.** The ratio `gt / pred` is **0.64** and essentially identical on
every frame (0.61–0.65). Depth Pro overestimates depth by ~1.55× (it reads the cube at ~1.6 m when
the truth is ~1.05 m) — but consistently. Apply one global scale and the 603 mm collapses to the
9.4 mm oracle.

**3. The scale is not a focal-length bug.** We probed three focal settings on one frame:
`f = 579.4` → ratio 0.64, `f = None` (let it estimate) → 0.78, `f = 1390.6` (canonical-res guess)
→ 0.27. None reads natively metric. The scale error is the model's own, an out-of-distribution
effect on clean synthetic imagery — only an external anchor fixes it.

**4. The honest anchor nearly matches the oracle.** Recovering the scale from the known base pixel
alone gives 12.8 mm — within ~3 mm of the cheating oracle, with a *tighter* p95 because the anchor
adapts per frame. The base pixel reads 1.841 m, which times the 0.637 scale is 1.173 m — exactly
the true base depth, so the anchor is clean and unoccluded.

**5. Dense floor calibration helps only a little — and that's the key finding.** Using 476 known
floor pixels instead of one base point moves the median from 12.8 to 11.2 mm and the p95 from 26.1
to 24.7 mm. Small, because the scale was already uniform. The residual is therefore **not** scale
noise. It is Depth Pro **failing to resolve the cube**: the depth image
([`../viz/depth_depth_pro.png`](../viz/depth_depth_pro.png)) is almost entirely a smooth ground-plane
gradient (1.17 m near the bottom, 7.70 m far at the top), and the 3 cm cube barely registers. It
lands ~11 mm only because the cube sits on the floor, so the floor's depth at that pixel is close to
the cube's. No scale trick can beat the 9.4 mm oracle — that number is Depth Pro's intrinsic depth
resolution on this scene.

## Depth-map sanity (confirms the read convention)
Shape `(480, 640)`, dtype `float32`, range 1.170 → 7.698 m, median 1.885 m. At the cube pixel it
reads 1.626 m (true 1.048 m); at the base pixel 1.841 m (true 1.173 m). Orientation, indexing
(`depth[v, u]`), and units all check out.

## Verdict
Chosen as the stage-1 model. Honest single-shot ceiling ≈ **11 mm median / 25 mm p95** — coarse,
about 5× short of the 2 mm grasp bar, limited by object resolution rather than scale.
