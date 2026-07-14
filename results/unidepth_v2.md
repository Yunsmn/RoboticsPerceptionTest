# UniDepthV2 (`lpiccinelli-eth/UniDepth`, ViT-L/14)

The largest UniDepth variant, run at `resolution_level = 9` (its main accuracy knob). Unlike
Depth Pro, it is designed to be metric *given the camera intrinsics*, which we supply — so it was
the natural candidate to beat Depth Pro on absolute scale.

## Headline numbers (48 frames)

| Correction | median | p95 | notes |
|---|---|---|---|
| raw (native metric) | 1626.1 mm | 1826.7 mm | scale further off than Depth Pro |
| oracle-scaled (cheats) | 8.8 mm | 72.1 mm | good median, loose worst-case |
| **base-anchored (honest)** | 14.6 mm | 31.2 mm | worse than Depth Pro on both |

Run: `logs/run_20260713_131242.md` (resolution_level 9). An earlier run with the setting left unset
gave a looser 12.8 mm / 88.1 mm p95 — not a fair measurement, hence the fix.

## What we learned

**1. Taking the intrinsics did not make it metric.** Despite consuming `K`, UniDepth's absolute
scale is *worse* than Depth Pro's: the `gt / pred` ratio is **0.39** (it overestimates depth by
~2.6×), against Depth Pro's 0.64. This is the clearest sign that both models are simply
out-of-distribution on clean synthetic renders — the intrinsics fix the geometry, not the learned
scale prior.

**2. The weights loaded fine.** The console prints "Not loading pretrained weights for backbone",
which is benign — the predicted depth tracks the true depth monotonically across frames (2.49 →
2.51 → 2.59 … as truth goes 0.956 → 0.979 → 1.002 …), which an unweighted network could not do. The
weights come from the checkpoint, not a separate backbone download.

**3. Same plane read, and less consistent.** The depth image
([`../viz/depth_unidepth_v2.png`](../viz/depth_unidepth_v2.png)) is the same smooth ground-plane
gradient as Depth Pro, with the objects even fainter. Its honest anchored p95 (31.2 mm) is looser
than Depth Pro's (26.1 mm), meaning its worst frames are worse. On the same footing UniDepth is the
weaker of the two.

## Depth-map sanity
Range 1.929 → 9.029 m, median 3.037 m. At the cube pixel 2.598 m (true 1.048 m); at the base pixel
2.897 m (true 1.173 m) — consistent with the 0.39 scale.

## Verdict
Not selected. It confirms, rather than lifts, the single-shot ceiling: monocular metric depth on
this synthetic scene caps at ~1 cm regardless of which of the two leading models we use, because
neither resolves the small object.
