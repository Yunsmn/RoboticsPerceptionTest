# Reference results — pipelines already measured on SO-101

Same simulation, same fixed side camera (az 135°, el −35°, dist 1.15 m, lookat (0,0,0.04),
FOV 45°, 640×480). The cube is 30 mm; the grasp basin is ~2 mm, so localization error is the
number that gates pick success.

| pipeline | detector | metric-depth source | loc error (median) | pick | honesty |
|---|---|---|---|---|---|
| HSV + plane | red colour threshold | known support plane z=0.015 | ~2.0 mm | 85% | colour + plane priors (both cheats) |
| SAM + plane | FastSAM (colour-free) | known support plane z=0.015 | ~1.7 mm | 68% | plane prior |
| mono SAM + depth | FastSAM | Depth-Anything V2 (affine, calibrated) | ~12 mm | 53% | honest, no plane — but too coarse |
| 3-cam triangulation | FastSAM | ray geometry (3 fixed cams) | **1.1 mm / z 0.6 mm** | — | honest, plane-free, multi-view |

Takeaways that frame this benchmark:

- Accuracy is set by the **depth source**, not the detector. Swapping colour→SAM was a
  robustness/generality win at ~equal accuracy; the accuracy ladder is *assumed plane* (cheat,
  ~2 mm) → *monocular affine estimate* (honest, ~12 mm) → *multi-view geometry* (honest, ~1 mm).
- The open question this repo answers: can a **strong single-image metric-depth model**
  (Depth Pro / UniDepthV2 / Metric3D-v2) — no plane, no triangulation, no RGB-D — land inside
  the ~2 mm basin at this standoff? If yes, `find_object` is one camera + one model. If no, the
  metric model positions coarsely and the wrist camera closes the last millimetres.
- The container tolerates centimetres regardless, so single-shot metric depth is always enough
  for the place target.
