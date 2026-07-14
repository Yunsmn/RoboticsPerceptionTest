# Depth benchmark — results summary

The question we set out to answer was a narrow one, but it decides the shape of the whole
pick-and-place pipeline: can a single fixed camera, with one monocular metric-depth model and none
of the usual crutches — no second camera, no depth sensor, no assumption that the object sits on a
known table — locate a three-centimetre cube well enough to grasp it? From the earlier grasping
work we know "well enough" means a localisation error of around two millimetres; past that the
grasp starts to miss. Everything below is measured against that bar, on forty-eight simulated
frames whose true cube position we know exactly and use only to score, never to help.

The first thing worth saying is that the measuring stick itself is trustworthy. Before any model
runs, the harness checks that its own geometry is exact: projecting a known point and then
back-projecting it recovers the original to within machine precision (about 3e-16 of a metre), and
the stored pixels really are the projections of the true centres to within six ten-thousandths of a
pixel. So whatever error the models show is theirs, not the harness's. The methodology is written
up in [`results/methodology.md`](results/methodology.md).

We tested the two leading candidates, Depth Pro and UniDepthV2, and they told the same story.
Straight out of the box both are wildly wrong in absolute terms — Depth Pro's median error is about
six hundred millimetres, UniDepth's over one and a half metres — but that number is misleading,
because the error is almost entirely a single, uniform scale. Depth Pro reads every depth about one
and a half times too far, and it does so on every frame; UniDepth reads about two and a half times
too far, just as consistently. Once you divide that scale back out, both collapse to roughly one
centimetre.

The interesting part is *how* we divide it out honestly. We can't use the true depths to find the
scale — that would be cheating — but we can use something we genuinely know: the robot's own base.
The base is the origin of our coordinate frame and it always sits at a known place in the image, so
reading the model's depth at that one pixel and comparing it to the base's true distance gives us
the scale for free, with no knowledge of the cube at all. That single-point "anchor" gets Depth Pro
to about thirteen millimetres — remarkably close to the best a perfect scale could ever do (about
nine). Widening the anchor from one point to four hundred and seventy-six known floor pixels tightens
it a little further, to about eleven millimetres typical and twenty-five in the worst five percent of
frames.

And that small improvement is, oddly, the most important result. If dense calibration barely helps,
the leftover error is not a scale problem at all — it is the model simply failing to see the cube.
The depth images make this plain: both models render the scene as a smooth floor receding into the
distance, and the little cube barely disturbs it. The reason our numbers are as good as eleven
millimetres is almost an accident — the cube rests on the floor, so "the floor's depth here" is close
to "the cube's depth here." A cube lifted off the table would expose the gap immediately. No amount
of scale correction can beat that nine-millimetre floor, because at that point we are asking the model
for detail it never resolved.

Between the two, Depth Pro wins and becomes the model we carry forward — not on the median, where the
two roughly tie, but on the worst case, where its twenty-five-millimetre p95 beats UniDepth's
thirty-one. The lesson from UniDepth is worth keeping, though: it takes the camera intrinsics and is
built to be metric, and it was still the *less* accurate of the two. That rules out "just feed it the
calibration" as the fix. The problem is genuinely that these networks, trained on real photographs,
are out of their element on a clean synthetic render of a tiny, untextured object.

So the honest conclusion is that a single monocular depth model, however good, converges at about a
centimetre here — roughly five times short of what a grasp needs — and the limit is object
resolution, not scale, which anchoring has already solved. That is not a dead end so much as a
signpost. The one approach in our reference numbers that clears the bar, at about 1.7 mm, uses no
depth model at all: it takes the object's outline and intersects its contact point with the known
floor geometry. That is the direction the next round explores — making the model actually see the
cube by cropping and zooming onto it, and reading the cube's footprint against the floor it stands
on — before we fall back on a close-range wrist camera, which we would rather keep in reserve.

Per-model detail is in [`results/depth_pro.md`](results/depth_pro.md) and
[`results/unidepth_v2.md`](results/unidepth_v2.md); the raw run logs are in [`logs/`](logs/) and the
depth images in [`viz/`](viz/).

## Numbers at a glance

Localisation error over 48 frames, in millimetres. "Oracle" uses the true depths to set the scale and
so only marks the best achievable; the anchored rows are what an honest pipeline actually gets.

| model | raw median | oracle (cheats) | base-anchored | floor-anchored |
|---|---|---|---|---|
| **Depth Pro** | 603 | 9.4 (p95 30) | 12.8 (p95 26) | **11.2 (p95 25)** |
| UniDepthV2 (ViT-L, res 9) | 1626 | 8.8 (p95 72) | 14.6 (p95 31) | — |

For reference, pipelines measured earlier on the same simulation: three-camera triangulation ~1.1 mm,
SAM plus known floor plane ~1.7 mm, monocular SAM + Depth-Anything ~12 mm. The grasp bar is ~2 mm.

A note on comparing these fairly. Every one of these reference numbers is already *corrected* — none
is a raw metric reading, so they line up with the anchored columns above, not the raw one. Depth-Anything
in particular is a relative model with no native scale at all: its ~12 mm comes from fitting a full
two-parameter affine (scale *and* offset) to known anchors, whereas Depth Pro reaches ~11.2 mm from just
a one-parameter scale, because it is already metric up to a constant. So Depth Pro matches or beats it
with a weaker correction. And because the reference figures come from an earlier, different harness, a
few millimetres between any of them should not be read as a real difference; the only controlled
comparison here is Depth Pro against UniDepthV2 on the same 48 frames.
