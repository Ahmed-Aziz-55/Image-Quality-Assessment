# ImageQualityAssessment — Architecture Decisions

This document records key design decisions for the Image Quality Assessment
project, along with the reasoning behind each choice.

---

## 1. Classical computer vision heuristics, not a trained model (initial approach)

The initial detectors (blur, darkness, glare) use deterministic image
processing formulas — no training, no labeled dataset, no neural network.

**Reasoning:** no labeled dataset was provided for this project, and
collecting/labeling one would cost significant time against a tight
deadline. Classical CV heuristics (Laplacian variance for blur, mean
brightness for darkness, saturated-pixel ratio for glare) are well-known,
explainable, and require zero training data — they produce a working,
testable system immediately. This also has a practical evaluation
advantage: every decision is a documented formula, not a black box, so it
can be explained precisely if asked. A deep-learning comparison (e.g. a
CNN trained on synthetically-degraded images) is a planned follow-up if
time allows, to benchmark against this baseline.

---

## 2. Each detector returns a raw score, not just a boolean

`BlurDetector.is_blurry()`, `DarknessDetector.is_dark()`, and
`GlareDetector.has_glare()` all return `(bool, float)` — the decision AND
the underlying numeric score.

**Reasoning:** returning only a boolean would hide the reasoning behind a
decision. Keeping the raw score means thresholds can be tuned, re-evaluated,
or reported without re-running detection, and results can be sorted/ranked
by severity rather than just pass/fail.

---

## 3. Glare threshold tuned after false positives on real photos

Initial glare detection used `brightness_cutoff=240, area_threshold=0.05`
(pixels ≥240 brightness, flagged if they cover >5% of the image). Tested
against 10 real photos from an existing dataset (Flickr30k images, reused
read-only from a separate project for testing purposes only), 4 of 10 were
flagged as having glare.

Manual visual inspection of one flagged image (`1000344755.jpg`) showed a
wall in ordinary sunlight — bright, but not a genuine glare defect (no
flash reflection, no blown-out highlight spot). The threshold was too
sensitive: normal sunlit surfaces easily exceed 240/255 brightness without
being a quality defect.

**Fix:** raised `brightness_cutoff` to 250 (only near-pure-white pixels
count — genuine glare tends to be tightly clipped near 255, not just
"bright") and `area_threshold` to 0.08 (requires a larger saturated area
before flagging, reducing sensitivity to small bright patches like sky
slivers or highlights). Re-tested on the same 10 images: 0 false positives.

**Reasoning for documenting this:** this is a concrete example of why
testing against real data (not just synthetic test cases) matters —
the synthetic unit test for `GlareDetector` passed with the original
thresholds, because it used a deliberately extreme 100% synthetic patch.
Real photos exposed a calibration problem that synthetic tests alone did
not catch.
