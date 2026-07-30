# ImageQualityAssessment — Architecture Decisions

This document records key design decisions for the Image Quality Assessment
project, along with the reasoning behind each choice. Intended for
internship evaluation and as a personal reference.

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
read-only from a separate local project for testing purposes only — no
images are bundled with or committed to this repository), 4 of 10 were
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
Broader test on 50 real images after the fix: 46 passed, 4 failed for
confirmed genuine reasons (1 underexposed image, 3 images with real
glare — all visually verified).

**Reasoning for documenting this:** this is a concrete example of why
testing against real data (not just synthetic test cases) matters — the
synthetic unit test for `GlareDetector` passed with the original
thresholds, because it used a deliberately extreme 100%-saturated synthetic
patch. Real photos exposed a calibration problem that synthetic tests
alone did not catch.

---

## 4. Known interaction: extremely dark images can also trigger a blur flag

Tested with a real near-black textured image via the `/assess` endpoint:
`darkness.flagged=true` (score=4.44, correctly very dark) but also
`blur.flagged=true` (score=59.13, below the 100.0 threshold).

**Why this happens:** Laplacian variance measures edge contrast. In a
near-black image, pixel values are naturally compressed into a narrow
range near 0 — even genuinely sharp edges produce a smaller absolute
variance than the same edges would in a well-lit image, because there's
less brightness range for the edge to "jump" across. This is not a bug in
either detector; it's an inherent property of measuring edge contrast on
low-brightness input.

**Implication:** on very dark images, the blur score is less reliable as
a standalone signal — a low blur score there may reflect lack of light,
not lack of focus. This is one motivation for using `QualityAssessor`'s
combined verdict rather than treating blur in isolation: if an image is
already flagged dark, its blur flag should be interpreted with that
context in mind. Documented here rather than "fixed," since attempting to
compensate (e.g. normalizing contrast before measuring blur) would change
the blur metric's behavior on well-lit images too, and no evidence yet
shows that's necessary — this is a one-image observation, not a validated
pattern across many dark images.

---

## 5. Positive validation: genuine high-glare image correctly flagged

To confirm the tuned glare threshold (Decision 3) wasn't tuned into being
*too* lenient, tested against a deliberate high-glare stock image (a
"starburst" graphic with a large, intensely bright center radiating
outward). Result: `glare.flagged=true, score=0.2168` (21.68% of pixels
near-white) — correctly detected.

**Reasoning:** Decision 3 only demonstrated the detector avoiding false
positives (not over-flagging). This test demonstrates the complementary
case — the detector still catches true positives after tuning, confirming
the threshold change didn't simply make the detector permissive enough to
pass everything. Together, decisions 3 and 5 bracket the threshold: too
sensitive (original) triggers on ordinary sunlight; the tuned threshold
lets ordinary sunlight through but still catches genuine, extreme glare.

---

## 6. FastAPI endpoint decodes uploads in memory, never writes to disk

`POST /assess` (`app/routers/quality.py`) reads the uploaded file's bytes,
converts them into an OpenCV image via `np.frombuffer` + `cv2.imdecode`,
and passes that directly to `QualityAssessor.assess()`.

**Reasoning:** unlike VisionSeek (which processes a fixed, pre-indexed
dataset), this service has no dataset of its own — every request is an
arbitrary image supplied by the caller. Saving each upload to disk before
processing would add unnecessary I/O and leave temp files to clean up.
Decoding directly from the uploaded bytes in memory avoids both, and keeps
the endpoint stateless (nothing about a given request persists after the
response is sent).

`QualityAssessor` is instantiated once at import time (not per-request) —
it holds no per-request state, so a single shared instance is safe and
avoids re-creating three detector objects on every call.
