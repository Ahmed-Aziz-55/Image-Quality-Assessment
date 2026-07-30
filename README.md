# ImageQualityAssessment

ImageQualityAssessment is a no-reference image quality checker that detects
three common defects — blur, darkness (underexposure), and glare
(overexposed highlights) — using classical computer vision heuristics. No
training or labeled dataset is required.

## Approach

All three detectors use deterministic, explainable formulas rather than a
trained model:

1. **Blur** — measured via the variance of the Laplacian (an edge-detection
   operator). Sharp images have high-variance edges; blurred images have
   low-variance, smeared edges.
2. **Darkness** — measured via mean grayscale pixel brightness (0-255
   scale). A low mean indicates an underexposed image.
3. **Glare** — measured via the proportion of near-white (saturated)
   pixels. Unlike overall brightness, glare is usually localized to a
   small region, so counting the fraction of extreme-bright pixels catches
   it where a whole-image average would not.

No labeled dataset was available for this project, so a classical CV
baseline was chosen first — it's fast to build, requires zero training
data, and every decision is a documented formula rather than a black box.
See [app/docs/Decisions.md](app/docs/Decisions.md) for the full reasoning,
including a real calibration issue found and fixed during testing.

## Components

- [app/quality/blur_detector.py](app/quality/blur_detector.py) —
  `BlurDetector`, Laplacian-variance blur detection.
- [app/quality/darkness_detector.py](app/quality/darkness_detector.py) —
  `DarknessDetector`, mean-brightness darkness detection.
- [app/quality/glare_detector.py](app/quality/glare_detector.py) —
  `GlareDetector`, saturated-pixel-ratio glare detection.
- [app/quality/assessor.py](app/quality/assessor.py) — `QualityAssessor`,
  combines all three checks into a single pass/fail verdict with per-check
  scores.
- [app/core/logging_config.py](app/core/logging_config.py) — centralized
  logging setup.
- [app/tests/](app/tests/) — unit tests for all three detectors (9 tests).

## Usage

```python
from app.quality.assessor import QualityAssessor

assessor = QualityAssessor()
result = assessor.assess_path("path/to/image.jpg")

print(result)
# {
#   "image_path": "path/to/image.jpg",
#   "loaded": True,
#   "passed": True,
#   "blur": {"is_blurry": False, "score": 4963.9},
#   "darkness": {"is_dark": False, "score": 147.6},
#   "glare": {"has_glare": False, "score": 0.0409}
# }
```

Individual detectors can also be used standalone:

```python
from app.quality.blur_detector import BlurDetector
import cv2

img = cv2.imread("path/to/image.jpg")
detector = BlurDetector(threshold=100.0)
is_blurry, score = detector.is_blurry(img)
```

## Thresholds (defaults)

| Check | Parameter | Default | Meaning |
|---|---|---|---|
| Blur | `threshold` | 100.0 | Laplacian variance below this = blurry |
| Darkness | `threshold` | 50.0 | Mean brightness (0-255) below this = dark |
| Glare | `brightness_cutoff` | 250 | Pixel value counted as "saturated" |
| Glare | `area_threshold` | 0.08 | Fraction of saturated pixels to flag glare |

All thresholds are configurable per-detector or via `QualityAssessor`'s
constructor.

## Validation

Tested against 50 real photographs (read-only, borrowed from a separate
local project's dataset for testing purposes — no images are bundled with
or committed to this repository): 46 passed, 4 failed for genuine reasons
(visually confirmed — 1 underexposed photo, 3 photos with real glare).
Initial glare thresholds produced false positives on ordinary sunlit
surfaces; thresholds were tuned and re-validated with zero false positives
on the same test set. Full details in
[app/docs/Decisions.md](app/docs/Decisions.md).

## Setup

```bash
pip install -r requirements.txt
```

For development (tests):

```bash
pip install -r requirements-dev.txt
```

Run tests:

```bash
pytest app/tests/ -v
```


## API (FastAPI)

`app/main.py` exposes `QualityAssessor` over HTTP.

Run locally:

```bash
uvicorn app.main:app --reload --port 8001
```

### Endpoints

- `GET /health` — returns `{"status": "ok"}`.
- `POST /assess` — upload an image, get a quality verdict back.

  Request: multipart form upload, field name `file`.

  Response:
  ```json
  {
    "filename": "photo.jpg",
    "passed": true,
    "blur": {"flagged": false, "score": 4963.9},
    "darkness": {"flagged": false, "score": 147.6},
    "glare": {"flagged": false, "score": 0.0409}
  }
  ```

Interactive docs (Swagger UI) at `http://127.0.0.1:8001/docs`.


## Docker

Build and run:

```bash
docker compose up --build
```

API available at `http://127.0.0.1:8001` (host port 8001 maps to
container port 8000, to avoid clashing with other local services).

## Project layout

```text
app/
  quality/    BlurDetector, DarknessDetector, GlareDetector, QualityAssessor
  core/       logging_config
  tests/      unit tests for all detectors
  docs/       architecture decisions
  schemas/    (reserved for future API request/response models)
  routers/    (reserved for future FastAPI endpoints)
```

## Related notes

- [app/docs/Decisions.md](app/docs/Decisions.md) records the architecture
  decisions behind this project, including the classical-CV-vs-deep-learning
  trade-off and a real threshold-calibration bug found during testing.
