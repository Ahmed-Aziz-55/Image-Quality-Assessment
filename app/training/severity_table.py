"""
app/training/severity_table.py

Finalized severity ranges (Mild / Medium / High) for each of the 7
injectable defect types, validated via the pilot visual-review
experiment (see Decisions.md). Poor Framing is deliberately excluded --
it has no injection path and is never part of ground-truth labeling.

Units match degradation_engine.py's apply_* functions directly:
  blur         -- Gaussian sigma
  darkness     -- % brightness reduction
  overexposure -- % brightness increase
  motion       -- kernel length (px)
  occlusion    -- patch area (%)
  resolution   -- downscale factor (note: LOWER value = MORE severe,
                  opposite direction from the others -- Mild is the
                  highest-numbered range here)
  glare        -- bright patch area (%). Numeric ranges assumed equal
                  to occlusion's (both are area-% based) since the
                  original design only specified qualitative
                  small/medium/large for glare -- flagged as an
                  assumption pending its own dedicated pilot pass.

Pilot review status (see Decisions.md for full methodology writeup):
  blur    -- itemized breakpoints confirmed via visual review
  others  -- qualitatively confirmed ("looked similar" pass), not
             individually itemized; values carried forward from the
             original design table
"""

# (low, high) inclusive-ish range to sample a random value from, per tier.
SEVERITY_RANGES = {
    "blur": {"mild": (1.0, 2.0), "medium": (3.0, 4.0), "high": (5.0, 10.0)},
    "darkness": {"mild": (10.0, 25.0), "medium": (25.0, 45.0), "high": (45.0, 80.0)},
    "overexposure": {"mild": (10.0, 25.0), "medium": (25.0, 45.0), "high": (45.0, 80.0)},
    "motion": {"mild": (3.0, 7.0), "medium": (8.0, 15.0), "high": (16.0, 25.0)},
    "occlusion": {"mild": (2.0, 8.0), "medium": (8.0, 15.0), "high": (15.0, 30.0)},
    # resolution: downscale factor -- Mild is LEAST severe = closest to 1.0
    "resolution": {"mild": (0.7, 0.9), "medium": (0.4, 0.7), "high": (0.1, 0.4)},
    "glare": {"mild": (2.0, 8.0), "medium": (8.0, 15.0), "high": (15.0, 30.0)},
}

INJECTABLE_DEFECTS = list(SEVERITY_RANGES.keys())
SEVERITY_TIERS = ("mild", "medium", "high")