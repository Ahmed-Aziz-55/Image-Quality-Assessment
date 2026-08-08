"""
app/pilot/generate_severity_ladders.py

Pilot-experiment helper: applies each of the 7 injectable defect types
at every severity value in its ladder (see degradation_engine.py) to
every sample image in an input folder, saving each result as its own
file so you can open them in a file browser / image viewer and eyeball
where Mild -> Medium -> High actually breaks.

Usage:
    python -m app.pilot.generate_severity_ladders \\
        --input-dir path/to/sample_images \\
        --output-dir path/to/pilot_output

Output layout:
    pilot_output/
      blur/
        photo1_blur_sigma0.5.jpg
        photo1_blur_sigma1.jpg
        ...
      darkness/
        photo1_darkness_pct10.jpg
        ...
      ...

Each defect gets its own subfolder; files are numbered/named so that
sorting alphabetically also sorts by ascending severity, making it easy
to flip through them in order in any image viewer.
"""

import argparse
from pathlib import Path

import cv2

from app.pilot.degradation_engine import DEGRADATION_LADDERS

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _format_value(value: float) -> str:
    """Zero-pads and formats a severity value so filenames sort correctly
    alphabetically in ascending severity order (e.g. 2 -> '02', 0.5 -> '00.5')."""
    if isinstance(value, float) and not value.is_integer():
        return f"{value:05.1f}"
    return f"{int(value):03d}"


def generate_ladders(input_dir: Path, output_dir: Path) -> None:
    image_paths = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} sample image(s). Generating ladders for "
          f"{len(DEGRADATION_LADDERS)} defect types...")

    for defect_name, (apply_fn, severity_values) in DEGRADATION_LADDERS.items():
        defect_dir = output_dir / defect_name
        defect_dir.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"  [skip] could not load {image_path}")
                continue

            stem = image_path.stem
            for value in severity_values:
                degraded = apply_fn(image, value)
                value_str = _format_value(value)
                out_name = f"{stem}_{defect_name}_{value_str}.jpg"
                cv2.imwrite(str(defect_dir / out_name), degraded)

        print(f"  [{defect_name}] {len(severity_values)} severity levels x "
              f"{len(image_paths)} images -> {defect_dir}")

    print(f"\nDone. Open each subfolder of {output_dir} in your image viewer, "
          f"sort by name, and flip through in order to find where Mild -> "
          f"Medium -> High actually breaks for that defect.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                         help="Folder of clean sample images (jpg/png/bmp)")
    parser.add_argument("--output-dir", required=True, type=Path,
                         help="Folder to write severity-ladder images into")
    args = parser.parse_args()

    generate_ladders(args.input_dir, args.output_dir)