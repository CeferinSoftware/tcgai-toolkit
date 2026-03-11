#!/usr/bin/env python3
"""Example: Full card analysis pipeline.

Demonstrates how to use the tcgai-toolkit to crop a card from a photo,
analyze its centering, and check for surface defects.

Usage:
    python analyze_card.py path/to/card_photo.jpg
"""

import sys
from pathlib import Path

import cv2

from tcgai_toolkit import CardCropper, CenteringAnalyzer, SurfaceAnalyzer


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_card.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    print(f"Analyzing: {image_path.name}")
    print("=" * 50)

    # Step 1: Crop the card from the photo
    cropper = CardCropper()
    try:
        card = cropper.crop(str(image_path))
        print("[OK] Card detected and cropped")
    except RuntimeError as e:
        print(f"[WARN] {e}")
        print("       Proceeding with raw image...")
        card = cv2.imread(str(image_path))

    # Step 2: Analyze centering
    analyzer = CenteringAnalyzer(border_method="gradient")
    result = analyzer.analyze(card)
    print(f"\n{result.summary()}")
    print(f"     Left: {result.left_px}px  |  Right: {result.right_px}px")
    print(f"     Top:  {result.top_px}px   |  Bottom: {result.bottom_px}px")

    # Step 3: Surface defect check
    surface = SurfaceAnalyzer(sensitivity=0.5)
    report = surface.analyze(card)
    print(f"\n{report.summary()}")

    for i, defect in enumerate(report.defects[:5], 1):
        print(
            f"  #{i} {defect.kind}: "
            f"({defect.x}, {defect.y}) "
            f"severity={defect.severity:.2f}"
        )

    # Step 4: Save outputs
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    stem = image_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_cropped.jpg"), card)

    _, overlay = analyzer.analyze_with_overlay(card)
    cv2.imwrite(str(output_dir / f"{stem}_centering.jpg"), overlay)

    heatmap = surface.generate_heatmap(card)
    cv2.imwrite(str(output_dir / f"{stem}_heatmap.jpg"), heatmap)

    print(f"\nOutput saved to {output_dir}/")


if __name__ == "__main__":
    main()