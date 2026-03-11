#!/usr/bin/env python3
"""Example: Compare two card scans side by side.

Useful for checking card condition changes over time (before/after),
or verifying that a card matches a reference image.

Usage:
    python compare_cards.py card_a.jpg card_b.jpg
"""

import sys
from pathlib import Path

import cv2

from tcgai_toolkit import CardComparator


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_cards.py <image_a> <image_b>")
        sys.exit(1)

    path_a, path_b = Path(sys.argv[1]), Path(sys.argv[2])

    for p in (path_a, path_b):
        if not p.exists():
            print(f"Error: File not found: {p}")
            sys.exit(1)

    img_a = cv2.imread(str(path_a))
    img_b = cv2.imread(str(path_b))

    if img_a is None or img_b is None:
        print("Error: Could not read one or both images.")
        sys.exit(1)

    comparator = CardComparator()
    result = comparator.compare(img_a, img_b)

    print(f"Comparing: {path_a.name} vs {path_b.name}")
    print("=" * 50)
    print(f"SSIM Score:      {result.ssim:.4f}  (1.0 = identical)")
    print(f"Pixel Diff:      {result.pixel_diff_pct:.2f}%")
    print(f"Match:           {'YES' if result.is_match else 'NO'}")
    print(f"Diff regions:    {result.diff_region_count}")

    if result.ssim > 0.95:
        print("\nVerdict: Cards are virtually identical.")
    elif result.ssim > 0.85:
        print("\nVerdict: Cards are very similar with minor differences.")
    elif result.ssim > 0.70:
        print("\nVerdict: Noticeable differences detected.")
    else:
        print("\nVerdict: Cards are significantly different.")

    # Save visual diff
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    diff_img = comparator.visual_diff(img_a, img_b)
    out_path = output_dir / f"diff_{path_a.stem}_vs_{path_b.stem}.jpg"
    cv2.imwrite(str(out_path), diff_img)
    print(f"\nDiff overlay saved to {out_path}")


if __name__ == "__main__":
    main()