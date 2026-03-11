#!/usr/bin/env python3
"""Example: Batch process a folder of card images.

Scans a directory for card images, crops each one, and generates
a CSV report with centering and surface scores.

Usage:
    python batch_process.py path/to/images/ --output report.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2

from tcgai_toolkit import CardCropper, CenteringAnalyzer, SurfaceAnalyzer

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def find_images(directory: Path):
    """Recursively find image files in a directory."""
    for ext in IMAGE_EXTENSIONS:
        yield from directory.rglob(f"*{ext}")


def process_card(path: Path, cropper, centering, surface):
    """Process a single card image and return a result dict."""
    try:
        card = cropper.crop(str(path))
    except RuntimeError:
        card = cv2.imread(str(path))
        if card is None:
            return None

    c_result = centering.analyze(card)
    s_report = surface.analyze(card)

    return {
        "file": path.name,
        "lr_ratio": f"{c_result.lr_ratio:.1f}/{100 - c_result.lr_ratio:.1f}",
        "tb_ratio": f"{c_result.tb_ratio:.1f}/{100 - c_result.tb_ratio:.1f}",
        "centering_grade": c_result.grade,
        "is_gem_mint": c_result.is_gem_mint,
        "defect_count": len(s_report.defects),
        "scratch_count": sum(
            1 for d in s_report.defects if d.kind == "scratch"
        ),
        "stain_count": sum(
            1 for d in s_report.defects if d.kind == "stain"
        ),
        "overall_score": s_report.overall_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze TCG card images"
    )
    parser.add_argument("directory", type=Path, help="Image directory")
    parser.add_argument(
        "--output", "-o", type=Path, default=Path("report.csv"),
        help="Output CSV path (default: report.csv)",
    )
    parser.add_argument(
        "--sensitivity", "-s", type=float, default=0.5,
        help="Surface detection sensitivity 0-1 (default: 0.5)",
    )
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Error: {args.directory} is not a directory")
        sys.exit(1)

    images = sorted(find_images(args.directory))
    if not images:
        print(f"No images found in {args.directory}")
        sys.exit(1)

    print(f"Found {len(images)} images in {args.directory}")

    cropper = CardCropper()
    centering = CenteringAnalyzer()
    surface = SurfaceAnalyzer(sensitivity=args.sensitivity)

    results = []
    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] {img_path.name}...", end=" ")
        row = process_card(img_path, cropper, centering, surface)
        if row:
            results.append(row)
            print(f"centering={row['centering_grade']}, "
                  f"defects={row['defect_count']}")
        else:
            print("SKIPPED (unreadable)")

    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nReport saved to {args.output}")
        print(f"Processed {len(results)}/{len(images)} cards")

        gem_count = sum(1 for r in results if r["is_gem_mint"])
        print(f"Gem Mint centering: {gem_count}/{len(results)} cards")
    else:
        print("No cards could be processed.")


if __name__ == "__main__":
    main()