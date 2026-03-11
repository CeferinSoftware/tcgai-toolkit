# tcgai-toolkit

Open-source Python toolkit for analyzing trading card images using computer vision.

Measure card centering, detect surface defects, auto-crop cards from photos, and compare scans — all with a clean Python API built on OpenCV.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## Features

- **Centering Analysis** — Compute left/right and top/bottom centering ratios with PSA-style grading (Gem Mint, Mint, Near Mint, Off-Center)
- **Surface Defect Detection** — Find scratches, stains, and print lines with configurable sensitivity
- **Auto Crop** — Detect and extract cards from photos with perspective correction
- **Card Comparison** — Compare two scans using SSIM and pixel-level diff overlays
- **Batch Processing** — Analyze entire folders and export CSV reports

## Installation

```bash
pip install git+https://github.com/CeferinSoftware/tcgai-toolkit.git
```

Or clone and install locally:

```bash
git clone https://github.com/CeferinSoftware/tcgai-toolkit.git
cd tcgai-toolkit
pip install -e .
```

### Requirements

- Python 3.9+
- OpenCV 4.8+
- NumPy 1.24+
- Pillow 10.0+
- scikit-image 0.21+

## Quick Start

### Analyze Card Centering

```python
from tcgai_toolkit import CenteringAnalyzer

analyzer = CenteringAnalyzer(border_method="gradient")
result = analyzer.analyze("card_scan.jpg")

print(f"Centering: {result.lr_ratio:.1f}/{100-result.lr_ratio:.1f} "
      f"- {result.tb_ratio:.1f}/{100-result.tb_ratio:.1f}")
print(f"Grade: {result.grade}")
print(f"Gem Mint eligible: {result.is_gem_mint}")
```

### Detect Surface Defects

```python
from tcgai_toolkit import SurfaceAnalyzer

surface = SurfaceAnalyzer(sensitivity=0.5)
report = surface.analyze("card_scan.jpg")

print(f"Found {len(report.defects)} defects")
print(f"Surface score: {report.overall_score:.2f}")

for defect in report.defects:
    print(f"  {defect.kind} at ({defect.x},{defect.y}) "
          f"severity={defect.severity:.2f}")
```

### Auto-Crop Card from Photo

```python
from tcgai_toolkit import CardCropper

cropper = CardCropper(target_size=(750, 1050))
card = cropper.crop("photo_of_card.jpg")

# Multiple cards in one photo
cards = cropper.crop_all("binder_page.jpg")
print(f"Found {len(cards)} cards")
```

### Compare Two Scans

```python
from tcgai_toolkit import CardComparator
import cv2

img_a = cv2.imread("card_front.jpg")
img_b = cv2.imread("card_ref.jpg")

comparator = CardComparator()
result = comparator.compare(img_a, img_b)

print(f"SSIM: {result.ssim:.4f}")
print(f"Match: {result.is_match}")
```

## Examples

See the [`examples/`](examples/) directory for complete scripts:

| Script | Description |
|--------|-------------|
| [`analyze_card.py`](examples/analyze_card.py) | Full pipeline: crop → centering → surface analysis |
| [`batch_process.py`](examples/batch_process.py) | Process a folder of card images and export CSV |
| [`compare_cards.py`](examples/compare_cards.py) | Side-by-side comparison of two card scans |

```bash
# Run the full analysis pipeline
python examples/analyze_card.py path/to/card.jpg

# Batch process a folder
python examples/batch_process.py ./my_cards/ --output report.csv
```

## API Reference

### `CenteringAnalyzer`

| Method | Returns | Description |
|--------|---------|-------------|
| `analyze(image)` | `CenteringResult` | Compute centering ratios |
| `analyze_with_overlay(image)` | `(CenteringResult, ndarray)` | Centering + annotated image |

**CenteringResult fields:** `lr_ratio`, `tb_ratio`, `left_px`, `right_px`, `top_px`, `bottom_px`, `grade`, `is_gem_mint`

### `SurfaceAnalyzer`

| Method | Returns | Description |
|--------|---------|-------------|
| `analyze(image)` | `SurfaceReport` | Detect surface defects |
| `generate_heatmap(image)` | `ndarray` | Defect heatmap overlay |

**SurfaceReport fields:** `defects` (list of Defect), `overall_score` (0-1)

### `CardCropper`

| Method | Returns | Description |
|--------|---------|-------------|
| `crop(image)` | `ndarray` | Extract single card with perspective correction |
| `crop_all(image)` | `list[ndarray]` | Extract all cards from a multi-card photo |

### `CardComparator`

| Method | Returns | Description |
|--------|---------|-------------|
| `compare(img_a, img_b)` | `ComparisonResult` | SSIM + pixel diff metrics |
| `visual_diff(img_a, img_b)` | `ndarray` | Color-coded diff overlay |

## Documentation

- [Centering Analysis Guide](docs/centering.md) — Understanding centering ratios and grading thresholds
- [Surface Defect Detection](docs/surface-analysis.md) — Configuring sensitivity and interpreting results

## Project Structure

```
tcgai-toolkit/
├── tcgai_toolkit/
│   ├── __init__.py        # Package exports
│   ├── centering.py       # Centering analysis
│   ├── crop.py            # Auto-cropping
│   ├── surface.py         # Surface defect detection
│   ├── compare.py         # Card comparison
│   └── utils.py           # Shared utilities
├── examples/              # Usage examples
├── tests/                 # Unit tests
├── docs/                  # Guides
├── setup.py
├── requirements.txt
└── LICENSE
```

## Running Tests

```bash
pip install -e .
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.

---

Built by the team at [TCG AI PRO](https://tcgai.pro) — AI-powered trading card grading.