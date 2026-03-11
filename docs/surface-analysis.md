# Surface Defect Detection Guide

Surface condition is a key factor in card grading. The `SurfaceAnalyzer`
uses computer vision techniques to detect scratches, stains, and print lines
on trading card surfaces.

## Defect Types

### Scratches
Thin linear marks on the card surface. Detected via morphological line
filtering and edge analysis. Common on holographic and foil cards.

### Stains
Discolored patches that differ from the surrounding area. Detected using
local color deviation analysis. Includes water damage, oil marks, and
fingerprint residue.

### Print Lines
Manufacturing defects that appear as faint horizontal or vertical lines
across the card face. Detected using frequency domain analysis (FFT)
combined with line detection.

## Basic Usage

```python
from tcgai_toolkit import SurfaceAnalyzer

analyzer = SurfaceAnalyzer(sensitivity=0.5)
report = analyzer.analyze("card_scan.jpg")

print(f"Found {len(report.defects)} defects")
print(f"Overall score: {report.overall_score:.2f}")

for defect in report.defects:
    print(f"  {defect.kind} at ({defect.x}, {defect.y}) "
          f"severity={defect.severity:.2f}")
```

## Sensitivity

The `sensitivity` parameter (0.0 to 1.0) controls detection thresholds:

| Value | Behavior |
|-------|----------|
| 0.1 - 0.3 | Only major, obvious defects |
| 0.4 - 0.6 | Balanced detection (recommended) |
| 0.7 - 0.9 | Catches subtle marks, may include false positives |

```python
# Conservative — only clear defects
analyzer = SurfaceAnalyzer(sensitivity=0.3)

# Aggressive — find everything
analyzer = SurfaceAnalyzer(sensitivity=0.8)
```

## Heatmap Visualization

Generate a heatmap overlay showing areas of detected anomalies:

```python
heatmap = analyzer.generate_heatmap("card_scan.jpg")
cv2.imwrite("defect_heatmap.jpg", heatmap)
```

Red/yellow areas indicate detected defects. Blue/green areas are clean.

## SurfaceReport Fields

| Field | Type | Description |
|-------|------|-------------|
| `defects` | `list[Defect]` | All detected defects |
| `overall_score` | `float` | 0.0 (worst) to 1.0 (pristine) |

## Defect Fields

| Field | Type | Description |
|-------|------|-------------|
| `kind` | `str` | "scratch", "stain", or "print_line" |
| `x`, `y` | `int` | Top-left position of bounding box |
| `w`, `h` | `int` | Width and height of bounding box |
| `severity` | `float` | 0.0 (minor) to 1.0 (severe) |

## Best Practices

1. **Use high-resolution scans** — 600 DPI or higher captures fine scratches
2. **Scan both sides** — surface defects can appear on front or back
3. **Control lighting** — avoid glare on foil/holographic cards
4. **Calibrate sensitivity** — start at 0.5 and adjust based on your card type
5. **Check both raw and sleeved** — sleeves can hide or introduce artifacts