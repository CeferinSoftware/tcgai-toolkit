# Centering Analysis Guide

Card centering is one of the most critical factors in professional grading.
A perfectly centered card has equal borders on all four sides.

## How It Works

The `CenteringAnalyzer` measures the border widths on all four sides of a card
image and computes a left-right (LR) and top-bottom (TB) centering ratio.

### Border Detection Methods

**Gradient** (default): Uses Sobel edge detection to find the boundary between
the card border and the artwork area. More robust with textured or colored borders.

```python
from tcgai_toolkit import CenteringAnalyzer

analyzer = CenteringAnalyzer(border_method="gradient")
result = analyzer.analyze("card.jpg")
```

**Threshold**: Binarizes the image and finds the artwork rectangle via contour
detection. Works best with high-contrast borders (white, yellow).

```python
analyzer = CenteringAnalyzer(border_method="threshold")
```

## Understanding the Results

The `CenteringResult` provides:

| Field | Description |
|-------|-------------|
| `lr_ratio` | Left percentage of horizontal centering (50.0 = perfect) |
| `tb_ratio` | Top percentage of vertical centering (50.0 = perfect) |
| `left_px` | Left border width in pixels |
| `right_px` | Right border width in pixels |
| `top_px` | Top border width in pixels |
| `bottom_px` | Bottom border width in pixels |
| `grade` | Human-readable grade: Gem Mint, Mint, Near Mint, Off-Center |
| `is_gem_mint` | Whether centering qualifies for PSA Gem Mint 10 |

### Centering Notation

The industry-standard notation expresses centering as `left/right` and `top/bottom`:

- **50/50 - 50/50**: Perfect centering
- **55/45 - 50/50**: Slightly off left-right, perfect top-bottom
- **60/40 - 55/45**: Noticeable shift, may still grade well

### Grade Thresholds

| Grade | LR Range | TB Range |
|-------|----------|----------|
| Gem Mint | 45-55 | 45-55 |
| Mint | 40-60 | 40-60 |
| Near Mint | 35-65 | 35-65 |
| Off-Center | < 35 or > 65 | < 35 or > 65 |

## Visual Overlay

Generate an annotated image showing border measurements:

```python
result, overlay = analyzer.analyze_with_overlay("card.jpg")
cv2.imwrite("centering_overlay.jpg", overlay)
```

The overlay draws colored lines at the detected borders with pixel measurements.

## Tips for Best Results

1. **Scan or photograph the card straight-on** — perspective skew reduces accuracy
2. **Crop tight to the card edges** — excess background confuses border detection
3. **Use consistent lighting** — shadows on borders affect threshold detection
4. **Resolution matters** — at least 300 DPI scan or 1000px wide photo recommended