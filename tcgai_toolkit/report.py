"""HTML grading report generator.

Produces a self-contained HTML file with card images, centering data,
surface-defect analysis, and an overall summary — ready to open in any
browser or attach to an email.
"""

import base64
import datetime
import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from .centering import CenteringAnalyzer, CenteringResult
from .crop import CardCropper
from .surface import SurfaceAnalyzer, SurfaceReport


def _img_to_data_uri(img: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode a BGR image as a base64 data-URI for embedding in HTML."""
    ok, buf = cv2.imencode(fmt, img)
    if not ok:
        return ""
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _grade_color(grade: str) -> str:
    """Return a CSS colour for a centering grade label."""
    mapping = {
        "Gem Mint": "#00c853",
        "Mint": "#64dd17",
        "Near Mint": "#aeea00",
        "Excellent": "#ffd600",
        "Off-Center": "#ff6d00",
        "Miscut": "#d50000",
    }
    return mapping.get(grade, "#757575")


def _score_color(score: float) -> str:
    """Return a CSS colour for a 0-1 surface score."""
    if score >= 0.9:
        return "#00c853"
    if score >= 0.7:
        return "#64dd17"
    if score >= 0.5:
        return "#ffd600"
    if score >= 0.3:
        return "#ff6d00"
    return "#d50000"


def generate_report(
    source: Union[str, "Path", np.ndarray],
    output_path: Optional[str] = None,
    *,
    centering_method: str = "gradient",
    surface_sensitivity: float = 0.5,
    open_browser: bool = True,
) -> str:
    """Analyse a card image and write a full HTML grading report.

    Parameters
    ----------
    source : str, Path, or np.ndarray
        Card photograph (full photo or pre-cropped card).
    output_path : str, optional
        Where to save the HTML file.  Defaults to
        ``<source_stem>_report.html`` next to the source image, or
        ``card_report.html`` in the current directory when *source* is
        an ndarray.
    centering_method : str
        ``'gradient'`` or ``'threshold'``.
    surface_sensitivity : float
        0-1 sensitivity for the surface analyser.
    open_browser : bool
        If *True*, open the report in the default browser after writing.

    Returns
    -------
    str
        Absolute path of the generated HTML file.
    """
    # --- Load & crop --------------------------------------------------
    if isinstance(source, (str, Path)):
        original = cv2.imread(str(source))
        if original is None:
            raise FileNotFoundError(f"Cannot read image: {source}")
        source_name = Path(source).stem
    else:
        original = source
        source_name = "card"

    oh, ow = original.shape[:2]

    cropper = CardCropper()
    try:
        card = cropper.crop(original)
        cropped = True
    except RuntimeError:
        card = original
        cropped = False

    ch, cw = card.shape[:2]

    # --- Centering ----------------------------------------------------
    center_grad = CenteringAnalyzer(border_method="gradient")
    center_thresh = CenteringAnalyzer(border_method="threshold")

    res_grad = center_grad.analyze(card)
    res_thresh = center_thresh.analyze(card)

    # Pick primary result based on user preference
    primary_center: CenteringResult = (
        res_grad if centering_method == "gradient" else res_thresh
    )

    _, overlay = center_grad.analyze_with_overlay(card)

    # --- Surface ------------------------------------------------------
    surface = SurfaceAnalyzer(sensitivity=surface_sensitivity)
    surf_report: SurfaceReport = surface.analyze(card)
    heatmap = surface.generate_heatmap(card)

    # --- Encode images ------------------------------------------------
    uri_original = _img_to_data_uri(
        cv2.resize(original, (0, 0), fx=0.35, fy=0.35)
    )
    uri_card = _img_to_data_uri(card)
    uri_overlay = _img_to_data_uri(overlay)
    uri_heatmap = _img_to_data_uri(heatmap)

    # --- Build HTML ---------------------------------------------------
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    defect_rows = ""
    for i, d in enumerate(surf_report.defects[:15], 1):
        sev_pct = int(d.severity * 100)
        defect_rows += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td><span class='badge {d.kind}'>{d.kind}</span></td>"
            f"<td>({d.x}, {d.y})</td>"
            f"<td>{d.w} × {d.h}</td>"
            f"<td><div class='bar'><div class='fill' style='width:{sev_pct}%'></div></div> {d.severity:.2f}</td>"
            f"</tr>"
        )

    if not surf_report.defects:
        defect_rows = (
            "<tr><td colspan='5' style='text-align:center;color:#888'>"
            "No defects detected</td></tr>"
        )

    # Determine overall assessment
    g = primary_center.grade
    s = surf_report.overall_score
    if g in ("Gem Mint", "Mint") and s >= 0.8:
        overall = "Excellent Condition"
        overall_color = "#00c853"
    elif g in ("Gem Mint", "Mint", "Near Mint") and s >= 0.5:
        overall = "Good Condition"
        overall_color = "#64dd17"
    elif g in ("Excellent", "Near Mint") and s >= 0.3:
        overall = "Fair Condition"
        overall_color = "#ffd600"
    else:
        overall = "Needs Review"
        overall_color = "#ff6d00"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Card Grading Report — {source_name}</title>
<style>
  :root {{ --accent: #1565c0; --bg: #f5f5f5; --card: #fff; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,sans-serif; background:var(--bg); color:#222; }}
  .container {{ max-width:960px; margin:0 auto; padding:24px 16px; }}
  header {{ background:var(--accent); color:#fff; padding:20px 24px; border-radius:12px 12px 0 0; display:flex; justify-content:space-between; align-items:center; }}
  header h1 {{ font-size:22px; font-weight:600; }}
  header .date {{ font-size:13px; opacity:.85; }}
  .overall-banner {{ background:{overall_color}; color:#fff; text-align:center; padding:14px; font-size:20px; font-weight:700; letter-spacing:1px; }}
  section {{ background:var(--card); padding:24px; border:1px solid #e0e0e0; margin-bottom:0; }}
  section:last-child {{ border-radius:0 0 12px 12px; }}
  h2 {{ font-size:17px; color:var(--accent); margin-bottom:16px; border-bottom:2px solid var(--accent); padding-bottom:6px; }}
  .grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .grid img {{ width:100%; border-radius:8px; border:1px solid #ddd; }}
  .metric {{ display:flex; justify-content:space-between; padding:8px 0; border-bottom:1px solid #eee; }}
  .metric:last-child {{ border-bottom:none; }}
  .metric .label {{ color:#555; font-size:14px; }}
  .metric .value {{ font-weight:600; font-size:14px; }}
  .grade {{ display:inline-block; padding:3px 12px; border-radius:20px; color:#fff; font-weight:600; font-size:13px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:10px; }}
  th {{ text-align:left; padding:8px 6px; background:#f0f4f8; font-weight:600; color:#555; border-bottom:2px solid #ddd; }}
  td {{ padding:7px 6px; border-bottom:1px solid #eee; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; color:#fff; }}
  .badge.scratch {{ background:#ff9800; }}
  .badge.stain {{ background:#8e24aa; }}
  .badge.print_line {{ background:#1e88e5; }}
  .badge.dent {{ background:#e53935; }}
  .bar {{ display:inline-block; width:80px; height:10px; background:#e0e0e0; border-radius:5px; vertical-align:middle; margin-right:6px; }}
  .bar .fill {{ height:100%; background:#ff5722; border-radius:5px; }}
  .footer {{ text-align:center; padding:16px; font-size:12px; color:#999; }}
  .footer a {{ color:var(--accent); text-decoration:none; }}
  @media (max-width:600px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<div class="container">

<header>
  <h1>TCG Card Grading Report</h1>
  <span class="date">{now}</span>
</header>

<div class="overall-banner">{overall}</div>

<section>
  <h2>📷 Card Detection</h2>
  <div class="grid">
    <div>
      <p style="font-size:13px;color:#888;margin-bottom:6px">Original Photo ({ow}×{oh})</p>
      <img src="{uri_original}" alt="Original">
    </div>
    <div>
      <p style="font-size:13px;color:#888;margin-bottom:6px">Cropped Card ({cw}×{ch})</p>
      <img src="{uri_card}" alt="Cropped">
    </div>
  </div>
  <div class="metric" style="margin-top:14px">
    <span class="label">Auto-crop</span>
    <span class="value">{"✅ Card detected & perspective-corrected" if cropped else "⚠️ No card boundary found — using full image"}</span>
  </div>
</section>

<section>
  <h2>📐 Centering Analysis</h2>
  <div class="grid">
    <div>
      <img src="{uri_overlay}" alt="Centering overlay">
    </div>
    <div>
      <div class="metric">
        <span class="label">Grade</span>
        <span class="value"><span class="grade" style="background:{_grade_color(res_grad.grade)}">{res_grad.grade}</span></span>
      </div>
      <div class="metric">
        <span class="label">Left / Right</span>
        <span class="value">{res_grad.lr_ratio:.1f} / {100-res_grad.lr_ratio:.1f}</span>
      </div>
      <div class="metric">
        <span class="label">Top / Bottom</span>
        <span class="value">{res_grad.tb_ratio:.1f} / {100-res_grad.tb_ratio:.1f}</span>
      </div>
      <div class="metric">
        <span class="label">Left border</span>
        <span class="value">{res_grad.left_px} px</span>
      </div>
      <div class="metric">
        <span class="label">Right border</span>
        <span class="value">{res_grad.right_px} px</span>
      </div>
      <div class="metric">
        <span class="label">Top border</span>
        <span class="value">{res_grad.top_px} px</span>
      </div>
      <div class="metric">
        <span class="label">Bottom border</span>
        <span class="value">{res_grad.bottom_px} px</span>
      </div>
      <div class="metric">
        <span class="label">Gem Mint?</span>
        <span class="value">{"✅ Yes" if res_grad.is_gem_mint else "❌ No"}</span>
      </div>

      <p style="margin-top:14px;font-size:12px;color:#999">
        Threshold method: LR {res_thresh.lr_ratio:.1f}/{100-res_thresh.lr_ratio:.1f},
        TB {res_thresh.tb_ratio:.1f}/{100-res_thresh.tb_ratio:.1f}
        — {res_thresh.grade}
      </p>
    </div>
  </div>
</section>

<section>
  <h2>🔍 Surface Analysis</h2>
  <div class="grid">
    <div>
      <img src="{uri_heatmap}" alt="Defect heatmap">
    </div>
    <div>
      <div class="metric">
        <span class="label">Overall Score</span>
        <span class="value"><span class="grade" style="background:{_score_color(surf_report.overall_score)}">{surf_report.overall_score:.2f}</span></span>
      </div>
      <div class="metric">
        <span class="label">Total Defects</span>
        <span class="value">{len(surf_report.defects)}</span>
      </div>
      <div class="metric">
        <span class="label">Scratches</span>
        <span class="value">{sum(1 for d in surf_report.defects if d.kind == 'scratch')}</span>
      </div>
      <div class="metric">
        <span class="label">Stains</span>
        <span class="value">{sum(1 for d in surf_report.defects if d.kind == 'stain')}</span>
      </div>
      <div class="metric">
        <span class="label">Print Lines</span>
        <span class="value">{sum(1 for d in surf_report.defects if d.kind == 'print_line')}</span>
      </div>
      <div class="metric">
        <span class="label">Sensitivity</span>
        <span class="value">{surface_sensitivity}</span>
      </div>
    </div>
  </div>

  <table>
    <thead><tr><th>#</th><th>Type</th><th>Position</th><th>Size</th><th>Severity</th></tr></thead>
    <tbody>{defect_rows}</tbody>
  </table>
  {f'<p style="margin-top:8px;font-size:12px;color:#999">Showing top 15 of {len(surf_report.defects)} defects</p>' if len(surf_report.defects) > 15 else ''}
</section>

<div class="footer">
  Generated by <a href="https://tcgai.pro">tcgai-toolkit</a> · Powered by OpenCV
</div>

</div>
</body>
</html>"""

    # --- Write --------------------------------------------------------
    if output_path is None:
        output_path = f"{source_name}_report.html"

    out = Path(output_path).resolve()
    out.write_text(html, encoding="utf-8")

    if open_browser:
        import webbrowser
        webbrowser.open(str(out))

    return str(out)
