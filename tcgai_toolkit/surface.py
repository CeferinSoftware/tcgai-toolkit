"""Surface defect detection for trading cards.

Identifies scratches, print lines, and surface damage using image
processing techniques.  Useful as a quick pre-screen before sending
cards for professional grading.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from .utils import load_image, resize_for_processing


@dataclass
class Defect:
    """A single detected surface defect.

    Parameters
    ----------
    kind : str
        Type of defect: ``"scratch"``, ``"print_line"``, or ``"stain"``.
    x, y : int
        Top-left corner of the bounding box.
    w, h : int
        Width and height of the bounding box.
    severity : float
        Severity score from 0.0 (minor) to 1.0 (severe).
    """

    kind: str
    x: int
    y: int
    w: int
    h: int
    severity: float

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class SurfaceReport:
    """Aggregated surface analysis results.

    Parameters
    ----------
    defects : list of Defect
        All detected defects.
    overall_score : float
        Surface quality from 0.0 (heavily damaged) to 1.0 (pristine).
    """

    defects: List[Defect] = field(default_factory=list)
    overall_score: float = 1.0

    @property
    def scratch_count(self) -> int:
        return sum(1 for d in self.defects if d.kind == "scratch")

    @property
    def print_line_count(self) -> int:
        return sum(1 for d in self.defects if d.kind == "print_line")

    @property
    def stain_count(self) -> int:
        return sum(1 for d in self.defects if d.kind == "stain")

    def summary(self) -> str:
        return (
            f"Surface: {len(self.defects)} defects found  |  "
            f"Score: {self.overall_score:.2f}  |  "
            f"Scratches: {self.scratch_count}  |  "
            f"Stains: {self.stain_count}"
        )


class SurfaceAnalyzer:
    """Detect surface defects on a trading card image.

    Uses a combination of high-pass filtering and morphological analysis
    to find linear scratches and localised damage.

    Parameters
    ----------
    sensitivity : float
        Detection sensitivity from 0.0 (lenient) to 1.0 (strict).
        Higher values report more potential defects.  Default 0.5.
    """

    def __init__(self, sensitivity: float = 0.5):
        self._sensitivity = float(np.clip(sensitivity, 0.0, 1.0))

    @property
    def sensitivity(self) -> float:
        """Current detection sensitivity."""
        return self._sensitivity

    def analyze(self, source, max_dim: int = 1200) -> SurfaceReport:
        """Run surface analysis on a card image.

        Parameters
        ----------
        source : str, Path, or np.ndarray
            A cropped card image (use ``CardCropper`` first for best results).

        Returns
        -------
        SurfaceReport
        """
        img = load_image(source)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img, _ = resize_for_processing(img, max_dim)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        defects: List[Defect] = []
        defects.extend(self._detect_scratches(gray))
        defects.extend(self._detect_print_lines(gray))
        defects.extend(self._detect_stains(img, gray))

        score = self._compute_score(defects, gray.shape)
        return SurfaceReport(defects=defects, overall_score=score)

    def generate_heatmap(self, source, max_dim: int = 1200) -> np.ndarray:
        """Return a colour heatmap highlighting potential defects.

        Red regions indicate higher defect probability.
        """
        img = load_image(source)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img, _ = resize_for_processing(img, max_dim)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        high_pass = cv2.absdiff(gray, blur)

        normalized = cv2.normalize(
            high_pass, None, 0, 255, cv2.NORM_MINMAX
        )
        heatmap = cv2.applyColorMap(
            normalized.astype(np.uint8), cv2.COLORMAP_JET
        )

        blended = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
        return blended

    # ------------------------------------------------------------------

    def _detect_scratches(self, gray: np.ndarray) -> List[Defect]:
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        high_pass = cv2.absdiff(gray, blur)

        threshold = int(25 - self._sensitivity * 15)
        _, binary = cv2.threshold(
            high_pass, threshold, 255, cv2.THRESH_BINARY
        )

        line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        scratches_v = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel)

        line_kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        scratches_h = cv2.morphologyEx(binary, cv2.MORPH_OPEN, line_kernel_h)

        combined = cv2.bitwise_or(scratches_v, scratches_h)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        defects: List[Defect] = []
        h, w_img = gray.shape
        min_length = max(h, w_img) * 0.02

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            length = max(cw, ch)
            if length < min_length:
                continue
            aspect = max(cw, ch) / max(min(cw, ch), 1)
            if aspect < 3:
                continue
            severity = min(length / (max(h, w_img) * 0.15), 1.0)
            defects.append(Defect("scratch", x, y, cw, ch, round(severity, 2)))

        return defects

    def _detect_print_lines(self, gray: np.ndarray) -> List[Defect]:
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=int(80 - self._sensitivity * 40),
            minLineLength=int(gray.shape[1] * 0.3),
            maxLineGap=5,
        )

        defects: List[Defect] = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                if length < gray.shape[1] * 0.4:
                    continue
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if not (175 < angle or angle < 5 or 85 < angle < 95):
                    continue
                severity = min(length / gray.shape[1], 1.0) * 0.6
                x = min(x1, x2)
                y = min(y1, y2)
                defects.append(
                    Defect(
                        "print_line", x, y,
                        abs(x2 - x1), abs(y2 - y1),
                        round(severity, 2),
                    )
                )

        return defects[:5]

    def _detect_stains(
        self, img: np.ndarray, gray: np.ndarray
    ) -> List[Defect]:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, s_channel, _ = cv2.split(hsv)

        blur_s = cv2.GaussianBlur(s_channel, (25, 25), 0)
        diff = cv2.absdiff(s_channel, blur_s)

        threshold = int(35 - self._sensitivity * 15)
        _, binary = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        defects: List[Defect] = []
        h, w_img = gray.shape
        min_area = h * w_img * 0.001

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            severity = min(area / (h * w_img * 0.01), 1.0)
            defects.append(
                Defect("stain", x, y, cw, ch, round(severity, 2))
            )

        return defects[:10]

    @staticmethod
    def _compute_score(
        defects: List[Defect], shape: Tuple[int, ...]
    ) -> float:
        if not defects:
            return 1.0
        penalty = sum(d.severity * 0.15 for d in defects)
        return round(max(1.0 - penalty, 0.0), 2)