"""Card centering analysis using border detection.

Centering is one of the most important grading sub-categories.  PSA, BGS, and
CGC all evaluate how well the printed image is centred on the card stock.  This
module provides a pure-OpenCV implementation that measures left/right and
top/bottom border ratios.

Typical PSA centering thresholds
--------------------------------
- Gem Mint 10 : 55/45 or better on front, 75/25 or better on back
- Mint 9      : 60/40 or better on front, 90/10 or better on back
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import load_image, resize_for_processing, auto_rotate


@dataclass
class CenteringResult:
    """Container for a centering measurement.

    Parameters
    ----------
    lr_ratio : float
        Left-side percentage of horizontal centering (50.0 = perfect).
    tb_ratio : float
        Top-side percentage of vertical centering (50.0 = perfect).
    left_px, right_px, top_px, bottom_px : int
        Border widths in pixels (after resize).
    """

    lr_ratio: float
    tb_ratio: float
    left_px: int = 0
    right_px: int = 0
    top_px: int = 0
    bottom_px: int = 0

    @property
    def grade(self) -> str:
        """Human-readable centering grade."""
        max_dev = max(abs(self.lr_ratio - 50.0), abs(self.tb_ratio - 50.0))
        if max_dev <= 5:
            return "Gem Mint"
        elif max_dev <= 10:
            return "Mint"
        elif max_dev <= 15:
            return "Near Mint"
        return "Off-Center"

    @property
    def is_gem_mint(self) -> bool:
        """Check against PSA Gem Mint 10 front thresholds (55/45)."""
        return self.grade == "Gem Mint"

    def summary(self) -> str:
        return (
            f"Centering  LR: {self.lr_ratio:.1f}/{100 - self.lr_ratio:.1f}"
            f"  |  TB: {self.tb_ratio:.1f}/{100 - self.tb_ratio:.1f}"
            f"  =>  {self.grade}"
        )


class CenteringAnalyzer:
    """Measure card centering from a photograph.

    The algorithm:
    1. Convert to grayscale and apply adaptive thresholding.
    2. Find the largest contour (the card border).
    3. Approximate the contour to a quadrilateral.
    4. Detect the inner *print area* using edge gradients.
    5. Compute border widths and express as ratios.

    Parameters
    ----------
    border_method : str
        ``"gradient"`` (default) uses Sobel edge detection to find the
        transition between the border and the artwork.  ``"threshold"``
        uses a simpler colour-based approach that works best for cards
        with solid-colour borders (e.g., early Pokemon base set).
    """

    METHODS = ("gradient", "threshold")

    def __init__(self, border_method: str = "gradient"):
        if border_method not in self.METHODS:
            raise ValueError(
                f"border_method must be one of {self.METHODS}, "
                f"got {border_method!r}"
            )
        self._method = border_method

    @property
    def border_method(self) -> str:
        """The active border-detection method."""
        return self._method

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, source, max_dim: int = 1200) -> CenteringResult:
        """Run centering analysis on a card image.

        Parameters
        ----------
        source : str, Path, or np.ndarray
            The card image (file path or array).
        max_dim : int
            Resize longest edge to this value for faster processing.

        Returns
        -------
        CenteringResult
        """
        img = load_image(source)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = auto_rotate(img)
        img, _scale = resize_for_processing(img, max_dim)

        quad = self._find_card_quad(img)
        if quad is None:
            h, w = img.shape[:2]
            quad = np.array(
                [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
            )

        warped = self._perspective_correction(img, quad)
        borders = self._measure_borders(warped)
        return self._borders_to_result(borders, warped.shape)

    def analyze_with_overlay(
        self, source, max_dim: int = 1200
    ) -> Tuple[CenteringResult, np.ndarray]:
        """Like ``analyze`` but also returns a debug overlay image."""
        img = load_image(source)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = auto_rotate(img)
        img, _scale = resize_for_processing(img, max_dim)

        quad = self._find_card_quad(img)
        if quad is None:
            h, w = img.shape[:2]
            quad = np.array(
                [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
            )

        warped = self._perspective_correction(img, quad)
        borders = self._measure_borders(warped)
        result = self._borders_to_result(borders, warped.shape)
        overlay = self._draw_overlay(warped, borders)
        return result, overlay

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_card_quad(self, img: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        img_area = img.shape[0] * img.shape[1]
        if cv2.contourArea(largest) < img_area * 0.3:
            return None

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)

        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect).astype(np.float32)
        return box

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1).ravel()
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def _perspective_correction(
        self, img: np.ndarray, quad: np.ndarray
    ) -> np.ndarray:
        ordered = self._order_points(quad)
        tl, tr, br, bl = ordered
        width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
        width = max(width, 1)
        height = max(height, 1)
        dst = np.array(
            [[0, 0], [width, 0], [width, height], [0, height]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(img, M, (width, height))

    def _measure_borders(self, warped: np.ndarray) -> dict:
        if self._method == "gradient":
            return self._measure_borders_gradient(warped)
        return self._measure_borders_threshold(warped)

    @staticmethod
    def _measure_borders_gradient(warped: np.ndarray) -> dict:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        sobel_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

        margin = int(min(h, w) * 0.05)
        search = int(min(h, w) * 0.25)
        search = max(search, 1)

        safe_m = min(margin, h // 3, w // 3)

        # Left border
        col_energy = sobel_x[
            safe_m : h - safe_m, safe_m : safe_m + search
        ].mean(axis=0)
        left = safe_m + int(np.argmax(col_energy)) if len(col_energy) else safe_m

        # Right border
        col_energy_r = sobel_x[
            safe_m : h - safe_m, w - safe_m - search : w - safe_m
        ].mean(axis=0)
        right = (
            search - int(np.argmax(col_energy_r[::-1]))
            if len(col_energy_r) else safe_m
        )

        # Top border
        row_energy = sobel_y[
            safe_m : safe_m + search, safe_m : w - safe_m
        ].mean(axis=1)
        top = safe_m + int(np.argmax(row_energy)) if len(row_energy) else safe_m

        # Bottom border
        row_energy_b = sobel_y[
            h - safe_m - search : h - safe_m, safe_m : w - safe_m
        ].mean(axis=1)
        bottom = (
            search - int(np.argmax(row_energy_b[::-1]))
            if len(row_energy_b) else safe_m
        )

        return {"left": left, "right": right, "top": top, "bottom": bottom}

    @staticmethod
    def _measure_borders_threshold(warped: np.ndarray) -> dict:
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        mid_row = binary[h // 2, :]
        left = int(np.argmax(mid_row < 128)) if mid_row[0] >= 128 else 0
        right_edge = (
            w - int(np.argmax(mid_row[::-1] < 128))
            if mid_row[-1] >= 128 else w
        )

        mid_col = binary[:, w // 2]
        top = int(np.argmax(mid_col < 128)) if mid_col[0] >= 128 else 0
        bottom_edge = (
            h - int(np.argmax(mid_col[::-1] < 128))
            if mid_col[-1] >= 128 else h
        )

        return {
            "left": left,
            "right": int(w - right_edge),
            "top": top,
            "bottom": int(h - bottom_edge),
        }

    @staticmethod
    def _borders_to_result(
        borders: dict, shape: Tuple[int, ...]
    ) -> CenteringResult:
        l = max(borders["left"], 1)
        r = max(borders["right"], 1)
        t = max(borders["top"], 1)
        b = max(borders["bottom"], 1)
        lr_total = l + r
        tb_total = t + b
        return CenteringResult(
            lr_ratio=round(l / lr_total * 100, 1),
            tb_ratio=round(t / tb_total * 100, 1),
            left_px=l,
            right_px=r,
            top_px=t,
            bottom_px=b,
        )

    @staticmethod
    def _draw_overlay(warped: np.ndarray, borders: dict) -> np.ndarray:
        overlay = warped.copy()
        h, w = overlay.shape[:2]
        l, r = borders["left"], borders["right"]
        t, b = borders["top"], borders["bottom"]

        cv2.line(overlay, (l, 0), (l, h), (0, 255, 0), 2)
        cv2.line(overlay, (w - r, 0), (w - r, h), (0, 255, 0), 2)
        cv2.line(overlay, (0, t), (w, t), (0, 255, 0), 2)
        cv2.line(overlay, (0, h - b), (w, h - b), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(overlay, f"L:{l}px", (5, h // 2), font, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f"R:{r}px", (w - 80, h // 2), font, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f"T:{t}px", (w // 2 - 20, t + 20), font, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f"B:{b}px", (w // 2 - 20, h - 10), font, 0.5, (0, 255, 0), 1)

        return overlay