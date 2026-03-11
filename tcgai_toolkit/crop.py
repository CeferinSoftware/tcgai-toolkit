"""Automatic card cropping from photographs.

Detects the card in a photograph taken on any background and extracts a
clean, perspective-corrected crop suitable for grading analysis.

Uses multiple detection strategies (adaptive threshold, Canny edges,
``minAreaRect`` fallback) and validates candidates by aspect-ratio and
corner-angle geometry so that internal artwork edges or image-boundary
artefacts are rejected.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .utils import load_image, resize_for_processing

# Standard TCG card aspect ratio  (2.5 / 3.5 ≈ 0.714)
_CARD_AR = 2.5 / 3.5
# Tolerance band for AR validation (± this fraction)
_AR_TOL = 0.18


class CardCropper:
    """Detect and extract a trading card from a photograph.

    Works best when the card is placed on a contrasting, solid-colour
    surface (dark mat, white desk, etc.).

    Parameters
    ----------
    target_size : tuple of (width, height), optional
        Default output resolution for cropped cards.  ``(750, 1050)`` by
        default, matching a 2.5 × 3.5 inch card at 300 DPI.
    padding : int
        Extra pixels to include around the detected card edge.
    min_area_ratio : float
        Minimum fraction of the image area that the detected card must
        occupy.  Contours smaller than this are rejected.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (750, 1050),
        padding: int = 5,
        min_area_ratio: float = 0.10,
    ):
        self.target_size = target_size
        self.padding = padding
        self._min_area = min_area_ratio

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def crop(
        self,
        source,
        output_size: Optional[Tuple[int, int]] = None,
        max_dim: int = 1500,
    ) -> np.ndarray:
        """Detect the card and return a cropped, corrected image.

        Parameters
        ----------
        source : str, Path, or np.ndarray
            Input photograph.
        output_size : tuple of (width, height), optional
            Override the default ``target_size`` for this call.
        max_dim : int
            Internal processing resolution.

        Returns
        -------
        np.ndarray
            Perspective-corrected card image (BGR).

        Raises
        ------
        RuntimeError
            If no card-like quadrilateral is found.
        """
        original = load_image(source)
        h, w = original.shape[:2]

        if h < 50 or w < 40:
            raise RuntimeError(
                "Image is too small to contain a trading card."
            )

        img, scale = resize_for_processing(original, max_dim)

        quad = self._detect(img)
        if quad is None:
            raise RuntimeError(
                "Could not detect a card in the image. "
                "Ensure the card is on a contrasting background."
            )

        quad_orig = (quad / scale).astype(np.float32)

        # Expand the detected quad outward so the crop includes the
        # full card border rather than cutting into it.  Edge
        # detectors tend to land *on* the border pixels, which clips
        # the outermost ring of the card.
        quad_orig = self._expand_quad(quad_orig, h, w)

        size = output_size or self.target_size
        return self._warp(original, quad_orig, size)

    def crop_all(
        self, source, max_dim: int = 1500
    ) -> List[np.ndarray]:
        """Detect and crop *all* cards visible in the image.

        Returns a list of cropped card images sorted left-to-right.
        """
        original = load_image(source)
        h_img, w_img = original.shape[:2]
        img, scale = resize_for_processing(original, max_dim)

        quads = self._detect_all(img)
        results: List[np.ndarray] = []
        for q in quads:
            q_orig = (q / scale).astype(np.float32)
            q_orig = self._expand_quad(q_orig, h_img, w_img)
            results.append(self._warp(original, q_orig, self.target_size))

        if len(quads) > 1:
            paired = list(zip(quads, results))
            paired.sort(key=lambda p: p[0][:, 0].min())
            results = [r for _, r in paired]

        return results

    # ------------------------------------------------------------------
    # Internal detection
    # ------------------------------------------------------------------

    def _detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        candidates = self._detect_all(img)
        return candidates[0] if candidates else None

    def _detect_all(self, img: np.ndarray) -> List[np.ndarray]:
        """Find card-like quadrilaterals using multiple strategies."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_area = img.shape[0] * img.shape[1]
        img_h, img_w = img.shape[:2]

        scored: List[Tuple[float, np.ndarray]] = []

        # --- Strategy 1: adaptive threshold ---------------------------
        for block in (15, 25, 41):
            binary = cv2.adaptiveThreshold(
                cv2.GaussianBlur(gray, (5, 5), 0),
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                3,
            )
            binary = cv2.bitwise_not(binary)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            binary = cv2.morphologyEx(
                binary, cv2.MORPH_CLOSE, kernel, iterations=3
            )
            self._extract_quads(binary, img_area, img_h, img_w, scored)

        # --- Strategy 2: Canny edges ---------------------------------
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        for low, high in [(30, 90), (50, 150), (20, 60), (70, 200)]:
            edges = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE, kernel, iterations=2
            )
            self._extract_quads(edges, img_area, img_h, img_w, scored)

        # --- Strategy 3: Otsu threshold -------------------------------
        _, otsu = cv2.threshold(
            cv2.GaussianBlur(gray, (7, 7), 0),
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        for variant in (otsu, cv2.bitwise_not(otsu)):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            closed = cv2.morphologyEx(
                variant, cv2.MORPH_CLOSE, kernel, iterations=3
            )
            self._extract_quads(closed, img_area, img_h, img_w, scored)

        # --- Strategy 4: minAreaRect fallback -------------------------
        # If no valid quad was found, try using the minimum-area
        # rotated rectangle of the largest contour.
        if not scored:
            self._minrect_fallback(gray, img_area, img_h, img_w, scored)

        if not scored:
            return []

        # Sort by score (higher = better) and de-duplicate
        scored.sort(key=lambda x: x[0], reverse=True)
        return [quad for _, quad in self._dedupe(scored)]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_quads(
        self,
        binary: np.ndarray,
        img_area: int,
        img_h: int,
        img_w: int,
        scored: List[Tuple[float, np.ndarray]],
    ) -> None:
        """Find quadrilateral contours in a binary image and score them."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < img_area * self._min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            # Try several epsilon values for approxPolyDP
            for eps_factor in (0.015, 0.02, 0.03, 0.04):
                approx = cv2.approxPolyDP(cnt, eps_factor * peri, True)
                if len(approx) == 4:
                    quad = approx.reshape(4, 2).astype(np.float32)
                    score = self._score_quad(
                        quad, area, img_area, img_h, img_w
                    )
                    if score > 0:
                        scored.append((score, quad))
                    break  # use first 4-point approximation

    def _minrect_fallback(
        self,
        gray: np.ndarray,
        img_area: int,
        img_h: int,
        img_w: int,
        scored: List[Tuple[float, np.ndarray]],
    ) -> None:
        """Use ``cv2.minAreaRect`` on the largest contour as fallback."""
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 90)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, kernel, iterations=3
        )
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < img_area * self._min_area:
            return
        rect = cv2.minAreaRect(biggest)
        box = cv2.boxPoints(rect).astype(np.float32)
        score = self._score_quad(
            box, cv2.contourArea(biggest), img_area, img_h, img_w
        )
        if score > 0:
            scored.append((score, box))

    def _score_quad(
        self,
        quad: np.ndarray,
        area: float,
        img_area: int,
        img_h: int,
        img_w: int,
    ) -> float:
        """Return a quality score for a candidate quad (0 = reject).

        Checks:
        1. Aspect ratio should be close to a standard trading card.
        2. All interior angles should be near 90°.
        3. No vertex should lie exactly on the image boundary (common
           artefact when the detector grabs image edges).
        """
        ordered = self._order_points(quad)

        # --- Aspect ratio ---
        w_top = float(np.linalg.norm(ordered[1] - ordered[0]))
        w_bot = float(np.linalg.norm(ordered[2] - ordered[3]))
        h_left = float(np.linalg.norm(ordered[3] - ordered[0]))
        h_right = float(np.linalg.norm(ordered[2] - ordered[1]))

        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2

        if avg_h < 1 or avg_w < 1:
            return 0.0

        ar = min(avg_w, avg_h) / max(avg_w, avg_h)
        if abs(ar - _CARD_AR) > _AR_TOL:
            return 0.0

        ar_score = 1.0 - abs(ar - _CARD_AR) / _AR_TOL  # 0..1

        # --- Angles ---
        angles = []
        for i in range(4):
            p0 = ordered[i]
            p1 = ordered[(i + 1) % 4]
            p2 = ordered[(i + 2) % 4]
            v1 = p0 - p1
            v2 = p2 - p1
            denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
            if denom < 1e-6:
                return 0.0
            cos_a = float(np.dot(v1, v2)) / denom
            angles.append(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

        max_dev = max(abs(a - 90) for a in angles)
        if max_dev > 20:
            return 0.0
        angle_score = 1.0 - max_dev / 20.0

        # --- Boundary penalty: reject quads touching image edges ------
        edge_margin = 3  # pixels
        on_edge = 0
        for pt in ordered:
            x, y = float(pt[0]), float(pt[1])
            if (
                x <= edge_margin
                or y <= edge_margin
                or x >= img_w - edge_margin
                or y >= img_h - edge_margin
            ):
                on_edge += 1
        if on_edge >= 3:
            return 0.0  # at least 3 corners on the edge = image border
        edge_score = 1.0 - on_edge * 0.25

        # --- Size: prefer larger cards --------------------------------
        size_score = min(area / img_area, 1.0)

        return ar_score * angle_score * edge_score * size_score

    @staticmethod
    def _dedupe(
        scored: List[Tuple[float, np.ndarray]],
        iou_thresh: float = 0.5,
    ) -> List[Tuple[float, np.ndarray]]:
        """Remove near-duplicate quads (keep highest-scored)."""
        kept: List[Tuple[float, np.ndarray]] = []
        for score, quad in scored:
            is_dup = False
            for _, existing in kept:
                if _quad_iou(quad, existing) > iou_thresh:
                    is_dup = True
                    break
            if not is_dup:
                kept.append((score, quad))
        return kept

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Order four points as: top-left, top-right, bottom-right,
        bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        d = np.diff(pts, axis=1).ravel()
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        return rect

    def _expand_quad(
        self,
        quad: np.ndarray,
        img_h: int,
        img_w: int,
        expand_pct: float = 0.03,
    ) -> np.ndarray:
        """Push each vertex outward from the centroid by *expand_pct*.

        This compensates for edge detectors landing right *on* the card
        boundary, which would clip the outermost border pixels.
        The result is clamped to the image dimensions.
        """
        centroid = quad.mean(axis=0)
        expanded = np.empty_like(quad)
        for i in range(len(quad)):
            direction = quad[i] - centroid
            expanded[i] = quad[i] + direction * expand_pct
        # Clamp to image bounds
        expanded[:, 0] = np.clip(expanded[:, 0], 0, img_w - 1)
        expanded[:, 1] = np.clip(expanded[:, 1], 0, img_h - 1)
        return expanded.astype(np.float32)

    def _warp(
        self,
        img: np.ndarray,
        quad: np.ndarray,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        ordered = self._order_points(quad)
        tl, tr, br, bl = ordered

        if output_size:
            w, h = output_size
        else:
            w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
            h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
            if w > h:
                w, h = h, w

        w = max(w, 1)
        h = max(h, 1)

        dst = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(img, M, (w, h))


# ------------------------------------------------------------------
# Module-level helper
# ------------------------------------------------------------------

def _quad_iou(q1: np.ndarray, q2: np.ndarray) -> float:
    """Approximate IoU between two quads via their bounding boxes."""
    def _bbox(q: np.ndarray):
        return (
            q[:, 0].min(), q[:, 1].min(),
            q[:, 0].max(), q[:, 1].max(),
        )

    x0a, y0a, x1a, y1a = _bbox(q1)
    x0b, y0b, x1b, y1b = _bbox(q2)

    xi = max(0, min(x1a, x1b) - max(x0a, x0b))
    yi = max(0, min(y1a, y1b) - max(y0a, y0b))
    inter = xi * yi

    area_a = (x1a - x0a) * (y1a - y0a)
    area_b = (x1b - x0b) * (y1b - y0b)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0