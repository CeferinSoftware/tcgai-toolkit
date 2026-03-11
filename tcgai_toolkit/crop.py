"""Automatic card cropping from photographs.

Detects the card in a photograph taken on any background and extracts a
clean, perspective-corrected crop suitable for grading analysis.
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import load_image, resize_for_processing


class CardCropper:
    """Detect and extract a trading card from a photograph.

    Works best when the card is placed on a contrasting, solid-colour
    surface (dark mat, white desk, etc.).

    Parameters
    ----------
    min_area_ratio : float
        Minimum fraction of the image area that the detected card must
        occupy.  Contours smaller than this are rejected.  Default 0.15.
    """

    def __init__(self, min_area_ratio: float = 0.15):
        self._min_area = min_area_ratio

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
            If given, the cropped card is resized to exactly this size.
            A good default for high-res scans is ``(630, 880)``.
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
        img, scale = resize_for_processing(original, max_dim)

        quad = self._detect(img)
        if quad is None:
            raise RuntimeError(
                "Could not detect a card in the image. "
                "Ensure the card is on a contrasting background."
            )

        # Map detected quad back to original resolution
        quad_orig = (quad / scale).astype(np.float32)
        warped = self._warp(original, quad_orig, output_size)
        return warped

    def crop_all(
        self, source, max_dim: int = 1500
    ) -> list:
        """Detect and crop *all* cards visible in the image.

        Returns a list of cropped card images sorted left-to-right.
        """
        original = load_image(source)
        img, scale = resize_for_processing(original, max_dim)

        quads = self._detect_all(img)
        results = []
        for q in quads:
            q_orig = (q / scale).astype(np.float32)
            results.append(self._warp(original, q_orig))

        # Sort left-to-right by centre x
        results.sort(key=lambda c: c.shape[1])
        return results

    # ------------------------------------------------------------------

    def _detect(self, img: np.ndarray) -> Optional[np.ndarray]:
        candidates = self._detect_all(img)
        return candidates[0] if candidates else None

    def _detect_all(self, img: np.ndarray) -> list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Try multiple edge-detection strategies
        results = []
        for low, high in [(30, 90), (50, 150), (20, 60)]:
            edges = cv2.Canny(blurred, low, high)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            img_area = img.shape[0] * img.shape[1]

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < img_area * self._min_area:
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
                if len(approx) == 4:
                    results.append(approx.reshape(4, 2).astype(np.float32))

            if results:
                break

        return results

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
            # Ensure portrait
            if w > h:
                w, h = h, w

        dst = np.array(
            [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(img, M, (w, h))