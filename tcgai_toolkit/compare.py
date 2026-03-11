"""Side-by-side card comparison utilities.

Compare two scans of the same card (e.g., front vs. reference) to spot
differences, or compare two different cards to identify reprints,
counterfeits, or condition differences.
"""

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .utils import load_image, resize_for_processing


@dataclass
class ComparisonResult:
    """Result of comparing two card images.

    Parameters
    ----------
    ssim : float
        Structural Similarity Index from 0 (completely different) to
        1 (identical).
    pixel_diff_pct : float
        Percentage of pixels that differ significantly.
    match_score : float
        Overall match percentage (0-100).
    diff_region_count : int
        Number of distinct regions that differ between the two images.
    """

    ssim: float
    pixel_diff_pct: float
    match_score: float
    diff_region_count: int = 0

    @property
    def is_match(self) -> bool:
        """Whether the two cards are likely the same."""
        return self.ssim > 0.75

    def summary(self) -> str:
        return (
            f"Match: {self.match_score:.1f}%  |  "
            f"SSIM: {self.ssim:.4f}  |  "
            f"Pixel diff: {self.pixel_diff_pct:.2f}%  |  "
            f"Diff regions: {self.diff_region_count}"
        )


class CardComparator:
    """Compare two card images for similarity analysis.

    Common use cases:
    - Compare front scan against a known reference to detect counterfeits
    - Compare two copies of the same card to assess relative condition
    - Verify print consistency across different copies

    Parameters
    ----------
    resize_to : tuple of (width, height)
        Both images are resized to this resolution before comparison.
        Default ``(400, 560)`` balances speed and accuracy.
    """

    def __init__(self, resize_to: Tuple[int, int] = (400, 560)):
        self._size = resize_to

    def compare(self, source_a, source_b) -> ComparisonResult:
        """Compare two card images.

        Parameters
        ----------
        source_a, source_b : str, Path, or np.ndarray
            The two images to compare.

        Returns
        -------
        ComparisonResult
        """
        img_a = self._prepare(source_a)
        img_b = self._prepare(source_b)

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        ssim_val = structural_similarity(gray_a, gray_b)

        diff = cv2.absdiff(gray_a, gray_b)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        pixel_diff = np.count_nonzero(thresh) / thresh.size * 100

        # Count distinct diff regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        diff_region_count = len(contours)

        match_score = ssim_val * 60 + (100 - pixel_diff) * 0.4

        return ComparisonResult(
            ssim=round(ssim_val, 4),
            pixel_diff_pct=round(pixel_diff, 2),
            match_score=round(max(min(match_score, 100), 0), 1),
            diff_region_count=diff_region_count,
        )

    def visual_diff(self, source_a, source_b) -> np.ndarray:
        """Generate a visual diff between two card images.

        Returns a colour image where differences are highlighted in red.
        """
        img_a = self._prepare(source_a)
        img_b = self._prepare(source_b)

        gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(gray_a, gray_b)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        overlay = img_a.copy()
        overlay[mask > 0] = [0, 0, 255]

        blended = cv2.addWeighted(img_a, 0.6, overlay, 0.4, 0)
        return blended

    # Keep old name as alias for backward compatibility
    diff_image = visual_diff

    def _prepare(self, source) -> np.ndarray:
        img = load_image(source)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return cv2.resize(img, self._size, interpolation=cv2.INTER_AREA)