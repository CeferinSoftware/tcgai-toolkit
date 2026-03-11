"""Shared utility functions for image loading, preprocessing, and validation."""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


# Standard TCG card dimensions (mm) — used as reference aspect ratio
STANDARD_CARD_WIDTH_MM = 63.0
STANDARD_CARD_HEIGHT_MM = 88.0
STANDARD_ASPECT_RATIO = STANDARD_CARD_HEIGHT_MM / STANDARD_CARD_WIDTH_MM


def load_image(source, color_mode: str = "bgr") -> np.ndarray:
    """Load an image from a file path or accept an existing numpy array.

    Parameters
    ----------
    source : str, Path, or np.ndarray
        File path to an image or a pre-loaded numpy array.
    color_mode : str
        ``"bgr"`` (default, OpenCV native), ``"rgb"``, or ``"gray"``.

    Returns
    -------
    np.ndarray
        The loaded image in the requested colour space.

    Raises
    ------
    FileNotFoundError
        If *source* is a path that does not exist.
    ValueError
        If the file cannot be decoded or *color_mode* is invalid.
    """
    if isinstance(source, np.ndarray):
        img = source.copy()
    else:
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not decode image: {path}")

    if color_mode == "bgr":
        return img
    elif color_mode == "rgb":
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    else:
        raise ValueError(f"Unknown color_mode: {color_mode!r}")


def resize_for_processing(
    img: np.ndarray, max_dim: int = 1200
) -> Tuple[np.ndarray, float]:
    """Resize an image so its largest dimension is at most *max_dim* pixels.

    Returns the resized image and the scale factor applied.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA), scale


def auto_rotate(img: np.ndarray) -> np.ndarray:
    """Rotate the image so the card is in portrait orientation.

    Uses aspect ratio to decide — if the image is wider than it is tall, it
    is rotated 90 degrees clockwise.
    """
    h, w = img.shape[:2]
    if w > h:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img


def validate_card_aspect(
    img: np.ndarray, tolerance: float = 0.20
) -> bool:
    """Check whether the image has an aspect ratio close to a standard TCG card.

    Parameters
    ----------
    tolerance : float
        Maximum relative deviation from the standard 88:63 ratio.
    """
    h, w = img.shape[:2]
    ratio = max(h, w) / max(min(h, w), 1)
    return abs(ratio - STANDARD_ASPECT_RATIO) / STANDARD_ASPECT_RATIO <= tolerance