"""Tests for the card cropping module."""

import unittest

import numpy as np

from tcgai_toolkit.crop import CardCropper


class TestCardCropper(unittest.TestCase):
    """Test the CardCropper class."""

    def setUp(self):
        self.cropper = CardCropper()

    def test_init_defaults(self):
        c = CardCropper()
        self.assertEqual(c.target_size, (750, 1050))
        self.assertEqual(c.padding, 5)

    def test_init_custom(self):
        c = CardCropper(target_size=(500, 700), padding=10)
        self.assertEqual(c.target_size, (500, 700))
        self.assertEqual(c.padding, 10)

    def test_crop_card_on_dark_bg(self):
        """White card on dark background should be detected."""
        bg = np.zeros((800, 600, 3), dtype=np.uint8)
        # Place a white rectangle (card) in the middle
        bg[150:650, 100:450] = 230

        card = self.cropper.crop(bg)
        self.assertIsNotNone(card)
        h, w = card.shape[:2]
        # Output should be roughly card-shaped
        self.assertGreater(h, w * 1.1, "Card should be portrait orientation")

    def test_crop_card_on_light_bg(self):
        """Dark card on light background should be detected."""
        bg = np.ones((800, 600, 3), dtype=np.uint8) * 230
        bg[150:650, 100:450] = 50

        card = self.cropper.crop(bg)
        self.assertIsNotNone(card)

    def test_crop_preserves_color(self):
        """Cropped output should maintain original color channels."""
        bg = np.zeros((800, 600, 3), dtype=np.uint8)
        bg[100:600, 100:400] = [200, 150, 100]  # BGR

        card = self.cropper.crop(bg)
        self.assertEqual(card.shape[2], 3)

    def test_crop_from_file_path(self):
        """String path should be accepted (though it may fail to load)."""
        with self.assertRaises((RuntimeError, FileNotFoundError)):
            self.cropper.crop("nonexistent_file.jpg")

    def test_crop_all_single_card(self):
        """crop_all should find at least one card."""
        bg = np.zeros((800, 600, 3), dtype=np.uint8)
        bg[100:600, 100:400] = 200

        cards = self.cropper.crop_all(bg)
        self.assertGreaterEqual(len(cards), 1)

    def test_crop_all_multiple_cards(self):
        """Two separate bright regions on dark bg."""
        bg = np.zeros((800, 1200, 3), dtype=np.uint8)
        bg[100:600, 50:370] = 210   # Card 1 (left)
        bg[100:600, 650:970] = 210  # Card 2 (right)

        cards = self.cropper.crop_all(bg)
        self.assertGreaterEqual(len(cards), 2)

    def test_crop_tiny_image_raises(self):
        """Image too small to contain a card."""
        tiny = np.zeros((20, 15, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            self.cropper.crop(tiny)

    def test_target_size_applied(self):
        """Output should match the configured target size."""
        c = CardCropper(target_size=(375, 525))
        bg = np.zeros((800, 600, 3), dtype=np.uint8)
        bg[100:600, 100:400] = 200

        card = c.crop(bg)
        self.assertEqual(card.shape[:2], (525, 375))


if __name__ == "__main__":
    unittest.main()