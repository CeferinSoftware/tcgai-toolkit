"""Tests for the centering analysis module."""

import unittest

import numpy as np

from tcgai_toolkit.centering import CenteringAnalyzer, CenteringResult


class TestCenteringResult(unittest.TestCase):
    """Test the CenteringResult dataclass."""

    def test_perfect_centering(self):
        result = CenteringResult(
            lr_ratio=50.0,
            tb_ratio=50.0,
            left_px=20,
            right_px=20,
            top_px=30,
            bottom_px=30,
        )
        self.assertTrue(result.is_gem_mint)
        self.assertEqual(result.grade, "Gem Mint")

    def test_off_center_fails_gem_mint(self):
        result = CenteringResult(
            lr_ratio=65.0,
            tb_ratio=50.0,
            left_px=30,
            right_px=10,
            top_px=20,
            bottom_px=20,
        )
        self.assertFalse(result.is_gem_mint)

    def test_grade_categories(self):
        # Near perfect
        r1 = CenteringResult(48.0, 52.0, 19, 21, 30, 30)
        self.assertEqual(r1.grade, "Gem Mint")

        # Slightly off
        r2 = CenteringResult(42.0, 50.0, 16, 24, 30, 30)
        self.assertEqual(r2.grade, "Mint")

        # Moderately off
        r3 = CenteringResult(35.0, 50.0, 14, 26, 30, 30)
        self.assertEqual(r3.grade, "Near Mint")

        # Heavily off
        r4 = CenteringResult(25.0, 50.0, 10, 30, 30, 30)
        self.assertEqual(r4.grade, "Off-Center")

    def test_summary_format(self):
        result = CenteringResult(50.0, 50.0, 20, 20, 30, 30)
        summary = result.summary()
        self.assertIn("LR", summary)
        self.assertIn("TB", summary)
        self.assertIn("50.0", summary)


class TestCenteringAnalyzer(unittest.TestCase):
    """Test the CenteringAnalyzer class."""

    def setUp(self):
        self.analyzer = CenteringAnalyzer(border_method="threshold")

    def test_init_default(self):
        a = CenteringAnalyzer()
        self.assertEqual(a.border_method, "gradient")

    def test_init_custom_method(self):
        a = CenteringAnalyzer(border_method="threshold")
        self.assertEqual(a.border_method, "threshold")

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            CenteringAnalyzer(border_method="invalid")

    def test_analyze_synthetic_centered(self):
        """A perfectly centered card-like image should score near 50/50."""
        # Create a synthetic card: white border, dark rectangle center
        img = np.ones((350, 250, 3), dtype=np.uint8) * 240  # white
        img[30:320, 20:230] = 30  # dark artwork area

        result = self.analyzer.analyze(img)
        self.assertIsInstance(result, CenteringResult)
        self.assertAlmostEqual(result.lr_ratio, 50.0, delta=10)
        self.assertAlmostEqual(result.tb_ratio, 50.0, delta=10)

    def test_analyze_off_center_horizontal(self):
        """Card shifted right should show lr_ratio != 50."""
        img = np.ones((350, 250, 3), dtype=np.uint8) * 240
        # Shift artwork to the right: left border wider
        img[30:320, 50:240] = 30

        result = self.analyzer.analyze(img)
        # Left border is wider, so left_px > right_px
        self.assertGreater(result.left_px, result.right_px)

    def test_analyze_returns_positive_values(self):
        img = np.ones((350, 250, 3), dtype=np.uint8) * 200
        img[25:325, 15:235] = 40

        result = self.analyzer.analyze(img)
        self.assertGreater(result.left_px, 0)
        self.assertGreater(result.right_px, 0)
        self.assertGreater(result.top_px, 0)
        self.assertGreater(result.bottom_px, 0)

    def test_analyze_with_overlay(self):
        img = np.ones((350, 250, 3), dtype=np.uint8) * 220
        img[30:320, 20:230] = 30

        result, overlay = self.analyzer.analyze_with_overlay(img)
        self.assertIsInstance(result, CenteringResult)
        # Overlay is a 3-channel image (may be warped to different size)
        self.assertEqual(len(overlay.shape), 3)
        self.assertEqual(overlay.shape[2], 3)
        self.assertGreater(overlay.shape[0], 0)
        self.assertGreater(overlay.shape[1], 0)

    def test_gray_input(self):
        """Grayscale input should still work."""
        img = np.ones((350, 250), dtype=np.uint8) * 220
        img[30:320, 20:230] = 30

        result = self.analyzer.analyze(img)
        self.assertIsInstance(result, CenteringResult)


if __name__ == "__main__":
    unittest.main()