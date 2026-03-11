"""Tests for the surface defect detection module."""

import unittest

import numpy as np

from tcgai_toolkit.surface import SurfaceAnalyzer, SurfaceReport, Defect


class TestDefect(unittest.TestCase):
    """Test the Defect dataclass."""

    def test_create_defect(self):
        d = Defect(kind="scratch", x=100, y=200, w=50, h=5, severity=0.7)
        self.assertEqual(d.kind, "scratch")
        self.assertAlmostEqual(d.severity, 0.7)

    def test_defect_kinds(self):
        for kind in ("scratch", "stain", "print_line"):
            d = Defect(kind=kind, x=0, y=0, w=10, h=10, severity=0.5)
            self.assertEqual(d.kind, kind)


class TestSurfaceReport(unittest.TestCase):
    """Test the SurfaceReport dataclass."""

    def test_clean_card(self):
        report = SurfaceReport(defects=[], overall_score=1.0)
        self.assertEqual(len(report.defects), 0)
        self.assertAlmostEqual(report.overall_score, 1.0)

    def test_summary(self):
        defects = [
            Defect("scratch", 10, 20, 30, 2, 0.5),
            Defect("stain", 50, 60, 15, 15, 0.3),
        ]
        report = SurfaceReport(defects=defects, overall_score=0.7)
        summary = report.summary()
        self.assertIn("2", summary)


class TestSurfaceAnalyzer(unittest.TestCase):
    """Test the SurfaceAnalyzer class."""

    def setUp(self):
        self.analyzer = SurfaceAnalyzer(sensitivity=0.5)

    def test_init_default(self):
        a = SurfaceAnalyzer()
        self.assertAlmostEqual(a.sensitivity, 0.5)

    def test_init_custom(self):
        a = SurfaceAnalyzer(sensitivity=0.8)
        self.assertAlmostEqual(a.sensitivity, 0.8)

    def test_sensitivity_clamped(self):
        a = SurfaceAnalyzer(sensitivity=1.5)
        self.assertLessEqual(a.sensitivity, 1.0)

    def test_clean_surface(self):
        """Uniform surface should have no defects."""
        img = np.ones((500, 350, 3), dtype=np.uint8) * 180
        report = self.analyzer.analyze(img)
        self.assertIsInstance(report, SurfaceReport)
        self.assertEqual(len(report.defects), 0)
        self.assertAlmostEqual(report.overall_score, 1.0, places=1)

    def test_scratched_surface(self):
        """Thin bright line on dark surface should be detected as scratch."""
        img = np.ones((500, 350, 3), dtype=np.uint8) * 60
        # Add a horizontal scratch (thin bright line)
        img[250, 50:300] = 200

        report = self.analyzer.analyze(img)
        scratches = [d for d in report.defects if d.kind == "scratch"]
        self.assertGreater(len(scratches), 0, "Scratch should be detected")

    def test_stain_detection(self):
        """Large dark patch on uniform surface should be detected."""
        img = np.ones((500, 350, 3), dtype=np.uint8) * 200
        # Add a circular stain
        for y in range(200, 260):
            for x in range(150, 210):
                if (x - 180) ** 2 + (y - 230) ** 2 < 25 ** 2:
                    img[y, x] = [80, 70, 60]

        report = self.analyzer.analyze(img)
        stains = [d for d in report.defects if d.kind == "stain"]
        self.assertGreater(len(stains), 0, "Stain should be detected")

    def test_heatmap_output(self):
        """Heatmap should match input dimensions."""
        img = np.ones((500, 350, 3), dtype=np.uint8) * 180
        heatmap = self.analyzer.generate_heatmap(img)
        self.assertEqual(heatmap.shape[:2], img.shape[:2])
        self.assertEqual(heatmap.shape[2], 3)

    def test_high_sensitivity(self):
        """Higher sensitivity should find more defects on noisy surface."""
        rng = np.random.RandomState(42)
        img = rng.randint(100, 200, (500, 350, 3), dtype=np.uint8)

        low = SurfaceAnalyzer(sensitivity=0.2)
        high = SurfaceAnalyzer(sensitivity=0.9)

        r_low = low.analyze(img)
        r_high = high.analyze(img)

        self.assertLessEqual(
            len(r_low.defects), len(r_high.defects),
            "Higher sensitivity should detect >= as many defects",
        )

    def test_grayscale_input(self):
        """Grayscale input should be handled."""
        img = np.ones((500, 350), dtype=np.uint8) * 180
        report = self.analyzer.analyze(img)
        self.assertIsInstance(report, SurfaceReport)


if __name__ == "__main__":
    unittest.main()