import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from build_project import ITEMS, audit_dataset

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


class AuditTests(unittest.TestCase):
    def test_bundled_dataset(self):
        report = audit_dataset()
        self.assertEqual(report["rows"], 703)
        self.assertEqual(report["rule_matches"], 703)
        self.assertEqual(report["negative_labels"], 514)
        self.assertEqual(report["by_score"][6], {"negative": 70, "positive": 0})
        self.assertEqual(report["by_score"][7], {"negative": 0, "positive": 57})

    def test_invalid_total_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.csv"
            with path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([*ITEMS, "result", "class_asd"])
                writer.writerow([*([0] * 10), 7, 1])
            with self.assertRaisesRegex(ValueError, "result must equal"):
                audit_dataset(path)

    def test_app_reports_unavailable_dataset(self):
        with patch("build_project.audit_dataset", side_effect=FileNotFoundError("test")):
            app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertIn("Dataset audit unavailable", app.error[0].value)
        self.assertFalse(app.metric)
        self.assertFalse(app.select_slider)

    def test_app_score_boundary(self):
        app = AppTest.from_file(str(APP)).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertEqual(app.metric[0].value, "100.0%")
        self.assertTrue(any("Negative label" in item.value for item in app.markdown))
        app.select_slider[0].set_value(7).run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Positive label" in item.value for item in app.markdown))


if __name__ == "__main__":
    unittest.main()
