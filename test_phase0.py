"""
Test Phase 0 implementation in main.py without camera or GUI execution.
"""
import ast
import os
import sys
import time
import io
import unittest
from unittest.mock import patch

class TestPhase0(unittest.TestCase):

    def setUp(self):
        with open("main.py", "r", encoding="utf-8") as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)

    def test_perf_profile_config_block(self):
        """Check PERF_PROFILE config block exists near the top."""
        self.assertIn("PERF_PROFILE", self.source[:1000], "PERF_PROFILE should be defined near top")
        # Check defaults
        ns = {"os": os}
        # Execute the first 30 lines
        lines = self.source.splitlines()[:30]
        exec("\n".join(lines), ns)
        self.assertIn("PERF_PROFILE", ns)
        self.assertEqual(ns["PERF_PROFILE"], "normal")

    def test_smooth_landmarks_is_noop(self):
        """smooth_landmarks must return current_lms unchanged."""
        fn_src = None
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'smooth_landmarks':
                lines = self.source.splitlines()
                fn_src = '\n'.join(lines[node.lineno - 1 : node.end_lineno])
                break
        self.assertIsNotNone(fn_src)
        ns = {}
        exec(fn_src, ns)
        smooth_landmarks = ns['smooth_landmarks']
        cur = object()
        prev = object()
        self.assertIs(smooth_landmarks(cur, prev), cur)
        self.assertIsNone(smooth_landmarks(None, prev))

    def test_perf_monitor_metrics_and_printing(self):
        """Test PerfMonitor tracks all 9 metrics and prints correctly when PERF=1."""
        # Find PerfMonitor and helper classes in source
        lines = self.source.splitlines()
        class_lines = []
        capture = False
        for line in lines:
            if "class _RollingAverage:" in line:
                capture = True
            if capture:
                class_lines.append(line)
                if line.startswith("perf_monitor = PerfMonitor()"):
                    break
        monitor_code = "\n".join(class_lines)
        self.assertTrue(len(monitor_code) > 0, "PerfMonitor code block not found")

        ns = {
            "collections": __import__("collections"),
            "threading": __import__("threading"),
            "time": time,
            "ENABLE_PERF": True
        }
        exec(monitor_code, ns)
        perf_monitor = ns["perf_monitor"]

        # Add samples for all 9 metrics
        perf_monitor.cam_read.add(1.5)
        perf_monitor.holistic.add(45.2)
        perf_monitor.feat_extract.add(2.1)
        perf_monitor.motion_pred.add(12.3)
        perf_monitor.letter_pred.add(3.4)
        perf_monitor.gui_render.add(6.7)
        perf_monitor.canvas_push.add(4.1)
        perf_monitor.ml_fps.tick()
        time.sleep(0.01)
        perf_monitor.ml_fps.tick()
        perf_monitor.gui_fps.tick()
        time.sleep(0.01)
        perf_monitor.gui_fps.tick()

        # Check values
        self.assertAlmostEqual(perf_monitor.cam_read.get(), 1.5, places=1)
        self.assertAlmostEqual(perf_monitor.holistic.get(), 45.2, places=1)
        self.assertAlmostEqual(perf_monitor.feat_extract.get(), 2.1, places=1)
        self.assertAlmostEqual(perf_monitor.motion_pred.get(), 12.3, places=1)
        self.assertAlmostEqual(perf_monitor.letter_pred.get(), 3.4, places=1)
        self.assertAlmostEqual(perf_monitor.gui_render.get(), 6.7, places=1)
        self.assertAlmostEqual(perf_monitor.canvas_push.get(), 4.1, places=1)
        self.assertGreater(perf_monitor.ml_fps.get(), 0.0)
        self.assertGreater(perf_monitor.gui_fps.get(), 0.0)

        # Test printing once per second
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            # First call when time diff < 1.0 shouldn't print if recently initialized
            perf_monitor.last_print_t = time.time()
            perf_monitor.maybe_print()
            self.assertEqual(captured_output.getvalue(), "")

            # Set last_print_t to >1.0 second ago
            perf_monitor.last_print_t = time.time() - 1.5
            perf_monitor.maybe_print()
            out = captured_output.getvalue()
            self.assertIn("[PERF]", out)
            self.assertIn("camera read:", out)
            self.assertIn("holistic:", out)
            self.assertIn("feature extraction:", out)
            self.assertIn("motion model predict:", out)
            self.assertIn("letter model predict:", out)
            self.assertIn("GUI render:", out)
            self.assertIn("canvas push:", out)
            self.assertIn("ML-loop FPS:", out)
            self.assertIn("GUI FPS:", out)

    def test_disabled_perf_prints_nothing(self):
        """When PERF is not 1, maybe_print does not print anything."""
        lines = self.source.splitlines()
        class_lines = []
        capture = False
        for line in lines:
            if "class _RollingAverage:" in line:
                capture = True
            if capture:
                class_lines.append(line)
                if line.startswith("perf_monitor = PerfMonitor()"):
                    break
        monitor_code = "\n".join(class_lines)
        ns = {
            "collections": __import__("collections"),
            "threading": __import__("threading"),
            "time": time,
            "ENABLE_PERF": False
        }
        exec(monitor_code, ns)
        perf_monitor = ns["perf_monitor"]
        perf_monitor.last_print_t = time.time() - 2.0
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            perf_monitor.maybe_print()
            self.assertEqual(captured_output.getvalue(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
