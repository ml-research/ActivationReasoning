import importlib
import tempfile
import unittest
from pathlib import Path

from activation_reasoning.logic.config import LogicConfig


class PublicAPITest(unittest.TestCase):
    def test_config_roundtrip(self):
        cfg = LogicConfig(search_top_k=5, detection_threshold="auto")
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "logic_config.json"
            cfg.save(str(cfg_path))
            loaded = LogicConfig.load(str(cfg_path))
        self.assertEqual(loaded.search_top_k, 5)
        self.assertEqual(loaded.detection_threshold, "auto")

    def test_public_package_imports(self):
        pkg = importlib.import_module("activation_reasoning")
        self.assertTrue(hasattr(pkg, "ActivationReasoning"))
        self.assertTrue(hasattr(pkg, "LogicConfig"))
        self.assertTrue(hasattr(pkg, "LogicalParser"))


if __name__ == "__main__":
    unittest.main()
