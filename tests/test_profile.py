import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("local_tts", ROOT / "scripts/zipvoice_distill_local.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / "profiles/a-1.05-4steps.example.json").read_text())

    def test_approved_profile(self):
        self.assertEqual(module.validate_profile(self.profile)["speed"], 1.05)
        self.assertEqual(self.profile["steps"], 4)

    def test_invalid_settings(self):
        cases = [("speed", 0), ("speed", float("nan")), ("speed", True),
                 ("steps", 2), ("threads", 0), ("pre_clamp_peak_limit", 1),
                 ("vocoder_device", "mps"), ("model_name", "zipvoice")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                p = dict(self.profile, **{key: value})
                with self.assertRaises(ValueError):
                    module.validate_profile(p)

    def test_unrecognized_fields(self):
        with self.assertRaises(ValueError):
            module.validate_profile(dict(self.profile, voice_embedding="not allowed"))

    def test_missing_fields(self):
        del self.profile["speed"]
        with self.assertRaises(ValueError):
            module.validate_profile(self.profile)

    def test_import_requires_no_models_or_torch(self):
        self.assertTrue(callable(module.main))


if __name__ == "__main__":
    unittest.main()
