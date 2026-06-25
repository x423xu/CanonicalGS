from pathlib import Path
import re
import unittest


class DebugFeatureSaveTest(unittest.TestCase):
    def test_sem_seg_feature_saves_are_opt_in(self):
        repo_root = Path(__file__).resolve().parents[1]
        sources = [
            repo_root / "canonicalgs" / "model" / "encoder" / "encoder_canonicalgs.py",
            repo_root / "canonicalgs" / "model" / "model_wrapper.py",
        ]

        for source in sources:
            text = source.read_text()
            for match in re.finditer(r"torch\.save\([^\n]*(?:sem_seg|input_feature_dir)", text):
                guard_window = text[max(0, match.start() - 250):match.start()]
                self.assertIn(
                    "CANONICALGS_SAVE_SEM_SEG_FEATURES",
                    guard_window,
                    f"{source} has an unconditional sem_seg debug save",
                )


if __name__ == "__main__":
    unittest.main()
