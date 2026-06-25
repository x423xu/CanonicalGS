from pathlib import Path
import unittest


class Swin3DInstallPatchTest(unittest.TestCase):
    def test_installer_patches_attention_sum_coffs_division(self):
        repo_root = Path(__file__).resolve().parents[1]
        install_script = (repo_root / "scripts" / "install_canonicalgs_env.sh").read_text()

        self.assertIn(
            "norm_attn_feats = raw_attn_feats / torch.clamp(sum_coffs, min=1e-6)",
            install_script,
        )


if __name__ == "__main__":
    unittest.main()
