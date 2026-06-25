from pathlib import Path
import unittest


class Swin3DSubmoduleTest(unittest.TestCase):
    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_uses_canonicalgs_swin3d_fork(self):
        gitmodules = (self.repo_root / ".gitmodules").read_text()

        self.assertIn("x423xu/Swin3D.git", gitmodules)

    def test_installer_does_not_patch_swin3d_sources(self):
        install_script = (self.repo_root / "scripts" / "install_canonicalgs_env.sh").read_text()

        self.assertNotIn("apply_swin3d_patch", install_script)
        self.assertNotIn("root / \"Swin3D", install_script)
        self.assertIn("validate_swin3d_submodule", install_script)

    def test_swin3d_submodule_contains_canonicalgs_fixes(self):
        swin3d_root = self.repo_root / "third_party" / "Swin3D" / "Swin3D"
        model_text = (swin3d_root / "models" / "Swin3D.py").read_text()
        layers_text = (swin3d_root / "modules" / "mink_layers.py").read_text()
        attn_text = (swin3d_root / "sparse_dl" / "attn" / "attn_coff.py").read_text()

        self.assertIn("other_down_stride=2", model_text)
        self.assertIn("stem_norm='bn'", model_text)
        self.assertIn("norm=stem_norm", model_text)
        self.assertIn("norm='bn'", layers_text)
        self.assertIn("MinkowskiInstanceNorm", layers_text)
        self.assertIn("torch.clamp(sum_coffs, min=1e-6)", attn_text)


if __name__ == "__main__":
    unittest.main()
