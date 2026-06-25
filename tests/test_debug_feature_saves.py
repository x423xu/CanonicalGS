from pathlib import Path
import unittest


class LatentSceneFeatureSaveTest(unittest.TestCase):
    def test_sem_seg_feature_saves_are_removed(self):
        repo_root = Path(__file__).resolve().parents[1]
        sources = [
            repo_root / "canonicalgs" / "model" / "encoder" / "encoder_canonicalgs.py",
            repo_root / "canonicalgs" / "model" / "model_wrapper.py",
            repo_root / "scripts" / "evaluation.py",
            repo_root / "config" / "main.yaml",
        ]

        for source in sources:
            text = source.read_text()
            self.assertNotIn("CANONICALGS_SAVE_SEM_SEG_FEATURES", text, source)
            self.assertNotIn("sem_seg", text, source)

    def test_latent_scene_package_is_saved_with_expected_keys(self):
        import tempfile
        import torch
        from canonicalgs.model.model_wrapper import save_latent_scene_package

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = save_latent_scene_package(
                Path(tmpdir),
                "scene_a",
                latent_scene=torch.ones(3, 5),
                coords=torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6], [0, 7, 8, 9]]),
                scene_lattice_xyz_min=torch.zeros(1, 1, 3),
                scene_lattice_cell_sizes=torch.ones(1, 3),
                context={
                    "extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
                    "intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
                    "near": torch.tensor([[0.5]]),
                    "far": torch.tensor([[100.0]]),
                    "image": torch.zeros(1, 2, 3, 16, 32),
                    "index": torch.tensor([[1, 2]]),
                },
                target={
                    "extrinsics": torch.eye(4).reshape(1, 1, 4, 4),
                    "intrinsics": torch.eye(3).reshape(1, 1, 3, 3),
                    "near": torch.tensor([[0.5]]),
                    "far": torch.tensor([[100.0]]),
                    "index": torch.tensor([[3]]),
                },
            )

            payload = torch.load(output_path, map_location="cpu")

        self.assertEqual(output_path.name, "latent_scene.pt")
        self.assertEqual(output_path.parent.name, "scene_a")
        self.assertEqual(tuple(payload["latent_scene"].shape), (3, 5))
        self.assertEqual(tuple(payload["coords"].shape), (3, 4))
        self.assertEqual(payload["scene"], "scene_a")
        self.assertEqual(payload["image_shape"], (16, 32))
        self.assertIn("target", payload)
        self.assertIn("context", payload)


if __name__ == "__main__":
    unittest.main()
