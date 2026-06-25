import unittest


class NamingAlignmentTest(unittest.TestCase):
    def test_scene_field_imports_and_legacy_aliases(self):
        from canonicalgs.model.encoder.scene_field import (
            GPDecoderHead,
            SceneFieldEncoder,
            SceneLattice,
        )
        from canonicalgs.model.encoder.gs_cube import (
            GSCubeEncoder,
            GSCubeHead,
            GSCubeInput,
        )

        self.assertIs(GSCubeEncoder, SceneFieldEncoder)
        self.assertIs(GSCubeHead, GPDecoderHead)
        self.assertIs(GSCubeInput, SceneLattice)

    def test_checkpoint_state_dict_keys_are_renamed_to_scene_field(self):
        from scripts.rename_checkpoints import rename_state_dict_keys

        state_dict = {
            "encoder.gs_cube_encoder.stem_layer.conv_layers.0.kernel": "stem",
            "encoder.gs_cube_encoder.gs_cube_head.l1.weight": "head",
            "encoder.depth_predictor.weight": "depth",
        }

        renamed = rename_state_dict_keys(state_dict)

        self.assertIn("encoder.scene_field_encoder.stem_layer.conv_layers.0.kernel", renamed)
        self.assertIn("encoder.scene_field_encoder.gp_decoder_head.l1.weight", renamed)
        self.assertIn("encoder.depth_predictor.weight", renamed)
        self.assertNotIn("encoder.gs_cube_encoder.stem_layer.conv_layers.0.kernel", renamed)
        self.assertNotIn("encoder.gs_cube_encoder.gs_cube_head.l1.weight", renamed)

    def test_dense_adapter_accepts_gaussians_per_voxel_name(self):
        import torch
        from canonicalgs.model.encoder.common.gaussian_adapter import (
            DenseGaussianAdapter,
            GaussianAdapterCfg,
        )

        adapter = DenseGaussianAdapter(
            GaussianAdapterCfg(gaussian_scale_min=1e-10, gaussian_scale_max=3.0, sh_degree=0)
        )

        class SceneField:
            C = torch.tensor([[0, 0, 0, 0]], dtype=torch.int32)
            F = torch.zeros((1, 20), dtype=torch.float32)

        gaussians = adapter.forward(
            torch.eye(4).reshape(1, 1, 4, 4),
            torch.eye(3).reshape(1, 1, 3, 3),
            torch.zeros((1, 2, 3), dtype=torch.float32),
            torch.ones((1, 1, 2), dtype=torch.float32),
            SceneField(),
            input_images=torch.zeros((1, 3), dtype=torch.float32),
            gpv=2,
        )

        self.assertEqual(tuple(gaussians.means.shape), (1, 2, 3))
        self.assertEqual(tuple(gaussians.opacities.shape), (1, 2))


if __name__ == "__main__":
    unittest.main()
