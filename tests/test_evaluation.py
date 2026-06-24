import importlib.util
import os
from pathlib import Path
import unittest


def load_evaluation_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("evaluation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class EvaluationScriptTest(unittest.TestCase):
    def setUp(self):
        self.previous_re10k_root = os.environ.get("CANONICALGS_RE10K_ROOT")

    def tearDown(self):
        if self.previous_re10k_root is None:
            os.environ.pop("CANONICALGS_RE10K_ROOT", None)
        else:
            os.environ["CANONICALGS_RE10K_ROOT"] = self.previous_re10k_root

    def test_default_re10k_2v_len100_overrides(self):
        os.environ["CANONICALGS_RE10K_ROOT"] = "/datasets/re10k"
        evaluation = load_evaluation_module()

        args = evaluation.parse_args([])
        overrides = evaluation.build_overrides(args, Path("/repo"))

        self.assertIn("dataset.roots=[/datasets/re10k]", overrides)
        self.assertIn("dataset/view_sampler=evaluation", overrides)
        self.assertIn("dataset.view_sampler.index_path=/repo/assets/re10k_2v_canonical_len100_indicies.json", overrides)
        self.assertIn("dataset.view_sampler.num_context_views=2", overrides)
        self.assertIn("dataset.test_len=100", overrides)
        self.assertIn("checkpointing.pretrained_model=/repo/checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_10-step_340070.ckpt", overrides)
        self.assertIn("mode=test", overrides)
        self.assertIn("test.compute_scores=true", overrides)
        self.assertIn("test.render_chunk_size=10", overrides)
        self.assertIn("model.encoder.cube_merge_type=max_conf", overrides)

    def test_output_dir_includes_scale(self):
        evaluation = load_evaluation_module()

        args = evaluation.parse_args(["--output-dir", "notes/eval", "--cell-scale", "2.8"])
        overrides = evaluation.build_overrides(args, Path("/repo"))

        self.assertIn("output_dir=notes/eval_scale2.8", overrides)


if __name__ == "__main__":
    unittest.main()
