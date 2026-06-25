import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


def load_evaluation_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "evaluation.py"
    spec = importlib.util.spec_from_file_location("evaluation", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_index(repo_root: Path, entries: int = 3) -> Path:
    index_path = repo_root / "assets" / "re10k_2v_canonical_len100_indicies.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps({
        f"scene_{idx}": {"context": [0, 1], "target": [2]}
        for idx in range(entries)
    }))
    return index_path


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

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            write_index(repo_root)
            args = evaluation.parse_args([])
            overrides = evaluation.build_overrides(args, repo_root)

        self.assertIn("dataset.roots=[/datasets/re10k]", overrides)
        self.assertIn("dataset/view_sampler=evaluation", overrides)
        index_override = next(
            override for override in overrides
            if override.startswith("dataset.view_sampler.index_path=")
        )
        self.assertTrue(index_override.endswith("assets/re10k_2v_canonical_len100_indicies.json"))
        self.assertIn("dataset.view_sampler.num_context_views=2", overrides)
        self.assertIn("dataset.test_len=100", overrides)
        checkpoint_override = next(
            override for override in overrides
            if override.startswith("checkpointing.pretrained_model=")
        )
        self.assertTrue(checkpoint_override.endswith("checkpoints/gscube-depth22-gpc1-scale4-with-skip-small-woknn-maxconf-scratch-4w/checkpoints_backups/epoch_10-step_340070.ckpt"))
        self.assertIn("mode=test", overrides)
        self.assertIn("test.compute_scores=true", overrides)
        self.assertIn("test.render_chunk_size=10", overrides)
        self.assertIn("model.encoder.cube_merge_type=mean", overrides)


    def test_num_scenes_materializes_capped_evaluation_index(self):
        evaluation = load_evaluation_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            source_index = repo_root / "source_index.json"
            source_index.write_text(json.dumps({
                "scene_a": {"context": [0, 1], "target": [2]},
                "scene_b": {"context": [0, 1], "target": [2]},
                "scene_c": {"context": [0, 1], "target": [2]},
            }))

            args = evaluation.parse_args([
                "--index-path", str(source_index),
                "--num-scenes", "2",
            ])
            overrides = evaluation.build_overrides(args, repo_root)

            index_override = next(
                override for override in overrides
                if override.startswith("dataset.view_sampler.index_path=")
            )
            capped_index = Path(index_override.split("=", 1)[1])
            capped_data = json.loads(capped_index.read_text())

        self.assertEqual(["scene_a", "scene_b"], list(capped_data.keys()))

    def test_output_dir_includes_scale(self):
        evaluation = load_evaluation_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            write_index(repo_root)
            args = evaluation.parse_args(["--output-dir", "notes/eval", "--cell-scale", "2.8"])
            overrides = evaluation.build_overrides(args, repo_root)

        self.assertIn("output_dir=notes/eval_scale2.8", overrides)


if __name__ == "__main__":
    unittest.main()
