import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
from safetensors.torch import load_file
import torch
from torch import nn

from flower.evaluation.multistep_sequences import get_sequences
from flower.evaluation.utils import load_mode_from_safetensor
from flower.models.flower import FLOWERVLA
from flower.rollout.rollout_long_horizon import sequences_for_rank
from scripts.export_calvin_checkpoint import export


ROOT = Path(__file__).resolve().parents[1]


class TinyPolicy(nn.Module):
    _load_pretrained_weights = FLOWERVLA._load_pretrained_weights

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 3)

    @property
    def device(self):
        return self.linear.weight.device


class CalvinSequenceTests(unittest.TestCase):
    def test_reference_fixture_identity(self):
        path = ROOT / "conf/evaluation_sequences/calvin_1000_seed1.json"
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            "0862eef5270f059d53cf87a08eee82ccf5c81a34280cbbbc84aaf825d7948f56",
        )
        data = json.loads(path.read_text())
        sequences = get_sequences()
        self.assertEqual(len(sequences), 1000)
        for (state, tasks), row in zip(sequences, data["sequences"]):
            self.assertEqual(list(state.items()), list(row["initial_state"].items()))
            self.assertEqual(list(tasks), row["tasks"])

    def test_smoke_run_is_a_prefix_without_changing_numpy_rng(self):
        get_sequences.cache_clear()
        before = np.random.get_state()
        self.assertEqual(get_sequences(5), get_sequences()[:5])
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_distributed_partitions_preserve_order_and_coverage(self):
        for world_size in (1, 2, 3, 8):
            parts = []
            for rank in range(world_size):
                with patch("flower.rollout.rollout_long_horizon.dist.get_rank", return_value=rank), patch(
                    "flower.rollout.rollout_long_horizon.dist.get_world_size", return_value=world_size
                ):
                    parts.extend(sequences_for_rank(1000))
            self.assertEqual(parts, get_sequences())

    def test_training_and_standalone_share_the_release_protocol(self):
        training = OmegaConf.load(ROOT / "conf/callbacks/rollout_lh/calvin.yaml")
        evaluation = OmegaConf.load(ROOT / "conf/eval_calvin.yaml")
        for key, expected in {"num_sequences": 1000, "sequence_seed": 1, "ep_len": 1000}.items():
            self.assertEqual(training[key], expected)
            self.assertEqual(evaluation[key], expected)
        self.assertEqual(evaluation.seed, 0)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "policy.ckpt"
        self.model = TinyPolicy()
        self.state = {key: torch.ones_like(value) for key, value in self.model.state_dict().items()}
        self.ema = [torch.full_like(value, 2) for value in self.state.values()]

    def save(self, ema):
        torch.save({"state_dict": self.state, "callbacks": {"EMA": {"ema_weights": ema}}}, self.path)

    def test_ema_is_loaded_in_state_dict_order(self):
        self.save(self.ema)
        self.model._load_pretrained_weights(str(self.path))
        for value in self.model.state_dict().values():
            torch.testing.assert_close(value, torch.full_like(value, 2))

    def test_truncated_ema_is_rejected(self):
        self.save(self.ema[:-1])
        with self.assertRaisesRegex(ValueError, "EMA tensor count"):
            self.model._load_pretrained_weights(str(self.path))

    def test_reordered_ema_is_rejected(self):
        self.save(self.ema[::-1])
        with self.assertRaisesRegex(ValueError, "EMA shape mismatch"):
            self.model._load_pretrained_weights(str(self.path))

    def test_missing_model_weights_are_rejected(self):
        torch.save({"state_dict": {"linear.weight": self.state["linear.weight"]}}, self.path)
        with self.assertRaisesRegex(RuntimeError, "Missing key"):
            self.model._load_pretrained_weights(str(self.path))

    def test_export_preserves_ema_and_produces_a_portable_config(self):
        torch.save({
            "state_dict": self.state,
            "callbacks": {"EMA": {"ema_weights": self.ema}},
            "epoch": 49,
            "global_step": 50000,
        }, self.path)
        config_path = Path(self.directory.name) / "training.yaml"
        OmegaConf.save(OmegaConf.create({
            "seed": 242,
            "model": {"vlm_path": "microsoft/Florence-2-large"},
            "datamodule": {"root_data_dir": "/private/dataset"},
        }), config_path)
        bundle = Path(self.directory.name) / "bundle"
        export(self.path, config_path, bundle)
        restored = load_file(str(bundle / "model.safetensors"))
        for key, ema in zip(self.state, self.ema):
            self.assertTrue(torch.equal(restored[key], ema))
        config = OmegaConf.load(bundle / "config.yaml")
        self.assertTrue(OmegaConf.is_missing(config.datamodule, "root_data_dir"))
        self.assertIsNone(config.model.pretrained_model_path)
        manifest = json.loads((bundle / "manifest.json").read_text())
        self.assertEqual(manifest["global_step"], 50000)
        for name, digest in manifest["files"].items():
            self.assertEqual(hashlib.sha256((bundle / name).read_bytes()).hexdigest(), digest)

    def test_explicit_config_supports_a_relocated_checkpoint(self):
        self.path.touch()
        config_path = Path(self.directory.name) / "release.yaml"
        OmegaConf.save(
            OmegaConf.create({"model": {"_target_": "example.Policy", "pre_vlm_selection_mode": "split_text_proprio"}}),
            config_path,
        )
        with patch("flower.evaluation.utils.hydra.utils.instantiate") as instantiate:
            model = load_mode_from_safetensor(self.path, config_path=config_path)
        self.assertIs(model, instantiate.return_value)
        config = instantiate.call_args.args[0]
        self.assertEqual(config.pretrained_model_path, str(self.path))
        self.assertTrue(config.load_pretrained)


if __name__ == "__main__":
    unittest.main()
