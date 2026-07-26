from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.omnivoice_utils import (  # noqa: E402
    build_omnivoice_configs,
    build_tokenize_command,
    build_train_command,
    calculate_training_steps,
    create_omnivoice_manifests,
    find_omnivoice_checkpoint,
    package_omnivoice_checkpoint,
)


class OmniVoiceUtilsTests(unittest.TestCase):
    def test_manifest_conversion_uses_normalized_text_and_language(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            wavs = dataset / "wavs"
            wavs.mkdir(parents=True)
            (wavs / "one.wav").write_bytes(b"RIFF")
            (wavs / "two.wav").write_bytes(b"RIFF")
            (dataset / "metadata_train.csv").write_text(
                "one|Original text|Normalized text\n", encoding="utf-8"
            )
            (dataset / "metadata_val.csv").write_text(
                "two|Validation text|Validation normalized\n", encoding="utf-8"
            )

            result = create_omnivoice_manifests(dataset, root / "workspace", "fr")

            train = json.loads(Path(result["train_jsonl"]).read_text().strip())
            dev = json.loads(Path(result["dev_jsonl"]).read_text().strip())
            self.assertEqual(train["text"], "Normalized text")
            self.assertEqual(train["language_id"], "fr")
            self.assertTrue(Path(train["audio_path"]).is_absolute())
            self.assertEqual(dev["text"], "Validation normalized")
            self.assertEqual(result["sample_counts"], {"train": 1, "dev": 1})

    def test_configs_commands_and_step_calculation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_path, data_path, unused = build_omnivoice_configs(
                root,
                token_dir=root / "tokens",
                output_dir=root / "output",
                base_model="k2-fsa/OmniVoice",
                restore_path=None,
                steps=123,
                grad_accum=4,
                include_dev=False,
                extra_overrides={"learning_rate": 2e-5, "unknown": True},
            )
            train = json.loads(train_path.read_text())
            data = json.loads(data_path.read_text())
            self.assertEqual(train["steps"], 123)
            self.assertEqual(train["gradient_accumulation_steps"], 4)
            self.assertEqual(train["attn_implementation"], "sdpa")
            self.assertEqual(train["learning_rate"], 2e-5)
            self.assertEqual(data["dev"], [])
            self.assertEqual(unused, {"unknown": True})
            self.assertEqual(calculate_training_steps(10, 17, 8), 30)

            tokenize = build_tokenize_command(
                "python",
                input_jsonl=root / "train.jsonl",
                token_dir=root / "tokens",
                split="train",
                tokenizer_path="tokenizer",
            )
            training = build_train_command(
                "python",
                train_config=train_path,
                data_config=data_path,
                output_dir=root / "output",
            )
            self.assertIn("omnivoice.scripts.extract_audio_tokens", tokenize)
            self.assertIn("omnivoice.cli.train", training)

    def test_checkpoint_discovery_and_packaging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "output" / "checkpoint-10"
            checkpoint.mkdir(parents=True)
            (checkpoint / "config.json").write_text("{}", encoding="utf-8")
            (checkpoint / "model.safetensors").write_bytes(b"model")
            self.assertEqual(find_omnivoice_checkpoint(root / "output"), checkpoint)

            packaged = package_omnivoice_checkpoint(checkpoint, root / "ready")
            self.assertTrue((packaged / "config.json").is_file())
            self.assertTrue((packaged / "model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
