import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_pipeline import resolve_config_name, should_skip_norm, wandb_enabled


class TrainPipelineResolveTest(unittest.TestCase):
    def test_config_name_from_shell_config(self) -> None:
        self.assertEqual(
            resolve_config_name(
                preset="low_mem",
                config_name="pi05_g2_vr_low_mem",
                preset_from_cli=False,
                config_name_from_cli=False,
            ),
            "pi05_g2_vr_low_mem",
        )

    def test_smoke_preset_overrides_shell_config(self) -> None:
        self.assertEqual(
            resolve_config_name(
                preset="smoke",
                config_name="pi05_g2_vr",
                preset_from_cli=True,
                config_name_from_cli=False,
            ),
            "pi05_g2_vr_low_mem",
        )

    def test_cli_config_name_wins_over_smoke(self) -> None:
        self.assertEqual(
            resolve_config_name(
                preset="smoke",
                config_name="pi05_g2_vr",
                preset_from_cli=True,
                config_name_from_cli=True,
            ),
            "pi05_g2_vr",
        )

    def test_should_skip_norm_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            norm_stats = Path(tmp) / "norm_stats.json"
            self.assertFalse(should_skip_norm("auto", norm_stats))
            norm_stats.write_text("{}", encoding="utf-8")
            self.assertTrue(should_skip_norm("auto", norm_stats))

    def test_wandb_enabled(self) -> None:
        self.assertFalse(wandb_enabled("auto", "smoke"))
        self.assertTrue(wandb_enabled("auto", "low_mem"))
        self.assertFalse(wandb_enabled("off", "low_mem"))


if __name__ == "__main__":
    unittest.main()
