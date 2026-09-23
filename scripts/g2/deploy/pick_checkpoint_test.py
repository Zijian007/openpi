import tempfile
import time
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pick_checkpoint import (
    autoselect_policy_step_directory,
    resolve_policy_step_directory,
    select_highest_numeric_step_directory,
)


class PickCheckpointTest(unittest.TestCase):
    def test_select_highest_numeric_step_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            experiment = Path(tmp) / "exp_a"
            (experiment / "5000").mkdir(parents=True)
            (experiment / "29999").mkdir(parents=True)
            self.assertEqual(
                select_highest_numeric_step_directory(experiment),
                experiment / "29999",
            )

    def test_autoselect_newest_experiment_then_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_step = root / "old_exp" / "30000"
            new_step_low = root / "new_exp" / "19"
            new_step_high = root / "new_exp" / "5000"
            for path in (old_step, new_step_low, new_step_high):
                path.mkdir(parents=True)
            old_step.touch()
            time.sleep(0.02)
            new_step_low.touch()
            time.sleep(0.02)
            new_step_high.touch()
            selection = autoselect_policy_step_directory(root)
            self.assertIsNotNone(selection)
            selected_step, experiment_name = selection
            self.assertEqual(experiment_name, "new_exp")
            self.assertEqual(selected_step, new_step_high)

    def test_resolve_explicit_step_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            step_directory = root / "pi05_g2_vr_low_mem" / "exp_a" / "1000"
            (step_directory / "params").mkdir(parents=True)
            resolved = resolve_policy_step_directory(
                train_config_name="pi05_g2_vr_low_mem",
                project_root_dir=root,
                explicit_step_directory=str(step_directory),
            )
            self.assertEqual(resolved, step_directory.resolve())


if __name__ == "__main__":
    unittest.main()
