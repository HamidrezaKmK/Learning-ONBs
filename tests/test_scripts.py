"""Smoke test for fpca.py: each 1-D experiment trains briefly and writes checkpoints.

Only the 1-D experiments run here, since they need no downloaded data.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT, check=True)


def _train(experiment: str, checkpoint_root: Path) -> Path:
    _run(
        "fpca.py",
        f"+fpca_experiment={experiment}",
        "n_epochs=2",
        "grad_accumulation_steps=1",
        "model_state_kwargs.num_steps=4",
        "checkpointing.checkpoint_every_n_steps=1",
        f"checkpointing.checkpoint_dir={checkpoint_root}",
    )
    (run_dir,) = checkpoint_root.iterdir()
    return run_dir


@pytest.mark.parametrize("experiment", ["1d_fpca", "1d_fpca_haar", "1d_fpca_voronoi"])
def test_fpca_trains_and_writes_checkpoints(experiment, tmp_path):
    run_dir = _train(experiment, tmp_path / "checkpoints")
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "step_2.pt").is_file()

