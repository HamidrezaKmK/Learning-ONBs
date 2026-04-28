"""Run notebooks and extract every PNG output to outputs/figures/<name>/v<i>/cell_<j>.png.

Each entry in ``NOTEBOOKS`` is a ``(notebook_path, params)`` tuple. ``params``
is a dict of variable overrides; they are injected via a code cell placed
right after the notebook's ``parameters``-tagged cell (papermill convention).
When the same notebook appears multiple times, outputs go into ``v0``,
``v1``, ... in order of appearance.

Cell-by-cell execution: on the first error, the notebook is flagged with the
failing cell index and the rest of its cells are skipped.

Usage:
    python extract_notebook_figures.py
"""
from __future__ import annotations

import base64
import shutil
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError


# ── Edit me ─────────────────────────────────────────────────────────────────
NOTEBOOKS: list[tuple[str, dict]] = [
    # 1D PCA with all of the ablations
    # ("notebooks/fpca_1d.ipynb", {
    #     "CHECKPOINT_DIR":  "../outputs/checkpoints/wandb-5yylqjxg",
    #     "CHECKPOINT_FILE": "step_3520.pt",
    # }),
    # FPCA experiment on INRs
    # ("notebooks/inr_fpca.ipynb", {
    #     "CHECKPOINT_DIR":  "../outputs/checkpoints/wandb-39iw3oae",
    #     "CHECKPOINT_FILE": "step_2400.pt",
    #     "EXP_NAME": "CIFAR10"
    # }),
    # ("notebooks/inr_fpca.ipynb", {
    #     "CHECKPOINT_DIR":  "../outputs/checkpoints/wandb-a2rwqque",
    #     "CHECKPOINT_FILE": "step_10000.pt",
    #     "EXP_NAME": "MNIST"
    # }),
    ## NTK experiment
    # ("notebooks/ntk.ipynb", {
    #     "INSET_COL": 0,
    #     "INSET_LEFT": 1/5.,
    #     "INSET_RIGHT": 1/2. + 0.1,
    #     "INSET_DOWN": 1/5.,
    #     "INSET_UP": 1/2. + 0.1,
    #     "INSET_TARGET": 64,
    # }),
    ### FPCA on CelebA
    # ("notebooks/fpca.ipynb", {
    #     "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-i28fso8p",
    #     "CHECKPOINT_FILE": "step_3480.pt",
    # }),
    # ("notebooks/fpca.ipynb", {
    #     "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-n4tfx41g",
    #     "CHECKPOINT_FILE": "step_5100.pt",
    # }),
    # ("notebooks/fpca.ipynb", {
    #     "CHECKPOINT_DIR": "../outputs/checkpoints/wandb-4g6w8iu7",
    #     "CHECKPOINT_FILE": "step_2160.pt",
    # }),
    ### Euclidean Group
    # ("notebooks/euclidean_group.ipynb", {
    #     "CKPT_DIR": "outputs/checkpoints/wandb-fkhe366y",
    #     "CKPT_FILE": "step_1314.pt",
    # }),
    # ("notebooks/euclidean_group.ipynb", {
    #     "CKPT_DIR": "outputs/checkpoints/wandb-2ymvx2m8",
    #     "CKPT_FILE": "step_3650.pt",
    # }),
]

# Overleaf path for sync
OUTPUT_DIR = Path("../overleaf/Learning-Basis-Functions/notebook_generations")
DPI = 200
EXECUTION_TIMEOUT = -1  # per-cell, in seconds; -1 = no timeout
# ────────────────────────────────────────────────────────────────────────────


HIGH_DPI_PREAMBLE = f"""\
import matplotlib as _mpl
_mpl.rcParams['figure.dpi']   = {DPI}
_mpl.rcParams['savefig.dpi']  = {DPI}
_mpl.rcParams['savefig.bbox'] = 'tight'
try:
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats('png2x')
except Exception:
    pass
"""


def find_parameters_cell(nb) -> int | None:
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code" and "parameters" in cell.metadata.get("tags", []):
            return i
    return None


def render_overrides(params: dict) -> str:
    body = "\n".join(f"{k} = {v!r}" for k, v in params.items())
    return f"# Injected overrides from extract_notebook_figures.py\n{body}\n"


def save_pngs_from_cell(cell, target_dir: Path, cell_index: int) -> int:
    """Write every PNG attached to a cell. Returns how many were written."""
    if cell.cell_type != "code":
        return 0
    written = 0
    for output in cell.get("outputs", []):
        png_b64 = output.get("data", {}).get("image/png")
        if png_b64 is None:
            continue
        suffix = "" if written == 0 else f"_{written}"
        path = target_dir / f"cell_{cell_index}{suffix}.png"
        path.write_bytes(base64.b64decode(png_b64))
        written += 1
    return written


def run_notebook(nb_path: Path, params: dict, target_dir: Path) -> tuple[bool, str | None, int]:
    """Execute ``nb_path`` cell-by-cell. Returns (ok, error_msg, png_count)."""
    nb = nbformat.read(str(nb_path), as_version=4)

    # Inject parameter overrides right after the user's `parameters` cell.
    if params:
        param_idx = find_parameters_cell(nb)
        if param_idx is None:
            return (
                False,
                f"no `parameters`-tagged cell found; cannot inject {list(params)}",
                0,
            )
        override = nbformat.v4.new_code_cell(source=render_overrides(params))
        override.metadata["__figure_extractor_injected__"] = True
        nb.cells.insert(param_idx + 1, override)

    # Inject the high-DPI matplotlib preamble at the very top.
    preamble = nbformat.v4.new_code_cell(source=HIGH_DPI_PREAMBLE)
    preamble.metadata["__figure_extractor_injected__"] = True
    nb.cells.insert(0, preamble)

    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    kernel_name = nb.metadata.get("kernelspec", {}).get("name", "python3")
    client = NotebookClient(
        nb,
        timeout=EXECUTION_TIMEOUT,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(nb_path.parent.resolve())}},
        allow_errors=False,
    )

    user_index = 0   # cell index excluding any injected cells
    total_pngs = 0

    with client.setup_kernel():
        for raw_index, cell in enumerate(nb.cells):
            try:
                client.execute_cell(cell, raw_index)
            except CellExecutionError as exc:
                first_line = exc.evalue.splitlines()[0] if exc.evalue else ""
                msg = f"cell {user_index}: {exc.ename}: {first_line}"
                return False, msg, total_pngs
            if cell.metadata.get("__figure_extractor_injected__"):
                continue
            total_pngs += save_pngs_from_cell(cell, target_dir, user_index)
            user_index += 1

    return True, None, total_pngs


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, int, str]] = []

    seen: dict[str, int] = {}
    for nb_str, params in NOTEBOOKS:
        nb_path = Path(nb_str)
        version = seen.get(nb_path.stem, 0)
        seen[nb_path.stem] = version + 1
        target_dir = OUTPUT_DIR / nb_path.stem / f"v{version}"

        if not nb_path.exists():
            print(f"[SKIP] {nb_str} v{version}: file not found")
            failures.append((nb_str, version, "file not found"))
            continue

        print(f"[RUN ] {nb_path} v{version}  ->  {target_dir}")
        ok, err, n_pngs = run_notebook(nb_path, params, target_dir)
        plural = "s" if n_pngs != 1 else ""
        if ok:
            print(f"[ OK ] {nb_path} v{version}  ({n_pngs} PNG{plural} written)")
        else:
            print(f"[FAIL] {nb_path} v{version}: {err}  ({n_pngs} PNG{plural} written before failure)")
            failures.append((nb_str, version, err or "unknown error"))

    print()
    if failures:
        print("Notebooks that failed:")
        for nb_str, v, msg in failures:
            print(f"  - {nb_str} v{v}: {msg}")
        return 1

    print("All notebooks executed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
