# CLAUDE.md

Guidance for Claude Code in this repository. This is the lean release: functional PCA only.
The regularizer (NTK, Taylor–Green, Euclidean group) and concept-basis experiments live on the
`full-reproducibility` branch.

## Setup

```bash
conda env create -f environment.yml   # pins torch 2.9.1 cu128 + torchvision 0.24.1
conda activate infidictionary
```

Also needs `safetensors` and `huggingface_hub` for the notebooks' pretrained downloads.

## Training

`fpca.py` is the only entrypoint (Hydra, `conf/fpca.yaml`). It jointly learns a mean function
and a `NeuralIsometry` that maximizes the energy the transformed dictionary captures.

```bash
python fpca.py +fpca_experiment=1d_fpca            # also 1d_fpca_haar, 1d_fpca_voronoi
python fpca.py +fpca_experiment=dwsnets_mnist_eulerian
python fpca.py +fpca_experiment=implicit_zoo_cifar_eulerian
python fpca.py +fpca_experiment=celeba_eulerian
```

Append `key=value` overrides (`seed=1`, `wandb=enabled`, `checkpointing.checkpoint_dir=...`).
`infidictionary/checkpointing.py` writes a rolling window of `step_N.pt` plus `config.yaml` to
`<checkpoint_dir>/<run>/`, where `<run>` is `wandb-<id>` with W&B on, otherwise a timestamp.

INR data is expected at `data/mnist-inrs` and `data/cifar_inrs_dataset` (gitignored; see README).
CelebA streams from Hugging Face `mattymchen/celeba-hq`; start concurrent CelebA jobs a minute
apart to avoid an HF datasets cache-lock deadlock.

## Layout (`infidictionary/`)

- `neural_isometries/`: `base.py` (`NeuralIsometry`: `pushforward` / `pullback`),
  `eulerian.py` (rank-R Cayley flow with a Woodbury solve; `max_gen_norm` caps the generator and
  is required for the 1-D flows to converge), `channel_orthogonal.py`, `chained.py`.
- `dictionaries/`: `base.py` (`InfiDictionary`, `orthogonal_synthesis`), `fourier.py`, `haar.py`,
  `voronoi.py` (grid-only `VoronoiPWC`).
- `networks/`: NeRF-style fields and the sinusoidal time embedding.
- `datasets/`: `synthetic.py` (`OneDimDiscontinuousGenerator`), `celeba.py`, `inr.py`
  (MNIST / CIFAR SIREN zoos sharing `_SirenDataset`).
- `domain_samplers.py`, `utils.py`, `checkpointing.py`.

Tensor convention: coords `(N, d)`, logabsdet `(N,)`, field `(B, N, C)`.

## Notebooks

`eulerian_transform`, `fpca_1d`, `fpca_celeba`, `inr_fpca`. `notebook_helpers.load_run` loads a
local run (`CHECKPOINT_DIR` / `CHECKPOINT_FILE`) or, when `CHECKPOINT_DIR` is `None`, the
published bundle from the Hugging Face repo `Shengdebao/Learning-ONBs`, one subfolder per
experiment named exactly like the experiment config.

## Tests

`pytest tests` (plain pytest, includes a 2-epoch smoke run of each 1-D experiment).
