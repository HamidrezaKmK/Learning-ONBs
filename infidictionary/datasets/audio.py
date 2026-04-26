import glob
import os

import numpy as np
import torch

from infidictionary.datasets.base import IrregularDataset


class SpokenDigitDataset(IrregularDataset):
    """Free Spoken Digit Dataset (FSDD) as 1-D irregular functional data.

    Each audio clip is a function  f : [0, 1] → R¹  whose observations live
    at positions  ``t_k = (k + 0.5) / L``  for ``k = 0, …, L-1``, where ``L``
    is the clip's natural sample count (after resampling to ``target_sr``).
    Because FSDD clips have variable durations (≈ 0.5 – 1.5 s after silence
    trimming), each clip has a different ``L`` and therefore a different
    coordinate grid — no interpolation is ever applied to the raw samples.

    ``get_batch`` returns **one clip** per call (the ``batch_size`` argument is
    ignored), subsampled to ``batch_n_points`` of its actual sample positions.
    Use ``grad_accumulation_steps ≥ 32`` in the training config to accumulate
    gradients across many clips.

    ``__call__(seed)`` returns the **full** clip (all ``L`` raw points) and is
    used by the notebook for visualisation.

    Setup
    -----
    ::

        git clone https://github.com/Jakobovski/free-spoken-digit-dataset
        # WAV files live in  free-spoken-digit-dataset/recordings/

    Parameters
    ----------
    data_dir : str
        Path to the ``recordings/`` directory.
    target_sr : int
        Resample clips to this rate before storing (default 8 000 Hz).
    batch_n_points : int
        Number of raw sample positions drawn per ``get_batch`` call.
    n_images : int
        Maximum number of clips to pre-load.
    digits : list[int] | None
        Filter by digit label (None = all 0–9).
    speakers : list[str] | None
        Filter by speaker name (None = all).
    device : str | torch.device
    """

    def __init__(
        self,
        data_dir: str,
        target_sr: int = 8_000,
        batch_n_points: int = 512,
        n_images: int = 200,
        digits: list[int] | None = None,
        speakers: list[str] | None = None,
        device: str | torch.device = "cpu",
    ):
        from scipy.io import wavfile
        from scipy.signal import resample as scipy_resample

        self.batch_n_points = batch_n_points
        self.channels       = 1
        self.device         = torch.device(device)

        # ------------------------------------------------------------------
        # Collect and filter WAV files.
        # Filename convention:  <digit>_<speaker>_<index>.wav
        # ------------------------------------------------------------------
        all_files = sorted(glob.glob(os.path.join(data_dir, "*.wav")))
        if not all_files:
            raise FileNotFoundError(f"No WAV files found in {data_dir!r}")

        selected = []
        for path in all_files:
            stem  = os.path.basename(path).rsplit(".", 1)[0]
            parts = stem.split("_")
            if len(parts) < 2:
                continue
            try:
                digit = int(parts[0])
            except ValueError:
                continue
            speaker = parts[1]
            if digits   is not None and digit   not in digits:
                continue
            if speakers is not None and speaker not in speakers:
                continue
            selected.append(path)

        if not selected:
            raise ValueError("No WAV files matched the digit/speaker filter.")

        # ------------------------------------------------------------------
        # Load, resample, normalise.  Keep every clip at its NATURAL length.
        # ------------------------------------------------------------------
        waveforms: list[torch.Tensor]  = []   # each (L_i,)
        coords_list: list[torch.Tensor] = []  # each (L_i, 1)

        for path in selected[:n_images]:
            sr, raw = wavfile.read(path)
            data = raw.astype(np.float32)
            if data.ndim > 1:
                data = data.mean(axis=1)
            # Convert integer PCM to float in [-1, 1].
            if raw.dtype == np.int16:
                data /= 32_768.0
            elif raw.dtype == np.int32:
                data /= 2_147_483_648.0
            elif raw.dtype == np.uint8:
                data = (data - 128.0) / 128.0

            if sr != target_sr:
                n_new = max(1, int(round(len(data) * target_sr / sr)))
                data  = scipy_resample(data, n_new).astype(np.float32)

            L  = len(data)
            # Coordinates: centre of each sample bin, normalised to (0, 1).
            ts = torch.linspace(0.5 / L, 1.0 - 0.5 / L, L)
            waveforms.append(torch.from_numpy(data).float())
            coords_list.append(ts.unsqueeze(-1))    # (L, 1)

        # Global amplitude normalisation (max absolute value → 1).
        global_max = max(w.abs().amax().item() for w in waveforms)
        scale = max(global_max, 1e-8)
        self.waveforms   = [w / scale for w in waveforms]   # list of (L_i,)
        self.coords_list = coords_list                       # list of (L_i, 1)
        self.n_images    = len(self.waveforms)

    # ------------------------------------------------------------------
    # IrregularDataset interface
    # ------------------------------------------------------------------

    def get_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one clip subsampled to ``batch_n_points`` raw sample positions.

        ``batch_size`` is intentionally ignored — each call returns a single
        clip (shape ``(1, K, 1)``) so that every observation is evaluated at
        its *own* natural coordinate grid, with no interpolation.
        Use ``grad_accumulation_steps`` in the config to accumulate gradients
        across multiple clips per optimiser step.

        Returns
        -------
        coords : (K, 1)    — sorted raw sample positions in (0, 1)
        values : (1, K, 1) — amplitude at those positions
        """
        i   = torch.randint(self.n_images, (1,)).item()
        wav = self.waveforms[i]                  # (L_i,)
        L_i = wav.shape[0]
        K   = min(self.batch_n_points, L_i)

        # Random subset of actual sample indices — NO interpolation.
        perm = torch.randperm(L_i)[:K]
        perm, _ = perm.sort()                    # preserve temporal order

        coords = self.coords_list[i][perm].to(self.device)              # (K, 1)
        vals   = wav[perm].unsqueeze(0).unsqueeze(-1).to(self.device)   # (1, K, 1)
        return coords, vals

    def __call__(self, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one clip at its full natural coordinate grid (all L_i points)."""
        i      = seed % self.n_images
        coords = self.coords_list[i].to(self.device)              # (L_i, 1)
        vals   = self.waveforms[i].unsqueeze(-1).to(self.device)  # (L_i, 1)
        return coords, vals

    def eval_at(self, t_dense: torch.Tensor, idx: int) -> torch.Tensor:
        """Evaluate clip ``idx`` at a dense sorted grid for visualisation.

        Uses nearest-neighbour lookup — returns the amplitude of the raw sample
        closest in time.  This is only for display; it is NOT called during training.

        Parameters
        ----------
        t_dense : (M,)  sorted time points in (0, 1)
        idx     : clip index

        Returns
        -------
        (M, 1) amplitude values
        """
        x_old = self.coords_list[idx].squeeze(-1).to(t_dense.device)  # (L_i,)
        y_old = self.waveforms[idx].to(t_dense.device)                 # (L_i,)
        nearest = (t_dense.unsqueeze(1) - x_old.unsqueeze(0)).abs().argmin(dim=1)
        return y_old[nearest].unsqueeze(-1)                            # (M, 1)
