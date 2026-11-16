"""Moving-patch data module with observation noise, after ocean4dvarnet.data.NoisyLazyDataModule (CIA-Oceanix).

Noise is added to the *input* only (the target stays clean), in metres, on observed cells (NaN stays NaN), before normalisation:
  train: scale ~ U(0, noise) drawn per patch, noise = scale * N(0, 1)     (a different noise amplitude for every sample); with aug_kw
         (our configs) the training input is built in AugmentedDataset, so the noise is added there (NoisyAugmentedDataset)
  val  : noise ~ U(-noise, +noise) per cell
  test : no noise
Same as MovingPatchDataModuleFastRecGPU otherwise. Config: `noise: 0.03` (metres) under `datamodule:`.
"""
import functools as ft

import numpy as np

from src.data import AugmentedDataset, TrainingItem
from contrib.moving_patches.movpatch_data_tests import MovingPatchDataModuleFastRecGPU, XrDatasetMovingPatchFastRecGPU


class NoisyAugmentedDataset(AugmentedDataset):
    """AugmentedDataset whose input (clean target seen through a permuted mask) gets noise, scale ~ U(0, noise_norm) per patch (normalised units).
    The training config uses aug_only=True, so the training input is built here, not in post_fn."""

    def __init__(self, *args, noise_norm=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_norm = noise_norm
        self._rng = np.random.default_rng()

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if not self.noise_norm:
            return item
        n = self._rng.uniform(0.0, self.noise_norm) * self._rng.normal(0.0, 1.0, item.input.shape)
        return item._replace(input=item.input + n.astype(np.float32))  # NaN (unobserved) stays NaN


class NoisyMovingPatchDataModuleFastRecGPU(MovingPatchDataModuleFastRecGPU):
    def __init__(self, *args, noise=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise = noise  # metres
        self._rng = np.random.default_rng()

    def post_fn_phase(self, phase):
        m, s = self.norm_stats()
        nl = self.noise

        def add_noise(x):
            if not nl or phase == "test":
                return x
            if phase == "train":
                noise = self._rng.uniform(0.0, nl) * self._rng.normal(0.0, 1.0, x.shape)
            else:
                noise = self._rng.uniform(-nl, nl, x.shape)
            return x + noise.astype(np.float32)

        return ft.partial(ft.reduce, lambda i, f: f(i), [
            TrainingItem._make,
            lambda item: item._replace(tgt=(item.tgt - m) / s),
            lambda item: item._replace(input=(add_noise(item.input) - m) / s),
        ])

    def setup(self, stage="test"):
        print("Entered NoisyMovingPatchDataModuleFastRecGPU, noise (m) =", self.noise)
        mk = lambda phase, rand: XrDatasetMovingPatchFastRecGPU(
            self.input_da.sel(self.domains[phase]), **self.xrds_kw, postpro_fn=self.post_fn_phase(phase), rand=rand)
        self.train_ds, self.val_ds, self.test_ds = mk("train", True), mk("val", False), mk("test", False)
        if self.aug_kw:
            m, s = self.norm_stats()
            self.train_ds = NoisyAugmentedDataset(self.train_ds, noise_norm=(self.noise / s) if self.noise else None, **self.aug_kw)
