"""ADM-style UNet (Dhariwal & Nichol 2021, as vendored in contrib/ocean4dvarnet_rollout/unet.py) used as a direct forecaster:
one forward pass maps the 29-day window (14 observed days, NaN -> 0) to the 29-day output, like src.models_UNet.UNet.
Drop-in `solver` for Plus4dVarNetForecastPatchGPU_UNet_OSSE (loss, weights, optimiser identical to UNet-N)."""
import torch
from contrib.ocean4dvarnet_rollout.unet import UNetModel


class ADMDirectForecaster(torch.nn.Module):
    def __init__(self, **unet_kwargs):
        super().__init__()
        self.net = UNetModel(**unet_kwargs)

    def forward(self, batch):
        x = torch.nan_to_num(batch.input)
        t = torch.zeros(x.shape[0], device=x.device)
        return self.net.predict(x, timesteps=t, extra=[])
