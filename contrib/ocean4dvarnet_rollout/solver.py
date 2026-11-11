"""Unrolled 4DVarNet solver with a UNet gradient model, from ocean4dvarnet (github CIA-Oceanix/ocean4dvarnet, branch
19-integrate-evolution-from-global-mapping, commit 3ffe812, ocean4dvarnet/models.py): GradSolver, DiffusionGradSolver and
GradModelWithCondition, copied without changes except:
  - GradModelWithCondition.forward passes `extra=[]` to the UNet when the solver gives `extra=None` (UNetModel.predict tests
    `"concat_conditioning" in extra`, which fails on None);
  - the ConvLSTM type test in DiffusionGradSolver.solver_step also accepts src.models.ConvLstmGradModel.
Prior and observation costs come from src.models (BilinAEPriorCost, BaseObsCost), as in the existing 4DVarNet configs.
Used by config/xp/forecast_SLA_4dvarnet29_unet_rollout_*.yaml with contrib.forecast_plus.models.Plus4dVarNetForecastPatchGPU,
which calls solver(batch) and uses the returned state (same interface as src.models.GradSolver).
"""
from typing import Optional

import torch
from torch import nn

from src.models import ConvLstmGradModel


class GradSolver(nn.Module):
    """Gradient-based solver for 4D-VarNet (ocean4dvarnet.models.GradSolver)."""

    def __init__(self, prior_cost, obs_cost, grad_mod, n_step, lr_grad=0.2, lbd=1.0, **kwargs):
        super().__init__()
        self.prior_cost = prior_cost
        self.obs_cost = obs_cost
        self.grad_mod = grad_mod
        self.n_step = n_step
        self.lr_grad = lr_grad
        self.lbd = lbd
        self._grad_norm = None

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init
        return batch.input.nan_to_num().detach().requires_grad_(True)

    def solver_step(self, state, batch, step):
        var_cost = self.prior_cost(state) + self.lbd**2 * self.obs_cost(state, batch)
        grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
        gmod = self.grad_mod(grad)
        state_update = 1 / (step + 1) * gmod + self.lr_grad * (step + 1) / self.n_step * grad
        return state - state_update

    def forward(self, batch):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch)
            self.grad_mod.reset_state(batch.input)
            for step in range(self.n_step):
                state = self.solver_step(state, batch, step=step)
                if not self.training:
                    state = state.detach().requires_grad_(True)
            if not self.training:
                state = self.prior_cost.forward_ae(state)
        return state


class DiffusionGradSolver(GradSolver):
    """Unrolled solver whose update is given by a (UNet) gradient model conditioned on the step (ocean4dvarnet.models.DiffusionGradSolver).

    input_grad_update: what the gradient model receives: "state", "obs+state", "obs-only", "subgrad", "subgrad+state", "grad",
    "grad+state", "gradsplit", "gradsplit+state" (optionally "+previous"). std_init: std of the random initial state (0: zeros).
    """

    def __init__(self, grad_mod, n_step, lr_grad=0.2, lbd=1.0, input_grad_update="grad", std_init=0.1,
                 prior_cost: Optional[nn.Module] = None, obs_cost: Optional[nn.Module] = None, **kwargs):
        super().__init__(prior_cost=prior_cost, obs_cost=obs_cost, grad_mod=grad_mod, n_step=n_step, lr_grad=lr_grad, lbd=lbd, **kwargs)
        self.input_grad_update = input_grad_update
        self.std_init = std_init

    def init_state(self, batch, x_init=None):
        if x_init is not None:
            return x_init.detach().requires_grad_(True)
        if self.std_init > 0:
            x0 = self.std_init * torch.randn_like(batch.input)
            return x0.detach().requires_grad_(True)
        return torch.zeros_like(batch.input).detach().requires_grad_(True)

    def init_h_state(self, batch, h_state=None):
        if h_state is not None:
            self.h_state = h_state
        else:
            self.h_state = torch.zeros_like(batch.input).detach().requires_grad_(True)

    def format2D_3D(self, x):
        if hasattr(self.grad_mod, "dim_3d"):
            if self.grad_mod.dim_3d:
                x = x.unsqueeze(1)
        return x

    def solver_step(self, state, batch, step, alpha_step=1.0):
        if isinstance(step, float):
            t = torch.tensor([step], device=state.device).repeat(state.shape[0])
        else:
            t = step

        if "subgrad" in self.input_grad_update:
            gobs = (batch.input - state).nan_to_num()
            gprior = state - self.prior_cost.forward_ae(state)
            grad = torch.concatenate((self.format2D_3D(gobs), self.format2D_3D(gprior)), dim=1)
            if "state" in self.input_grad_update:
                grad = torch.concatenate((grad, self.format2D_3D(state)), dim=1)
            if "previous" in self.input_grad_update:
                grad = torch.concatenate((grad, self.format2D_3D(self.h_state)), dim=1)
        elif "gradsplit" in self.input_grad_update:
            prior_cost = self.prior_cost(state)
            obs_cost = self.obs_cost(state, batch)
            grad_prior = torch.autograd.grad(prior_cost, state, create_graph=True)[0]
            grad_obs = torch.autograd.grad(obs_cost, state, create_graph=True)[0]
            grad = torch.concatenate((grad_prior, grad_obs), dim=1)
            if "state" in self.input_grad_update:
                grad = grad / ((grad**2).mean().sqrt().detach())
                grad = torch.concatenate((grad, state), dim=1)
        elif "grad" in self.input_grad_update:
            var_cost = self.prior_cost(state) + self.lbd**2 * self.obs_cost(state, batch)
            grad = torch.autograd.grad(var_cost, state, create_graph=True)[0]
            if "state" in self.input_grad_update:
                grad = grad / ((grad**2).mean().sqrt().detach())
                grad = torch.concatenate((grad, self.format2D_3D(state)), dim=1)
            if "previous" in self.input_grad_update:
                grad = torch.concatenate((grad, self.format2D_3D(self.h_state)), dim=1)
        elif self.input_grad_update == "obs-only":
            grad = batch.input.nan_to_num()
        elif self.input_grad_update == "obs+state":
            grad = torch.concatenate((self.format2D_3D(state), self.format2D_3D(batch.input.nan_to_num())), dim=1)

        if isinstance(self.grad_mod, ConvLstmGradModel):
            gmod = self.grad_mod(grad)
        else:
            gmod = self.grad_mod(grad, timesteps=t, extra=None)
        if hasattr(self.grad_mod, "dim_3d"):
            if self.grad_mod.dim_3d:
                gmod = gmod.squeeze(1)

        state_update = alpha_step * gmod
        if ("grad" in self.input_grad_update) and (self.lr_grad > 0.0):
            state_update += self.lr_grad * (step + 1) / self.n_step * grad[:, : state.shape[1], :, :]
        self.h_state = state_update
        return state - state_update

    def forward(self, batch, x_init=None, h_state=None, phase="test"):
        with torch.set_grad_enabled(True):
            state = self.init_state(batch, x_init=x_init)
            self.init_h_state(batch, h_state=h_state)
            self.grad_mod.reset_state(batch.input)
            if not self.training:
                if ("subgrad" in self.input_grad_update) or ("grad" not in self.input_grad_update):
                    state.requires_grad_(False)
                    self.h_state.requires_grad_(False)
            for step in range(self.n_step):
                alpha_step = 1.0 / self.n_step
                state = self.solver_step(state, batch, step=step / self.n_step, alpha_step=alpha_step)
                if (not self.training) and ("grad" in self.input_grad_update):
                    if "subgrad" in self.input_grad_update:
                        state = state.detach().requires_grad_(False)
                    else:
                        state = state.detach().requires_grad_(True)
        return state


class GradModelWithCondition(torch.nn.Module):
    """Wraps a gradient update model (e.g. the UNet) called with the step as "timestep" (ocean4dvarnet.models.GradModelWithCondition)."""

    def __init__(self, grad_model=False, dropout=0.0, use_grad_norm=True):
        super().__init__()
        self.grad_model = grad_model
        self.dropout = torch.nn.Dropout(dropout)
        self.use_grad_norm = use_grad_norm
        if hasattr(self.grad_model, "dim_3d"):
            self.dim_3d = self.grad_model.dim_3d
        if hasattr(self.grad_model, "dims"):
            if self.grad_model.dims == 3:
                self.dim_3d = True

    def reset_state(self, inp):
        self._grad_norm = None

    def forward(self, x, timesteps=None, extra=None):
        if extra is None:
            extra = []  # change from ocean4dvarnet: UNetModel.predict needs an iterable
        if self._grad_norm is None:
            self._grad_norm = (x**2).mean().sqrt() if self.use_grad_norm else 1.0
        x = x / self._grad_norm
        x = self.dropout(x)
        return self.grad_model.predict(x, timesteps=timesteps, extra=extra)
