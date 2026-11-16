"""OSSE evaluation of UNet-N and the unrolled 4DVarNet (ep304) on GLORYS12 SLA of a held-out year (default 2023).

Same OSSE as in training: truth = GLORYS12 SLA (zos - 2010-2019 mean), input = truth masked with a random daily nadir-like mask drawn from
the 2022 6-satellite mask list (as open_glorys12_data_sla), normalised with the mean/std of the training target (2010-01-01..2019-05-17).
Window 29 days: days t0-14..t0-1 observed, rest NaN. LT k = output channel 13+k, valid on t0+k-1. Weekly Wednesday starts.
Scored on lat -79..89, lon -179..179 (inference domain), RMSE in cm, global (unweighted and cos-lat weighted) and 3 boxes.
Usage: python osse_eval_glorys.py OUT.json [year] [unet|roll|both]
"""
import json, os, pickle, sys
from functools import partial
import numpy as np, pandas as pd, torch, xarray as xr
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models_UNet import UNet
from src.parts import ResBlock
from src.models import BilinAEPriorCost, BaseObsCost
from contrib.ocean4dvarnet_rollout.solver import DiffusionGradSolver, GradModelWithCondition
from contrib.ocean4dvarnet_rollout.unet import UNetModel

CK_UNET = ("/Odyssey/public/glorys/trainings/forecast_global_GPU_1patch_1y_SLA_UNet/base_forecast_global_softedge_fastrec_GPU_1patch_10y_SLA_UNet/"
           "checkpoints/val_mse=3.08068-epoch=204.ckpt")
CK_ROLL = ("/Odyssey/public/glorys/trainings/forecast_SLA_4dvarnet29_unet_rollout_OSSE_Nadir/forecast_SLA_4dvarnet29_unet_rollout_OSSE_Nadir/"
           "resume_ep178/checkpoints/val_mse=3.37574-epoch=304.ckpt")
TRAIN = "/Odyssey/public/glorys/reanalysis/glorys12_2010_2019_daily_sla_4th_float32.nc"
MASKS = "/Odyssey/public/glorys/obs_masks/global_obs_6sats_masks_2022.pickle"
FILES = {"2023": "/Odyssey/public/glorys/reanalysis/glorys12_2023_0.25deg_SLA.nc"}
BOXES = {"gulf_stream": (33, 43, -75, -50), "kuroshio": (30, 40, 140, 170), "agulhas": (-45, -25, 10, 45)}
DT, NOBS = 29, 14
dev = "cuda"


def sd_of(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    return {k[len("solver."):]: v for k, v in sd.items() if k.startswith("solver.")}


def load_unet():
    m = UNet(n_channels=DT, n_classes=DT, bilinear=True, block=partial(ResBlock), add_input=False)
    m.load_state_dict(sd_of(CK_UNET)); return m.to(dev).eval()


def load_roll():
    gm = UNetModel(in_channels=58, out_channels=DT, model_channels=32, num_res_blocks=2, attention_resolutions=[],
                   channel_mult=[1, 2, 4, 8], dims=2, dropout=0.0, use_checkpoint=True)
    s = DiffusionGradSolver(grad_mod=GradModelWithCondition(grad_model=gm, dropout=0.0, use_grad_norm=True), n_step=5, lr_grad=0.2,
                            input_grad_update="grad+state", std_init=0.0,
                            prior_cost=BilinAEPriorCost(dim_in=DT, dim_hidden=64, bilin_quad=False), obs_cost=BaseObsCost())
    s.load_state_dict(sd_of(CK_ROLL)); return s.to(dev).eval()


class B:  # minimal batch
    def __init__(self, x): self.input = x


def main():
    out = sys.argv[1]; year = sys.argv[2] if len(sys.argv) > 2 else "2023"; which = sys.argv[3] if len(sys.argv) > 3 else "both"
    tr = xr.open_dataset(TRAIN).sla.sel(time=slice("2010-01-01", "2019-05-17"))
    m, s = float(tr.mean(skipna=True)), float(tr.std(skipna=True)); print("train mean/std", m, s, flush=True)
    ds = xr.open_dataset(FILES[year]); sla = ds.sla.squeeze().load()
    if "latitude" in sla.dims: sla = sla.rename({"latitude": "lat", "longitude": "lon"})
    if "depth" in sla.dims: sla = sla.squeeze("depth")
    sla = sla.isel(lat=slice(40, None))  # lat -80..89.75, 680 rows as in training
    lat = sla.lat.values; lon = sla.lon.values; T = sla.values.astype(np.float32); print("grid", T.shape, lat[[0, -1]], lon[[0, -1]], flush=True)
    masks = np.array(pickle.load(open(MASKS, "rb")))[:, 40:] if np.array(pickle.load(open(MASKS, "rb"))).shape[1] == 720 else np.array(pickle.load(open(MASKS, "rb")))
    rng = np.random.RandomState(0); idx = rng.randint(0, len(masks), T.shape[0])
    X = np.where(np.isfinite(masks[idx]), T, np.nan).astype(np.float32)  # masked input per day
    print("obs fraction of ocean", np.isfinite(X).sum() / np.isfinite(T).sum(), flush=True)
    models = {}
    if which in ("unet", "both"): models["UNet-N"] = load_unet()
    if which in ("roll", "both"): models["4DVarNet-roll-ep304"] = load_roll()
    days = pd.DatetimeIndex(sla.time.values)
    starts = [d for d in pd.date_range(f"{year}-01-01", f"{year}-12-31") if d.dayofweek == 2 and d - pd.Timedelta(days=NOBS) >= days[0] and d + pd.Timedelta(days=6) <= days[-1]]
    print(len(starts), "starts", flush=True)
    # region masks (inference domain lat -79..89, lon -179..179)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dom = (LAT >= -79) & (LAT <= 89) & (LON >= -179) & (LON <= 179)
    regs = {"global": dom}
    for k, (a, b, c, d) in BOXES.items(): regs[k] = dom & (LAT >= a) & (LAT <= b) & (LON >= c) & (LON <= d)
    cosw = np.cos(np.deg2rad(LAT))
    acc = {mn: {r: np.zeros((7, 3)) for r in list(regs) + ["global_cos"] + [r + "_trk" for r in regs]} for mn in list(models) + ["zero"]}  # sum sq, sum w, n
    per_start = {mn: [] for mn in list(models)}
    for t0 in starts:
        i0 = int(np.where(days == t0)[0][0])
        win = X[i0 - NOBS:i0].copy(); x = np.full((DT,) + T.shape[1:], np.nan, np.float32); x[:NOBS] = win
        xn = torch.from_numpy((x - m) / s)[None].to(dev)
        for mn, mod in models.items():
            with torch.set_grad_enabled(mn != "UNet-N"):
                o = mod(B(xn))
            o = o.detach()[0].float().cpu().numpy() * s + m
            ps = []
            for k in range(7):
                tru = T[i0 + k]; err = o[NOBS + k] - tru; ok = np.isfinite(tru) & np.isfinite(err)
                for r, rm in regs.items():
                    sel = ok & rm; acc[mn][r][k] += [np.sum(err[sel] ** 2), sel.sum(), sel.sum()]
                sel = ok & dom; acc[mn]["global_cos"][k] += [np.sum(cosw[sel] * err[sel] ** 2), np.sum(cosw[sel]), sel.sum()]
                trk = np.isfinite(masks[idx[i0 + k]]) & ok
                for r, rm in regs.items():
                    sel = trk & rm; acc[mn][r + "_trk"][k] += [np.sum(err[sel] ** 2), sel.sum(), sel.sum()]
                sel = ok & dom
                ps.append(float(np.sqrt(np.mean(err[sel] ** 2))) * 100)
                if mn == list(models)[0]:  # zero forecast (climatology) once
                    for r, rm in regs.items():
                        sel2 = ok & rm; acc["zero"][r][k] += [np.sum(tru[sel2] ** 2), sel2.sum(), sel2.sum()]
                        sel2 = trk & rm; acc["zero"][r + "_trk"][k] += [np.sum(tru[sel2] ** 2), sel2.sum(), sel2.sum()]
            per_start[mn].append(ps)
        print(t0.date(), {mn: round(float(np.mean(per_start[mn][-1])), 2) for mn in models}, flush=True)
    res = {mn: {r: [float(np.sqrt(a[k, 0] / a[k, 1]) * 100) if a[k, 1] > 0 else None for k in range(7)] for r, a in d.items()} for mn, d in acc.items()}
    res["n_starts"] = len(starts); res["per_start_global_cm"] = per_start
    json.dump(res, open(out, "w"), indent=1)
    for mn in acc:
        for r in ("global", "global_trk", "gulf_stream_trk", "kuroshio_trk", "agulhas_trk"):
            if res[mn][r][0] is not None: print(mn, r, [round(v, 2) for v in res[mn][r]])


main()
