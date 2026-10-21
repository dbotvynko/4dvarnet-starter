"""SLA input loader of the OSE pipeline as it was in August 2025 (4dvarnet commit 5b5c485, contrib/data_loading/data.py,
load_ose_data_with_tgt_mask_SLA), i.e. the version that produced the 2023 forecasts of the OSE UNet trained on 2017-2022
(rec/glorys4_global_1patch_SLA_UNet_Filtered_OSE_DUACS_losses_2017-2022, 6 August 2025). Selected through the model-config
key `ose_input_loader`.

Changes from the August 2025 version, so that the same model can run on 2024 with the CLS inputs:
- the year is a parameter (it was hard-coded to 2023);
- if `tgt_path_l3_data` is set, it is read as the input file (as in load_ose_data_with_tgt_mask_SLA_2025);
- CLS grids (648 latitudes -80..81.75, longitudes -186..185.75) are cut to -180..179.75 and keep their latitudes; the
  target map is taken on that grid. The 720-latitude NRT grid follows the August 2025 path unchanged.
The target is only a constant map (DUACS of 2019-01-20) used for the grad mask and the normalisation; the normalisation is
fixed with `datamodule.norm_stats` in the model config so that it does not depend on the input grid.
"""
import xarray as xr
import numpy as np
from src.data import TrainingItem
import pandas as pd


def load_ose_data_with_tgt_mask_SLA_aug2025(path, tgt_path, tgt_path_not_glorys, tgt_path_l3_data, variable, year):
    """
        path: path to the gridded ose input (NRT pipeline file)
        tgt_path: complete DUACS SLA maps containing the day 2019-01-20
        tgt_path_l3_data: if set, input file used instead of `path` (CLS gridded files)
        variable: input variable to load
        year: year of the input to keep
    """
    if(len(tgt_path_not_glorys) != 0):
        tgt_path = tgt_path_not_glorys
    if(len(tgt_path_l3_data) != 0):
        path = tgt_path_l3_data

    print(f'tgt_path is {tgt_path}')
    print(f'path is {path}')
    ds_mask = xr.open_dataset(tgt_path)
    ds = xr.open_dataset(path)

    if 'latitude' in list(ds_mask.dims):
        ds_mask = ds_mask.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds.variables):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'latitude' in list(ds.dims):
        ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    if 'sla' in list(ds_mask.variables):
        ds_mask = ds_mask.rename({'sla':variable})

    ds['time'] = pd.to_datetime(ds['time'].values)
    ds = ds.sel(time=ds['time'].dt.year == int(year))
    print(ds)

    print('VARIABLE IS')
    print(variable)

    ds_mask = ds_mask.sel(time='2019-01-20')[variable]
    if ds.sizes['lat'] == 720:
        # August 2025 path (NRT grid -90..89.75, -180..179.75)
        ds_mask = ds_mask.interp(lat=ds['lat'], lon=ds['lon']).expand_dims(time=ds.time)
        ds = ds.isel(lat = np.arange(40, 720, 1))
        ds_mask = ds_mask.isel(lat = np.arange(40, 720, 1))
        ds_mask = ds_mask.assign_coords(ds.coords)
    else:
        # CLS grid: keep its latitudes, cut the longitudes to the DUACS grid; interp as in the August 2025 path, so that the
        # coastal cells set to NaN by the linear interpolation are the same as in 2023
        ds = ds.sel(lon = slice(-180, 179.75))
        ds_mask = ds_mask.interp(lat=ds['lat'], lon=ds['lon']).expand_dims(time=ds.time)
        ds_mask = ds_mask.assign_coords(ds.coords)
    print(ds_mask[0].shape)

    ds = (
        ds
        .assign(
            input=ds[variable],
            tgt= ds_mask
        )
    )

    return (
        ds[[*TrainingItem._fields]]
        .transpose("time", "lat", "lon")
        .to_array()
    )
