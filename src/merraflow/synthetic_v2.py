"""Tiny explicit synthetic v2 fixture. Never a meteorological skill benchmark."""
from pathlib import Path
import numpy as np
import xarray as xr
import yaml
from .synthetic import make_synthetic


def make_synthetic_v2(root, template):
    root = Path(root)
    # Reuse only raw fixture generation; v2 archives/checkpoints remain distinct.
    source = make_synthetic(root, template)
    cfg = yaml.safe_load(source.read_text())
    source.unlink()
    cfg['version'] = 'v2'
    cfg['data']['prepared'] = str(root/'prepared_v2')
    cfg['train'].pop('epochs', None)
    cfg['train'].update(output=str(root/'run_v2'), regression_epochs=2, flow_epochs=2,
                        ema_decay=.5, calibration_batches=2)
    cfg['inference'].update(output=str(root/'predictions_v2'), members=2)
    cfg['patch'].update(context_size=16, context_scale=2, sampling_stride=8)
    cfg['model'].update(blocks_per_level=2, attention_heads=2)
    path = next((root/'hr').glob('hwt_30mn_slv_LCC/*/*.nc4'))
    with xr.open_dataset(path) as ds:
        lat, lon = ds.lats.values.squeeze(), ds.lons.values.squeeze()
    h, w = lat.shape
    yy, xx = np.mgrid[:h, :w]
    land = (xx < w*.7).astype('float32')
    lake = (((xx-w*.35)**2+(yy-h*.5)**2) < 9).astype('float32')
    land *= 1-lake
    static_path = root/'surface_v2.nc'
    xr.Dataset({name: (('Ydim', 'Xdim'), value) for name, value in
                [('land_fraction', land), ('lake_fraction', lake), ('lats', lat), ('lons', lon)]}).to_netcdf(static_path, engine='h5netcdf')
    cfg['data']['static'] = dict(path=str(static_path), land='land_fraction', lake='lake_fraction', lat='lats', lon='lons')
    result = root/'config_v2.yaml'
    result.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return result
