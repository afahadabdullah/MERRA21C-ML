"""Tiny deterministic archive with real collection names; never a skill benchmark."""
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import xarray as xr
import yaml
from .physics import native_groups


def make_synthetic(root, template):
    root = Path(root).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Synthetic fixture output directory must be new or empty')
    root.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(template)
    d = cfg['data']
    d['synthetic'] = True
    d.update(lowres_lcc_root=str(root/'lr'), native_root=str(root/'native'), highres_root=str(root/'hr'), prepared=str(root/'prepared'),
             start='2025-08-31T22:30:00', end='2025-09-01T03:30:00', stats_stride=2)
    d['splits'] = {'train': ['2025-08-31', '2025-09-01'], 'val': ['2025-09-01', '2025-09-01T02:00:00'], 'test': ['2025-09-01T02:00:00', '2025-09-01T04:00:00']}
    cfg['patch'].update(size=16, halo=4, stride=12, samples_per_epoch=8)
    cfg['model'].update(base_channels=8, channel_mult=[1, 2, 2], time_dim=32, activation_checkpointing=True)
    cfg['train'].update(device='cpu', epochs=2, batch_size=2, accumulate=2, workers=0, precision='fp32', warmup_steps=0, val_batches=2, output=str(root/'run'))
    cfg['inference'].update(members=3, steps=2, output=str(root/'predictions'))
    h, w = 36, 44
    yy, xx = np.mgrid[:h, :w]
    lat = (33+yy*.03+.03*np.sin(xx/15)).astype('float32')
    lon = (-101+xx*.04+.02*np.sin(yy/10)).astype('float32')
    nlat, nlon = np.linspace(32.7, 34.4, 7), np.linspace(-101.3, -98.9, 9)
    groups, source = native_groups(nlat, nlon, lat, lon)
    area = (8.7e6*(1+.05*yy/h)).astype('float32')
    elevation = (800+700*np.sin(xx/12)*np.cos(yy/13)).astype('float32')
    for k in range(6):
        t = datetime.fromisoformat(d['start'])+timedelta(hours=k)
        end = t+timedelta(minutes=30)
        tag, etag = t.strftime('%Y%m%d_%H%M'), end.strftime('%Y%m%d_%H%M')
        phase = k*.3
        ny, nx = np.mgrid[:len(nlat), :len(nlon)]
        native_pr = np.maximum(np.sin(nx*.8+phase)+np.cos(ny*.9), 0).astype('float32')*3
        native = xr.Dataset({'PRECTOT': (('time', 'lat', 'lon'), native_pr[None]/3600, {'units': 'kg m-2 s-1'})}, coords={'time': [np.datetime64(t)], 'lat': nlat, 'lon': nlon})
        ref = native_pr.ravel()[source][groups]
        temp = (289-6*elevation/1000+np.sin(xx/18)+phase).astype('float32')
        u, v = np.full((h, w), 4+phase, dtype='float32'), np.full((h, w), 2, dtype='float32')
        pressure = (100000*np.exp(-elevation/8500)).astype('float32')
        lr_fields = {'T2M': (temp, 'K'), 'PRECTOT': (ref/3600, 'kg m-2 s-1'), 'PS': (pressure, 'Pa'),
                     'U10M': (u, 'm s-1'), 'V10M': (v, 'm s-1'), 'QV2M': (temp*0+.008, 'kg kg-1'),
                     'SLP': (pressure*0+101000, 'Pa'), 'TQV': (temp*0+20, 'kg m-2'),
                     'OMEGA500': (np.sin(xx/8).astype('float32'), 'Pa s-1'),
                     'PRECCON': (ref/7200, 'kg m-2 s-1'), 'PRECLSC': (ref/7200, 'kg m-2 s-1')}
        coords = {'time': [np.datetime64(t)], 'Ydim': np.arange(h), 'Xdim': np.arange(w)}
        lr = xr.Dataset({n: (('time', 'Ydim', 'Xdim'), a[None], {'units': unit}) for n, (a, unit) in lr_fields.items()}, coords=coords)
        hr_fields = {'TMP_2M': (temp+np.sin(xx*1.7)*.8, 'K'), 'PRES_SFC': (pressure+40*np.cos(yy), 'Pa'),
                     'UGRD_10M': (u+.7*np.sin(xx), 'm s-1'), 'VGRD_10M': (v+.5*np.cos(yy), 'm s-1'),
                     'HGT_SFC': (elevation*9.80665, 'm2 s-2'), 'AREA': (area, 'm2'),
                     'lats': (lat, 'degrees_north'), 'lons': (lon, 'degrees_east')}
        hr = xr.Dataset({n: (('time', 'Ydim', 'Xdim'), a[None].astype('float32'), {'units': unit}) for n, (a, unit) in hr_fields.items()}, coords=coords)
        precip = ref*(1.25+.8*np.sin(xx*1.3)**2)+.1*(np.sin(yy)>0)
        acc = xr.Dataset({'APCP': (('time', 'Ydim', 'Xdim'), precip[None].astype('float32'), {'units': 'mm'})}, coords={**coords, 'time': [np.datetime64(end)]})
        acc['time_bounds'] = (('time', 'bounds'), [[np.datetime64(t-timedelta(minutes=30)), np.datetime64(end)]])
        acc.time.attrs['bounds'] = 'time_bounds'
        paths = [(native, root/'native'/t.strftime('Y%Y/M%m')/f'f5295_fp.tavg1_2d_flx_Nx.{tag}z.nc4'),
                 (lr, root/'lr'/t.strftime('%Y%m')/f'f5295_fp.lowres_lcc_1hr.{tag}z.nc4'),
                 (hr, root/'hr'/'hwt_30mn_slv_LCC'/t.strftime('%Y%m')/f'Feature-c2160_L137.hwt_30mn_slv_LCC.{tag}z.nc4'),
                 (acc, root/'hr'/'hwt_01hr_acc_LCC'/end.strftime('%Y%m')/f'Feature-c2160_L137.hwt_01hr_acc_LCC.{etag}z.nc4')]
        for ds, path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            if 'time_bounds' in ds:
                ds.time.encoding.update(units='minutes since 1970-01-01', calendar='proleptic_gregorian')
                ds.time_bounds.encoding.update(units='minutes since 1970-01-01', calendar='proleptic_gregorian')
            ds.to_netcdf(path, engine='h5netcdf')
    path = root/'config.yaml'
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
