"""Data-backed v4.1 workflow: native coarse cells, matched fine-grid zooms, six targets.

The NPZ bundle is self-contained for meteorological maps. No interpolation,
synthetic fine detail, or model inference is performed by this renderer.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Ellipse
import numpy as np

INK, NAVY, BLUE, TEAL = '#17354c', '#0b3558', '#1769aa', '#118c91'
MUTED, LINE = '#60788b', '#bacbd8'
ZOOM = (-110., -102., 35., 41.5)
FIELDS = ('t2m','precip','ps','u10m','v10m','q2m')


def save_bundle(path, case):
    """Archive plotted physical fields and source labels, without object arrays."""
    from .evaluate_all_v4_1 import atomic_npz
    arrays = {k: np.asarray(case[k]) for k in ('hr','lat','lon')}
    if case.get('coarse') is not None: arrays['coarse'] = np.asarray(case['coarse'])
    for name in ('members_t2m','members_precip','prediction'):
        if case.get(name) is not None:arrays[name]=case[name]
    for name, source in case.get('native',{}).items():
        for key in ('values','lat','lon'): arrays[f'native_{name}_{key}'] = source[key]
    meta = {k:v for k,v in case.items() if k not in ('hr','lat','lon','coarse','members_t2m','members_precip','prediction','native')}
    arrays['metadata_json'] = np.asarray(json.dumps(meta))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    atomic_npz(path,**arrays)


def load_bundle(path):
    with np.load(path,allow_pickle=False) as f:
        case=json.loads(str(f['metadata_json']))
        case.update({k:f[k] for k in ('hr','lat','lon','coarse','members_t2m','members_precip','prediction') if k in f})
        case['native']={}
        for name in (*FIELDS,'wind_speed'):
            if f'native_{name}_values' in f:
                case['native'][name]={k:f[f'native_{name}_{k}'] for k in ('values','lat','lon')}
    return case


def from_evaluation(root, manifest):
    """Load the first predeclared saved case; no source archive access."""
    from .evaluate_all_v4_1 import digest
    selected=manifest['settings'].get('saved_cases',[])
    if not selected: raise ValueError('Map workflow needs saved case fields or --workflow-data bundle')
    root=Path(root); case_id=selected[0]
    record=json.loads((root/'cases'/f'{case_id}.json').read_text())
    item=record['saved_fields']['raw']; grid=manifest['saved_grid']
    if (record['run_id']!=manifest['run_id'] or item['path']!=f'fields/{case_id}_raw.npz'
            or grid['path']!='fields/grid.npz' or digest(root/item['path'])!=item['sha256']
            or digest(root/grid['path'])!=grid['sha256']):
        raise ValueError('Workflow map provenance mismatch')
    with np.load(root/item['path'],allow_pickle=False) as f, np.load(root/grid['path'],allow_pickle=False) as g:
        if str(f['run_id'])!=manifest['run_id'] or f['seeds'].tolist()!=record['seeds']:
            raise ValueError('Workflow member identity mismatch')
        native={name:{k:f[f'native_{name}_{k}'] for k in ('values','lat','lon')}
                for name in (*FIELDS,'wind_speed') if f'native_{name}_values' in f}
        ensemble=f['ensemble']
        return dict(hr=f['truth'],coarse=f['coarse'],members_t2m=ensemble[:2,0],members_precip=ensemble[:2,1],
            prediction=ensemble[0],lat=g['lat'],lon=g['lon'],
            native=native,time=record['time'],public_proxy=False,reference_label='HWT GEOS replay (~3 km)',
            native_label='GEOS-FP native 0.25° × 0.3125°',run_id=manifest['run_id'],state_shape=None,
            footer=f'{record["time"]} UTC | HWT reference and fixed raw members from the saved evaluation; same-grid metrics are reported separately.')


def centers_to_edges(values):
    values=np.asarray(values)
    if values.ndim!=1 or len(values)<2 or np.any(np.diff(values)<=0):
        raise ValueError('Native cell centers must increase along each axis')
    mid=(values[:-1]+values[1:])/2
    return np.r_[values[0]-(mid[0]-values[0]),mid,values[-1]+values[-1]-mid[-1]]


def crop(lon,lat,values,extent):
    """Subset native values; never reduce resolution or interpolate."""
    west,east,south,north=extent
    lon,lat,values=np.asarray(lon),np.asarray(lat),np.asarray(values)
    if lon.ndim==1:
        x=np.flatnonzero((lon>=west-1)&(lon<=east+1));y=np.flatnonzero((lat>=south-1)&(lat<=north+1))
    else:
        mask=(lon>=west-1)&(lon<=east+1)&(lat>=south-1)&(lat<=north+1)
        y,x=np.where(mask)
    if len(x)<2 or len(y)<2: raise ValueError('Map extent does not cover enough source cells')
    ys,xs=slice(y.min(),y.max()+1),slice(x.min(),x.max()+1)
    return (lon[xs],lat[ys],values[ys,xs]) if lon.ndim==1 else (lon[ys,xs],lat[ys,xs],values[ys,xs])


def zoom_limits(case, requested):
    lon,lat=case['lon'],case['lat']
    if (lon.min()<=requested[0]<requested[1]<=lon.max()
            and lat.min()<=requested[2]<requested[3]<=lat.max()):return tuple(requested)
    # Small synthetic domains and non-CONUS subsets retain a shared in-domain view.
    return (float(np.quantile(lon,.2)),float(np.quantile(lon,.8)),
            float(np.quantile(lat,.2)),float(np.quantile(lat,.8)))


def render(case, output, dpi=600, zoom=ZOOM, features=True):
    if case.get('public_proxy'):
        raise ValueError('HRRR demonstration maps are not accepted for the GEOS workflow; use saved project data or archived GEOS/v4.1 plots')
    try:
        import cartopy.crs as ccrs
    except ImportError as exc:
        raise ImportError("Map workflows require Cartopy: pip install -e '.[workflow]'") from exc
    from .evaluate_all_v4_1 import atomic_json,digest
    from .paper_figures_v4_1 import export,style
    style();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    pc=ccrs.PlateCarree()
    projection=ccrs.LambertConformal(central_longitude=-97.5,central_latitude=38.5,standard_parallels=(33,45))
    full=(-125,-66.5,24,50.5);zoom=zoom_limits(case,zoom)
    fig=plt.figure(figsize=(22,11),facecolor='#f7fafc')
    fig.text(.5,.961,'MERRA-Flow: GENERATIVE CONUS DOWNSCALING',ha='center',fontsize=23,weight='bold',color=NAVY)
    fig.text(.5,.928,'Native GEOS-FP (~25 km)  →  variable-specific conditional flow matching  →  six fields on the HWT (~3 km) grid',
             ha='center',fontsize=11,color=MUTED)
    lefts=[.018,.263,.508,.753];width=.229
    def text(x,y,s,fs=8,color=INK,**kw):return fig.text(x,y,s,fontsize=fs,color=color,**kw)
    def arrow(x1,y1,x2,y2):
        fig.patches.append(FancyArrowPatch((x1,y1),(x2,y2),transform=fig.transFigure,
            arrowstyle='-|>',mutation_scale=13,lw=1.3,color=NAVY))
    def frame(left,label,title,sub):
        fig.patches.append(Rectangle((left,.090),width,.785,transform=fig.transFigure,
            facecolor='white',edgecolor=LINE,lw=.85,zorder=-10))
        text(left+.012,.848,label,12,BLUE,weight='bold');text(left+.035,.848,title,10.5,weight='bold')
        fig.lines.append(plt.Line2D([left+.012,left+width-.012],[.830,.830],transform=fig.transFigure,color=BLUE,lw=1.3))
        text(left+.012,.807,sub,7.5,MUTED)
    boundaries={}
    if features:
        import gzip
        from shapely import from_wkb
        if 'boundaries' not in case:
            asset=Path(__file__).parent/'assets'/'conus_boundaries_50m.json.gz'
            with gzip.open(asset,'rt') as f: case['boundaries']=json.load(f)
        boundaries={name:[from_wkb(g) for g in geoms] for name,geoms in case['boundaries']['geometries'].items()}
    def map_panel(bounds,source,title,unit,cmap,limits,extent=full,cells=False,mark=False):
        ax=fig.add_axes(bounds,projection=projection)
        lon,lat,z=crop(source['lon'],source['lat'],source['values'],extent)
        opts=dict(cmap=cmap,vmin=limits[0],vmax=limits[1],rasterized=True,transform=pc)
        if lon.ndim==1:
            im=ax.pcolormesh(centers_to_edges(lon),centers_to_edges(lat),z,shading='flat',
                edgecolors=(0,0,0,.16) if cells else 'none',linewidth=.2 if cells else 0,**opts)
        else:
            im=ax.pcolormesh(lon,lat,z,shading='nearest',edgecolors='none',**opts)
        ax.set_extent(extent,crs=pc)
        if features:
            for name,geoms in boundaries.items():
                ax.add_geometries(geoms,pc,facecolor='none',edgecolor=INK if name!='states' else '#42576a',
                                  linewidth=.4 if name=='coast' else .28)
        if mark:
            w,e,s,n=zoom;ax.plot([w,e,e,w,w],[s,s,n,n,s],transform=pc,color='#062f4b',lw=1.5,zorder=10)
            ax.text(w,n,' B ',transform=pc,color=INK,fontsize=8,weight='bold',va='bottom',bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
        ax.set_title(title,loc='left',fontsize=8.2,color=INK,pad=5,weight='semibold')
        x,y,w,h=bounds
        cax=fig.add_axes([x+.005,y-.017,w-.010,.007])
        cb=fig.colorbar(im,cax=cax,orientation='horizontal',extend='both')
        cb.ax.tick_params(labelsize=6,length=2,pad=1);cb.set_label(unit,fontsize=6.5,labelpad=1)
        return ax
    hr=case['hr'];lat,lon=case['lat'],case['lon'];native=case.get('native',{})
    def fine(c,values=None):return dict(values=hr[c] if values is None else values,lat=lat,lon=lon)
    def coarse(name,c):
        if name in native:return native[name]
        if case.get('coarse') is None:raise ValueError(f'Missing native or saved coarse {name}')
        return dict(values=case['coarse'][c],lat=lat,lon=lon)
    native_t=coarse('t2m',0);native_t=dict(native_t,values=native_t['values']-273.15)
    fine_t=fine(0,hr[0]-273.15)
    tlow,thigh=np.quantile(fine_t['values'],[.02,.98])
    L=lefts[0];frame(L,'A','COARSE WEATHER INPUTS','Current + previous hour; static and time features')
    input_note='Native 0.25° × 0.3125° cells' if 't2m' in native else 'Interpolated coarse input; native grid unavailable'
    text(L+.016,.773,input_note,8,BLUE,weight='bold')
    map_panel((L+.015,.515,width-.030,.235),native_t,'2 m temperature','°C','RdYlBu_r',(tlow,thigh),mark=True)
    rain=coarse('precip',1)
    map_panel((L+.015,.224,width-.030,.224),rain,'Precipitation input','mm h$^{-1}$','YlGnBu',
              (0,max(float(np.quantile(rain['values'],.999)),.1)),mark=True)
    text(L+.016,.165,'Six weather fields at t − 1 h and t',9,weight='bold')
    text(L+.016,.135,'Local 192² patches + broad 576² context\nTerrain, land/sea, coordinates, cell area, time',8,linespacing=1.6)
    L=lefts[1];frame(L,'B','SAME REGION, TWO GRIDS','Identical extent and color scale; no display interpolation')
    _,_,zt=crop(fine_t['lon'],fine_t['lat'],fine_t['values'],zoom)
    _,_,zc=crop(native_t['lon'],native_t['lat'],native_t['values'],zoom)
    limits=(float(min(np.quantile(zt,.01),np.quantile(zc,.01))),float(max(np.quantile(zt,.99),np.quantile(zc,.99))))
    if limits[1]<=limits[0]:limits=(limits[0],limits[0]+1)
    map_panel((L+.018,.525,width-.036,.233),native_t,
              '~25 km: native cells' if 't2m' in native else 'Coarse information (on the HWT grid)',
              '°C','RdYlBu_r',limits,zoom,cells='t2m' in native)
    map_panel((L+.018,.231,width-.036,.233),fine_t,
              '~3 km GEOS replay reference (HWT)',
              '°C','RdYlBu_r',limits,zoom)
    text(L+.018,.165,'Temporal pairing and training-only statistics',8.7,weight='bold')
    text(L+.018,.132,'States: midpoint | Rain: ¼ P₋₃₀ + ½ P₀ + ¼ P₊₃₀\nChronological splits with gaps; packed training data',7.5,linespacing=1.7)
    generated=case.get('prediction')
    L=lefts[2];frame(L,'C','GENERATED ~3 km FIELDS' if generated is not None else 'REFERENCE FIELDS',
        'Actual first raw model member' if generated is not None else case['reference_label'])
    output_fields=generated if generated is not None else hr
    specs=[(output_fields[0]-273.15,'2 m temperature','°C','RdYlBu_r'),
           (output_fields[1],'Hourly mean rain rate','mm h$^{-1}$','YlGnBu'),
           (output_fields[2]*.01,'Surface pressure','hPa','Spectral_r'),
           (output_fields[3],'10 m zonal wind','m s$^{-1}$','RdBu_r'),
           (output_fields[4],'10 m meridional wind','m s$^{-1}$','RdBu_r'),
           (output_fields[5]*1000,'2 m specific humidity','g kg$^{-1}$','YlGnBu')]
    mw=(width-.044)/2
    for c,(z,title,unit,cmap) in enumerate(specs):
        lo,hi=np.quantile(z,[.02,.98])
        if c==1:lo,hi=0,max(float(np.quantile(z,.999)),.1)
        if c in (3,4):hi=max(abs(lo),abs(hi),.1);lo=-hi
        if hi<=lo:hi=lo+1e-5
        map_panel((L+.012+(c%2)*(mw+.020),.633-(c//2)*.206,mw,.128),fine(c,z),title,unit,cmap,(lo,hi))
    text(L+.013,.165,'Variable-specific generative coordinates',8.8,weight='bold')
    text(L+.013,.123,'Rain: direct √(1 + P/P₀) − 1; P₀ = 1 mm h⁻¹\nT, pressure, winds: frozen-regression residuals\nHumidity: coarse-field residual',7.4,linespacing=1.6)
    L=lefts[3];frame(L,'D','REGRESSION → GENERATION','Frozen conditioner + local and broad-context U-Net')
    text(L+.014,.770,'CONDITIONAL FLOW MATCHING',9,BLUE,weight='bold')
    xs=[L+.034,L+.113,L+.192]
    for x,s,col in zip(xs,['$z_0$','$z_t$','$v_\\theta$'],[BLUE,TEAL,NAVY]):
        fig.patches.append(Ellipse((x,.717),.056,.052,transform=fig.transFigure,facecolor='white',edgecolor=col,lw=1.5))
        text(x,.717,s,13,col,ha='center',va='center')
    arrow(xs[0]+.028,.717,xs[1]-.028,.717);arrow(xs[1]+.028,.717,xs[2]-.028,.717)
    text(xs[0],.675,'Gaussian noise',7,MUTED,ha='center');text(xs[1],.675,'Conditional path',7,MUTED,ha='center')
    text(xs[2],.675,'U-Net velocity',7,MUTED,ha='center')
    text(L+width/2,.640,'$z_t=(1-t)z_0+t z_1$   |   target: $z_1-z_0$',9,ha='center')
    # Encoder/decoder motif and explicit halo supervision replace the obsolete core-only loss.
    for i,y in enumerate([.587,.554,.521]):
        w=.048-i*.009
        for x,col in [(L+.035,BLUE),(L+.155,TEAL)]:
            fig.patches.append(Rectangle((x,y),w,.021,transform=fig.transFigure,color=col,alpha=1-i*.18))
        fig.lines.append(plt.Line2D([L+.083,L+.155],[y+.01,y+.01],transform=fig.transFigure,color=LINE,lw=.7))
    fig.patches.append(Rectangle((L+.098,.484),.036,.022,transform=fig.transFigure,color=NAVY))
    text(L+width/2,.459,'Residual blocks + broad-context conditioning',7.2,MUTED,ha='center')
    text(L+.014,.423,'AREA- AND CHANNEL-WEIGHTED VELOCITY LOSS',8,weight='bold')
    fig.patches.append(Rectangle((L+.018,.331),.067,.067,transform=fig.transFigure,color='#e5eff5',ec=BLUE,lw=1))
    fig.patches.append(Rectangle((L+.029,.342),.045,.045,transform=fig.transFigure,color='#76b9d6',ec=NAVY,lw=.7))
    text(L+.0515,.362,'128²',7,'white',ha='center',weight='bold')
    text(L+.100,.376,'192² input = 128² core + 32-pixel halo',7.4)
    text(L+.100,.349,'Loss supervises core and halo',7.4,TEAL,weight='bold')
    text(L+.014,.296,'SYNCHRONIZED TILED ENSEMBLE SAMPLING',8.3,BLUE,weight='bold')
    text(L+.014,.279,'Blend tile velocities → shared-state Heun update\nIndependent seeds → multiple conditional members',7.8,linespacing=1.7,va='top')
    if case.get('members_t2m') is not None:
        mw=(width-.048)/2
        for i,z in enumerate(case['members_t2m'][:2]):
            map_panel((L+.015+i*(mw+.018),.139,mw,.080),fine(0,z-273.15),f'Temperature member {i+1}',
                      '°C','RdYlBu_r',limits,zoom)
    else:
        for i in range(3):
            fig.patches.append(Rectangle((L+.028+i*.057,.159+i*.008),.058,.055,transform=fig.transFigure,facecolor='#e5f2f4',edgecolor=TEAL,lw=1))
        text(L+.020,.135,'Ensemble outputs are schematic here; no model was run.',7,MUTED)
    text(L+.014,.104,'Raw / spectral: paired members, no retraining',7.8,TEAL,weight='bold')
    for i in range(3):arrow(lefts[i]+width+.001,.49,lefts[i+1]-.001,.49)
    text(.5,.054,case['footer'],8,MUTED,ha='center')
    text(.5,.030,'Temperature zooms share extent and scale. Coarse-cell boundaries are shown only for native data. Fine maps retain source cells.',7.7,MUTED,ha='center')
    export(fig,out,'fig01_workflow',dpi)
    save_bundle(out/'workflow_data.npz',case)
    atomic_json(out/'workflow_provenance.json',dict(renderer_sha256=digest(__file__),data_sha256=digest(out/'workflow_data.npz'),
        zoom=list(zoom),native_variables=sorted(native),native_t2m_available='t2m' in native,features=features,
        public_proxy=case.get('public_proxy',False),time=case['time'],dpi=dpi,
        note='Native coarse cells and unmodified high-resolution values; high-resolution reference and predictions are distinct project GEOS products.'))
    return out
