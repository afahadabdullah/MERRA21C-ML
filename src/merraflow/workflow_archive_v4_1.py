"""Actual GEOS-FP → GEOS replay → v4.1 predictions, from archived vector maps.

Reuses original, identically scaled PDF map panels without recovering or inventing
meteorological values. Uses the same map-based layout with a merged flow/output
panel for both the README and paper figure. Saved numerical evaluation bundles remain the preferred input for
future complete-test figures.
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse

from .evaluate_all_v4_1 import atomic_json,digest

INK,NAVY,BLUE,TEAL,MUTED,LINE='#17354c','#0b3558','#1769aa','#118c91','#60788b','#bacbd8'
FIELDS=('t2m','precip','ps','u10m','v10m','q2m')
INPUT_VARIABLES=dict(
    weather=['PRECTOT','U10M','V10M','TQV','QV2M','SLP'],history_hours=[-1,0],
    current_coarse_baseline=['T2M','PRECTOT','PS','U10M','V10M'],
    static=['elevation','sin(latitude)','cos(latitude)','sin(longitude)','cos(longitude)',
            'log(cell_area)','land_fraction','lake_fraction','signed_water_boundary_distance',
            'grid_y_terrain_slope','grid_x_terrain_slope','ocean_fraction','lake_fraction_known'],
    temporal=['sin(annual_phase)','cos(annual_phase)','sin(UTC_phase)','cos(UTC_phase)',
              'sin(local_solar_phase)','cos(local_solar_phase)'])
# Same normalized geographic window in both archived axes: 3x linear zoom.
# Coordinates are fractions from the upper-left of the original map panel.
ZOOM_WINDOW=(1/3, .55-1/6, 2/3, .55+1/6)


def colorbar_spec(page,field):
    """Keep the source gradient and exact tick positions; retypeset sparse labels.

    The source PDFs outline each colorbar, including its extensions. Tick marks
    and numeric labels are vector objects, so no weather values are recovered
    from raster maps. Units and negative wind signs follow the source axes.
    """
    drawings=page.get_drawings()
    outlines=[d['rect'] for d in drawings if d['rect'].y0>page.rect.height-75
              and d['rect'].width>page.rect.width*.35 and 5<d['rect'].height<30]
    if not outlines:raise ValueError('Cannot identify original colorbar outline')
    clip=max(outlines,key=lambda r:r.width)
    positions=sorted({round(d['rect'].x0,4) for d in drawings
        if d['rect'].width<.01 and 2<d['rect'].height<5 and abs(d['rect'].y0-clip.y1)<.2})
    labels=sorted([w for w in page.get_text('words')
        if clip.y1-.1<w[1]<clip.y1+4 and w[4].replace('.','',1).isdigit()],key=lambda w:w[0])
    if len(labels)!=len(positions):raise ValueError('Colorbar ticks and labels do not match')
    values=[float(w[4]) for w in labels]
    if field in ('u10m','v10m'):
        values=[-v if i<len(values)//2 else v for i,v in enumerate(values)]
    indices={'t2m':(1,4,7),'precip':(0,5,9),'ps':(1,5),
             'u10m':(1,2,3),'v10m':(2,4,6),'q2m':(0,3,6)}[field]
    return tuple(clip),[((positions[i]-clip.x0)/clip.width,f'{values[i]:g}') for i in indices]


def source_clip(page,column,window=None):
    """Locate original equal-extent map axes; exclude old headings and scale labels."""
    images=[i for i in page.get_image_info() if i['width']>100 and i['height']>100]
    if len(images)<4:raise ValueError('Expected four original map images')
    first=min(images,key=lambda i:i['bbox'][0])['bbox']
    labels=('GEOS-FP 0.25°','Truth 2 km','One member','Ensemble mean (8)')
    matches=page.search_for(labels[column])
    if len(matches)!=1:raise ValueError(f'Cannot identify original panel: {labels[column]}')
    center=(matches[0].x0+matches[0].x1)/2
    width=first[2]-first[0]
    clip=(center-width/2,first[1],center+width/2,first[3])
    if window is None:return clip
    x0,y0,x1,y1=window
    if not (0<=x0<x1<=1 and 0<=y0<y1<=1):raise ValueError('Crop must lie inside the source map')
    height=clip[3]-clip[1]
    return (clip[0]+x0*width,clip[1]+y0*height,clip[0]+x1*width,clip[1]+y1*height)


def compose(base,output,placements,dpi):
    import pymupdf
    doc=pymupdf.open(base);page=doc[0];w,h=page.rect.width,page.rect.height
    opened={};records=[]
    try:
        for placement in placements:
            source,column,bounds=placement[:3]
            window=placement[3] if len(placement)>3 else None
            source=Path(source)
            if source not in opened:opened[source]=pymupdf.open(source)
            src=opened[source][0]
            if column=='colorbar':clip=window
            elif column=='bar':clip=(src.rect.width*.25,src.rect.height-68,src.rect.width*.78,src.rect.height)
            else:clip=source_clip(src,column,window)
            x,y,dx,dy=bounds
            rect=pymupdf.Rect(x*w,(1-y-dy)*h,(x+dx)*w,(1-y)*h)
            page.show_pdf_page(rect,opened[source],0,clip=pymupdf.Rect(clip),keep_proportion=column!='colorbar')
            records.append(dict(source=str(source),source_sha256=digest(source),column=column,
                                clip_points=list(clip),destination_fraction=list(bounds),
                                map_window_fraction=window if isinstance(column,int) else None))
        path=Path(output);doc.save(path,garbage=4,deflate=True)
        pix=page.get_pixmap(matrix=pymupdf.Matrix(dpi/72,dpi/72),alpha=False)
        pix.save(path.with_suffix('.png'))
        path.with_suffix('.svg').write_text(page.get_svg_image(text_as_path=False))
    finally:
        for source in opened.values():source.close()
        doc.close()
    return records


def make(presentation='present_last_v4_1',case_id='20260307_1930',output='output/figures/workflow_geos_v4_1',dpi=300):
    from .paper_figures_v4_1 import style
    style();root=Path(presentation);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((root/'metrics.json').read_text())
    entry=next(e for e in metadata['cases'] if e['id']==case_id)
    if metadata['members']!=8:raise ValueError('Archive compositor expects the documented eight-member presentation')
    folder=root/'cases'/case_id
    sources={f:folder/f'conus_{f}.pdf' for f in FIELDS}
    zoom=folder/'zoomA_t2m.pdf'
    for p in [*sources.values(),zoom]:
        if not p.is_file():raise FileNotFoundError(p)
    placements=[]
    # At the paper's 7.2-inch display width, 14 pt here prints at 6.3 pt.
    # Method details and run provenance belong in the caption and sidecar.
    fig=plt.figure(figsize=(16,9),facecolor='#f7fafc')
    def text(x,y,s,fs=14,color=INK,**kw):return fig.text(x,y,s,fontsize=fs,color=color,**kw)
    def arrow(x1,y1,x2,y2):
        fig.patches.append(FancyArrowPatch((x1,y1),(x2,y2),transform=fig.transFigure,
            arrowstyle='-|>',mutation_scale=16,lw=1.6,color=NAVY))
    def frame(left,label,title,sub,panel_width=.229):
        fig.patches.append(Rectangle((left,.080),panel_width,.785,transform=fig.transFigure,
            facecolor='white',edgecolor=LINE,lw=.85,zorder=-10))
        text(left+.012,.831,label,17,BLUE,weight='bold')
        text(left+.036,.831,title,16,weight='bold')
        fig.lines.append(plt.Line2D([left+.012,left+panel_width-.012],[.814,.814],transform=fig.transFigure,color=BLUE,lw=1.3))
        text(left+.012,.778,sub,14,MUTED)
    def image(source,col,bounds,title,window=None,fs=16):
        placements.append((source,col,bounds,window))
        text(bounds[0],bounds[1]+bounds[3]+.013,title,fs,weight='semibold')
    def bar(source,field,bounds):
        import pymupdf
        with pymupdf.open(source) as doc:clip,ticks=colorbar_spec(doc[0],field)
        placements.append((source,'colorbar',bounds,clip))
        x,y,w,h=bounds
        for fraction,label in ticks:text(x+fraction*w,y-.009,label,14,ha='center',va='top')
    text(.5,.952,'MERRA21C-ML: GENERATIVE CONUS DOWNSCALING',26,NAVY,ha='center',weight='bold')
    text(.5,.903,'GEOS-FP (~25 km)  →  Conditional flow matching  →  GEOS grid (~3 km)',18,MUTED,ha='center')
    lefts=[.018,.263,.508,.753];width=.229
    L=lefts[0];frame(L,'A','COARSE INPUTS','GEOS-FP · ~25 km')
    # Representative native predictor maps, stacked schematically. The full
    # input inventory below distinguishes six raw predictors from five baselines.
    for i,f in enumerate(('precip','q2m','u10m')):
        placements.append((sources[f],0,(L+.012+i*.018,.608-i*.023,.167,.151),None))
    text(L+.013,.529,'Weather at t − 1 h and t',14,weight='bold')
    for y,label in zip((.502,.477,.452,.427,.402),(
        'Rain rate · PRECTOT','10 m winds · U10M, V10M','Column vapor · TQV',
        '2 m humidity · QV2M','Sea-level pressure · SLP')):text(L+.013,y,label,14)
    text(L+.013,.363,'Coarse baselines at t',14,weight='bold')
    text(L+.013,.335,'T2M, PRECTOT, U10M, V10M',14)
    text(L+.013,.310,'Surface pressure · PS',14)
    text(L+.013,.273,'Static fields',14,weight='bold')
    for y,label in zip((.247,.222,.197,.172),(
        'Elevation · lat/lon · cell area','Land/lake/ocean fractions',
        'Water distance · x/y slopes','Lake-availability mask')):text(L+.013,y,label,14)
    text(L+.013,.135,'Time (sine/cosine)',14,weight='bold')
    text(L+.013,.110,'Annual · UTC · local solar',14)
    L=lefts[1];frame(L,'B','RESOLUTION','Same extent · same scale')
    image(zoom,0,(L+.018,.505,width-.036,.233),'GEOS-FP · ~25 km',window=ZOOM_WINDOW)
    bar(zoom,'t2m',(L+.018,.479,width-.036,.011))
    image(zoom,1,(L+.018,.188,width-.036,.223),'GEOS replay · ~3 km',window=ZOOM_WINDOW)
    bar(zoom,'t2m',(L+.018,.165,width-.036,.011))
    text(L+.018,.103,'Temperature (°C) · 3× zoom',14,weight='bold')
    # Merge the former generated-field panel into an expanded model panel.
    L=lefts[2]
    frame(L,'C','FLOW MATCHING + ENSEMBLE GENERATION',
          'Frozen regression · local patches + broad context',panel_width=.474)
    xs=[.558,.666,.774]
    for x,symbol,col in zip(xs,['$z_0$','$z_t$','$v_\\theta$'],[BLUE,TEAL,NAVY]):
        fig.patches.append(Ellipse((x,.699),.074,.068,transform=fig.transFigure,
            facecolor='white',edgecolor=col,lw=1.7))
        text(x,.699,symbol,23,col,ha='center',va='center')
    arrow(xs[0]+.037,.699,xs[1]-.037,.699)
    arrow(xs[1]+.037,.699,xs[2]-.037,.699)
    for x,label in zip(xs,['Noise','Flow state','Velocity']):
        text(x,.641,label,16,ha='center')
    text(.666,.593,'$z_t=(1-t)z_0+t z_1$',18,ha='center')
    text(.666,.540,'Conditional U-Net',19,ha='center',weight='bold')
    for i,y in enumerate([.481,.435,.389]):
        w=.067-i*.012
        for x,col in [(.548,BLUE),(.724,TEAL)]:
            fig.patches.append(Rectangle((x,y),w,.031,transform=fig.transFigure,color=col,alpha=1-i*.18))
        fig.lines.append(plt.Line2D([.615,.724],[y+.015,y+.015],transform=fig.transFigure,color=LINE,lw=1))
    fig.patches.append(Rectangle((.642,.341),.052,.032,transform=fig.transFigure,color=NAVY))
    text(.526,.290,'Weighted velocity loss',17,weight='bold')
    fig.patches.append(Rectangle((.530,.191),.065,.067,transform=fig.transFigure,color='#e5eff5',ec=BLUE,lw=1.2))
    fig.patches.append(Rectangle((.541,.202),.043,.045,transform=fig.transFigure,color='#76b9d6',ec=NAVY,lw=.9))
    text(.612,.239,'Core + halo',16)
    text(.612,.203,'Area + channels',16)
    text(.840,.739,'Six output fields',16,weight='bold')
    # All six original member maps, with T2m at the front of the variable stack.
    for i,f in enumerate(('precip','q2m','ps','u10m','v10m','t2m')):
        placements.append((sources[f],2,(.825+i*.006,.594-i*.022,.115,.107),None))
    text(.841,.441,'~3 km / member',16,weight='bold')
    text(.841,.400,'T₂m · rain · pₛ',16)
    text(.841,.365,'u₁₀ · v₁₀ · q₂m',16)
    text(.841,.306,f'{metadata["members"]} members',16,TEAL,weight='bold')
    text(.841,.272,'Independent seeds',14)
    # The shared state is advanced only after the overlapping tile velocities
    # have been assembled; this is not independent tile-by-tile sampling.
    text(.526,.139,'Tiled Heun  →  decode  →  ensemble',18,BLUE,weight='bold')
    text(.526,.101,'Blend tile velocities; advance one shared domain state',14)
    arrow(lefts[0]+width+.001,.49,lefts[1]-.001,.49)
    arrow(lefts[1]+width+.001,.49,lefts[2]-.001,.49)
    base=out/'layout_readme.pdf';fig.savefig(base,facecolor=fig.get_facecolor());plt.close(fig)
    records=compose(base,out/'fig01_workflow.pdf',placements,dpi)
    # Keep previous download names as aliases of the SAME supplied layout.
    for extension in ('pdf','png','svg'):
        shutil.copyfile(out/f'fig01_workflow.{extension}',out/f'fig01_workflow_paper.{extension}')
    base.unlink()
    atomic_json(out/'workflow_provenance.json',dict(time=entry['time'],case=case_id,case_selection=entry['reason'],focus_variable='t2m',
        layout='three panels; expanded flow matching with compact generated-field stack; identical paper and README figure',
        checkpoint=metadata['checkpoint'],epoch=metadata['epoch'],weights=metadata['weights'],members=metadata['members'],
        steps=metadata['steps'],post=metadata['post'],metrics_sha256=digest(root/'metrics.json'),
        renderer_sha256=digest(__file__),panels=records,input_variables=INPUT_VARIABLES,
        input_definition_sources={str(p):digest(p) for p in map(Path,(
            'configs/discover_annual_v2.yaml','configs/discover_v4_1.yaml',
            'src/merraflow/prepare_v2.py','src/merraflow/static_v2.py','src/merraflow/v4_1.py'))},
        note='Original GEOS/v4.1 vector maps with merged architecture/output panel; no HRRR fields and no meteorological values inferred from images.'))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--presentation',default='present_last_v4_1');p.add_argument('--case-id',default='20260307_1930')
    p.add_argument('--output',default='output/figures/workflow_geos_v4_1');p.add_argument('--dpi',type=int,default=300)
    print(make(**vars(p.parse_args())))


if __name__=='__main__':main()
