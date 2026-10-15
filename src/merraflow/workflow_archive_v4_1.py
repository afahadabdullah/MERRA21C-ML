"""Actual GEOS-FP → GEOS replay → v4.1 predictions, from archived vector maps.

Reuses original, identically scaled PDF map panels without recovering or inventing
meteorological values. Preserves the supplied four-panel layout for both the README and
paper figure. Saved numerical evaluation bundles remain the preferred input for
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


def source_clip(page,column):
    """Locate original equal-extent map axes; exclude old headings and scale labels."""
    images=[i for i in page.get_image_info() if i['width']>100 and i['height']>100]
    if len(images)<4:raise ValueError('Expected four original map images')
    first=min(images,key=lambda i:i['bbox'][0])['bbox']
    labels=('GEOS-FP 0.25°','Truth 2 km','One member','Ensemble mean (8)')
    matches=page.search_for(labels[column])
    if len(matches)!=1:raise ValueError(f'Cannot identify original panel: {labels[column]}')
    center=(matches[0].x0+matches[0].x1)/2
    width=first[2]-first[0]
    return (center-width/2,first[1],center+width/2,first[3])


def compose(base,output,placements,dpi):
    import pymupdf
    doc=pymupdf.open(base);page=doc[0];w,h=page.rect.width,page.rect.height
    opened={};records=[]
    try:
        for source,column,bounds in placements:
            source=Path(source)
            if source not in opened:opened[source]=pymupdf.open(source)
            src=opened[source][0]
            clip=(src.rect.width*.25,src.rect.height-68,src.rect.width*.78,src.rect.height) if column=='bar' else source_clip(src,column)
            x,y,dx,dy=bounds
            rect=pymupdf.Rect(x*w,(1-y-dy)*h,(x+dx)*w,(1-y)*h)
            page.show_pdf_page(rect,opened[source],0,clip=pymupdf.Rect(clip),keep_proportion=True)
            records.append(dict(source=str(source),source_sha256=digest(source),column=column,
                                clip_points=list(clip),destination_fraction=list(bounds)))
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
    when=entry['time'].replace('T',' ')+' UTC'
    placements=[]
    fig=plt.figure(figsize=(22,11),facecolor='#f7fafc')
    def text(x,y,s,fs=8,color=INK,**kw):return fig.text(x,y,s,fontsize=fs,color=color,**kw)
    def arrow(x1,y1,x2,y2):
        fig.patches.append(FancyArrowPatch((x1,y1),(x2,y2),transform=fig.transFigure,
            arrowstyle='-|>',mutation_scale=13,lw=1.3,color=NAVY))
    def frame(left,label,title,sub):
        fig.patches.append(Rectangle((left,.090),.229,.785,transform=fig.transFigure,
            facecolor='white',edgecolor=LINE,lw=.85,zorder=-10))
        text(left+.012,.848,label,12,BLUE,weight='bold');text(left+.035,.848,title,10.5,weight='bold')
        fig.lines.append(plt.Line2D([left+.012,left+.217],[.830,.830],transform=fig.transFigure,color=BLUE,lw=1.3))
        text(left+.012,.807,sub,7.5,MUTED)
    def image(source,col,bounds,title=None):
        placements.append((source,col,bounds))
        if title:text(bounds[0],bounds[1]+bounds[3]+.008,title,8.2,weight='semibold')
    def bar(source,bounds):placements.append((source,'bar',bounds))
    text(.5,.961,'MERRA21C-ML: GENERATIVE CONUS DOWNSCALING',23,NAVY,ha='center',weight='bold')
    text(.5,.928,'Native GEOS-FP (~25 km)  →  variable-specific conditional flow matching  →  six fields on the HWT (~3 km) grid',11,MUTED,ha='center')
    text(.5,.900,f'GEOS-FP + high-resolution GEOS replay + v4.1 predictions | {when} | epoch {metadata["epoch"]}, EMA, {metadata["members"]} members, {metadata["post"]}',8.5,TEAL,ha='center')
    lefts=[.018,.263,.508,.753];width=.229
    L=lefts[0];frame(L,'A','COARSE WEATHER INPUTS','Current + previous hour; static and time features')
    text(L+.016,.773,'Native 0.25° × 0.3125° cells',8,BLUE,weight='bold')
    image(sources['t2m'],0,(L+.015,.515,width-.030,.235),'2 m temperature')
    bar(sources['t2m'],(L+.015,.470,width-.030,.045))
    # The original project archive stores wind components, not full-domain speed.
    image(sources['u10m'],0,(L+.015,.224,width-.030,.224),'10 m wind: zonal component')
    bar(sources['u10m'],(L+.015,.179,width-.030,.045))
    text(L+.016,.165,'Six weather fields at t − 1 h and t',9,weight='bold')
    text(L+.016,.135,'Local 192² patches + broad 576² context\nTerrain, land/sea, coordinates, cell area, time',8,linespacing=1.6)
    L=lefts[1];frame(L,'B','SAME REGION, TWO GRIDS','Identical extent and color scale; no display interpolation')
    image(zoom,0,(L+.018,.525,width-.036,.233),'~25 km: native cells')
    bar(zoom,(L+.018,.480,width-.036,.045))
    image(zoom,1,(L+.018,.231,width-.036,.233),'~3 km: GEOS replay reference (HWT)')
    bar(zoom,(L+.018,.186,width-.036,.045))
    text(L+.018,.165,'Temporal pairing and training-only statistics',8.7,weight='bold')
    text(L+.018,.132,'States: midpoint | Rain: ¼ P₋₃₀ + ½ P₀ + ¼ P₊₃₀\nChronological splits with gaps; packed training data',7.5,linespacing=1.7)
    L=lefts[2];frame(L,'C','SIX JOINT OUTPUT FIELDS','Actual v4.1 model member on the high-resolution GEOS grid')
    labels=('2 m temperature','Hourly mean rain rate','Surface pressure','10 m zonal wind','10 m meridional wind','2 m specific humidity')
    mw=(width-.044)/2
    for c,f in enumerate(FIELDS):
        x=L+.012+(c%2)*(mw+.020);y=.633-(c//2)*.206
        image(sources[f],2,(x,y,mw,.128),labels[c])
        bar(sources[f],(x-.002,y-.045,mw+.004,.043))
    text(L+.013,.165,'Variable-specific generative coordinates',8.8,weight='bold')
    text(L+.013,.141,'Rain: direct √(1 + P/P₀) − 1; P₀ = 1 mm h⁻¹\nT, pressure, winds: regression residuals\nHumidity: coarse-field residual',7.4,linespacing=1.6,va='top')
    L=lefts[3];frame(L,'D','REGRESSION → GENERATION','Frozen conditioner + local and broad-context U-Net')
    text(L+.014,.770,'CONDITIONAL FLOW MATCHING',9,BLUE,weight='bold')
    xs=[L+.034,L+.113,L+.192]
    for x,symbol,col in zip(xs,['$z_0$','$z_t$','$v_\\theta$'],[BLUE,TEAL,NAVY]):
        fig.patches.append(Ellipse((x,.717),.056,.052,transform=fig.transFigure,facecolor='white',edgecolor=col,lw=1.5))
        text(x,.717,symbol,13,col,ha='center',va='center')
    arrow(xs[0]+.028,.717,xs[1]-.028,.717);arrow(xs[1]+.028,.717,xs[2]-.028,.717)
    text(xs[0],.675,'Gaussian noise',7,MUTED,ha='center');text(xs[1],.675,'Conditional path',7,MUTED,ha='center')
    text(xs[2],.675,'U-Net velocity',7,MUTED,ha='center')
    text(L+width/2,.640,'$z_t=(1-t)z_0+t z_1$   |   target: $z_1-z_0$',9,ha='center')
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
    for i in range(3):
        fig.patches.append(Rectangle((L+.028+i*.057,.159+i*.008),.058,.055,transform=fig.transFigure,
            facecolor='#e5f2f4',edgecolor=TEAL,lw=1))
    text(L+.020,.135,f'One coarse input → {metadata["members"]} generated high-resolution members',7,MUTED)
    text(L+.014,.104,'Raw / spectral: paired members, no retraining',7.8,TEAL,weight='bold')
    for i in range(3):arrow(lefts[i]+width+.001,.49,lefts[i+1]-.001,.49)
    text(.5,.054,f'Maps: native GEOS-FP, HWT dynamical GEOS replay reference and actual v4.1 predictions | {when} | {metadata["steps"]} Heun steps, {metadata["post"]}.',8,MUTED,ha='center')
    text(.5,.030,'Temperature zooms share extent and scale. Original map panels are preserved; flow matching performs learned downscaling to the GEOS reference grid.',7.7,MUTED,ha='center')
    base=out/'layout_readme.pdf';fig.savefig(base,facecolor=fig.get_facecolor());plt.close(fig)
    records=compose(base,out/'fig01_workflow.pdf',placements,dpi)
    # Keep previous download names as aliases of the SAME supplied layout.
    for extension in ('pdf','png','svg'):
        shutil.copyfile(out/f'fig01_workflow.{extension}',out/f'fig01_workflow_paper.{extension}')
    base.unlink()
    atomic_json(out/'workflow_provenance.json',dict(time=entry['time'],case=case_id,case_selection=entry['reason'],focus_variable='t2m',
        layout='four-panel reference layout; identical paper and README figure',
        checkpoint=metadata['checkpoint'],epoch=metadata['epoch'],weights=metadata['weights'],members=metadata['members'],
        steps=metadata['steps'],post=metadata['post'],metrics_sha256=digest(root/'metrics.json'),
        renderer_sha256=digest(__file__),panels=records,
        note='Original GEOS/v4.1 vector maps in the supplied four-stage layout; no HRRR fields and no meteorological values inferred from images.'))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--presentation',default='present_last_v4_1');p.add_argument('--case-id',default='20260307_1930')
    p.add_argument('--output',default='output/figures/workflow_geos_v4_1');p.add_argument('--dpi',type=int,default=300)
    print(make(**vars(p.parse_args())))


if __name__=='__main__':main()
