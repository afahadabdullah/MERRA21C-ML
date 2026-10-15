"""Actual GEOS-FP → GEOS replay → v4.1 predictions, from archived vector maps.

Reuses original, identically scaled PDF map panels without recovering or inventing
meteorological values. Produces a map-rich README figure and a legible compact
paper figure. Saved numerical evaluation bundles remain the preferred input for
future complete-test figures.
"""
import argparse
import json
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
    def text(x,y,s,fs=8,color=INK,**kw):fig.text(x,y,s,fontsize=fs,color=color,**kw)
    def arrow(x1,y1,x2,y2):
        fig.patches.append(FancyArrowPatch((x1,y1),(x2,y2),transform=fig.transFigure,
            arrowstyle='-|>',mutation_scale=13,lw=1.3,color=NAVY))
    def panel(left,label,title,sub):
        fig.patches.append(Rectangle((left,.09),.229,.785,transform=fig.transFigure,
            facecolor='white',edgecolor=LINE,lw=.85,zorder=-10))
        text(left+.012,.848,label,12,BLUE,weight='bold');text(left+.035,.848,title,10.4,weight='bold')
        fig.lines.append(plt.Line2D([left+.012,left+.217],[.830,.830],transform=fig.transFigure,color=BLUE,lw=1.3))
        text(left+.012,.807,sub,7.3,MUTED)
    def image(source,col,bounds,title=None):
        placements.append((source,col,bounds))
        if title:text(bounds[0],bounds[1]+bounds[3]+.008,title,8.4,weight='semibold')
    text(.5,.961,'MERRA21C-ML: GEOS-FP TO HIGH-RESOLUTION WEATHER',23,NAVY,ha='center',weight='bold')
    text(.5,.928,'Coarse GEOS-FP inputs (~25 km)  ·  dynamical GEOS replay training reference (~3 km)  ·  learned high-resolution ensemble predictions',10.5,MUTED,ha='center')
    text(.5,.900,f'Actual v4.1 maps | {when} | epoch {metadata["epoch"]}, EMA, {metadata["members"]} members, {metadata["steps"]} Heun steps, {metadata["post"]}',8.5,TEAL,ha='center')
    lefts=[.018,.263,.508,.753]
    L=lefts[0];panel(L,'A','NATIVE GEOS-FP INPUTS','Current and previous-hour conditions; maps show current hour')
    image(sources['t2m'],0,(L+.014,.525,.201,.222),'2 m temperature: coarse native cells')
    image(sources['t2m'],'bar',(L+.014,.473,.201,.050))
    image(sources['precip'],0,(L+.014,.235,.201,.222),'Precipitation: coarse native cells')
    image(sources['precip'],'bar',(L+.014,.183,.201,.050))
    text(L+.015,.145,'Six weather fields at t − 1 h and t',8.5,weight='bold')
    text(L+.015,.116,'Terrain, location, time and broad context\n~25 km information remains coarse after interpolation',7.6,linespacing=1.6)
    L=lefts[1];panel(L,'B','DYNAMICAL GEOS REFERENCE','Matched temperature zoom; common extent and color scale')
    image(zoom,0,(L+.024,.514,.181,.246),'GEOS-FP native (~25 km)')
    image(zoom,1,(L+.024,.227,.181,.246),'High-resolution GEOS replay (HWT)')
    image(zoom,'bar',(L+.012,.171,.205,.049))
    text(L+.015,.145,'GEOS replay supplies the dynamical reference',8.3,weight='bold')
    text(L+.015,.112,'States: midpoint | Rain: ¼ P₋₃₀ + ½ P₀ + ¼ P₊₃₀\nNominal 2–3 km archive; no HRRR substitution',7.5,linespacing=1.6)
    L=lefts[2];panel(L,'C','GENERATED ~3 km FIELDS','Actual first ensemble member, after spectral post-processing')
    labels=('2 m temperature (°C)','Precipitation (mm h⁻¹)','Surface pressure (hPa)',
            '10 m zonal wind (m s⁻¹)','10 m meridional wind (m s⁻¹)','2 m humidity (g kg⁻¹)')
    for c,f in enumerate(FIELDS):
        x=L+.012+(c%2)*.113;y=.635-(c//2)*.207
        image(sources[f],2,(x,y,.099,.125),labels[c])
        image(sources[f],'bar',(x-.002,y-.047,.103,.045))
    text(L+.014,.145,'Joint variable-specific generation',8.7,weight='bold')
    text(L+.014,.126,'Rain: direct √(1 + P/P₀) − 1; P₀ = 1 mm h⁻¹\nT, pressure, winds: regression residuals\nHumidity: coarse-field residual',7.6,linespacing=1.55,va='top')
    L=lefts[3];panel(L,'D','FLOW-MATCHING ENSEMBLES','Frozen regression conditioner + conditional velocity U-Net')
    text(L+.015,.769,'GAUSSIAN → CONDITIONAL FLOW → WEATHER',8.8,BLUE,weight='bold')
    for x,label,col in zip([L+.038,L+.114,L+.191],['$z_0$','$z_t$','$v_\\theta$'],[BLUE,TEAL,NAVY]):
        fig.patches.append(Ellipse((x,.716),.055,.052,transform=fig.transFigure,facecolor='white',edgecolor=col,lw=1.5))
        text(x,.716,label,13,col,ha='center',va='center')
    arrow(L+.066,.716,L+.086,.716);arrow(L+.142,.716,L+.163,.716)
    text(L+.114,.661,'$z_t=(1-t)z_0+t z_1$; velocity target $z_1-z_0$',8.4,ha='center')
    text(L+.015,.617,'TRAINING AND FULL-DOMAIN SAMPLING',8.5,weight='bold')
    text(L+.015,.588,'192² patches: 128² core + 32-pixel halo\nArea/channel-weighted loss includes halo\nBlend overlapping velocities at every Heun stage\nOne coarse input + independent noise → members',8.0,linespacing=1.8,va='top')
    text(L+.015,.452,'Actual generated 2 m temperature',9,BLUE,weight='bold')
    image(zoom,2,(L+.012,.246,.099,.183),'Member 1')
    image(zoom,3,(L+.124,.246,.099,.183),'Mean of 8 members')
    image(zoom,'bar',(L+.012,.184,.211,.052))
    text(L+.015,.145,'Filtering sharpens details without retraining',8.0,TEAL,weight='bold')
    text(L+.015,.111,'Shown: saved spectral ensemble, not a raw/filter test\nEnsemble spread describes conditional uncertainty',7.5,linespacing=1.6)
    for i in range(3):arrow(lefts[i]+.230,.49,lefts[i+1]-.001,.49)
    text(.5,.053,'Maps: actual GEOS-FP inputs, HWT GEOS replay reference, and v4.1 predictions; one selected held-out case illustrates the workflow.',8,MUTED,ha='center')
    text(.5,.029,'Original map/color scales are preserved. GEOS replay is the dynamical reference; flow matching performs learned generative downscaling.',7.8,MUTED,ha='center')
    base=out/'layout_readme.pdf';fig.savefig(base,facecolor=fig.get_facecolor());plt.close(fig)
    records=compose(base,out/'fig01_workflow.pdf',placements,dpi)
    # Compact journal version: four large temperature maps plus readable method blocks.
    fig=plt.figure(figsize=(7.2,5.65),facecolor='white');paper=[]
    titles=('(a) GEOS-FP native','(b) GEOS replay reference','(c) Generated member 1','(d) Ensemble mean (8)')
    for i,title in enumerate(titles):
        x=.018+i*.247
        fig.text(x,.947,title,fontsize=7.3,weight='bold',color=INK)
        paper.append((zoom,i,(x,.591,.229,.329)))
    paper.append((zoom,'bar',(.12,.487,.76,.090)))
    fig.text(.5,.425,f'2 m temperature (°C) | {when} | epoch {metadata["epoch"]}, EMA, {metadata["steps"]} Heun steps, {metadata["post"]}',fontsize=7.1,ha='center',color=MUTED)
    boxes=[(.025,'1. CONDITIONS','Current + previous-hour GEOS-FP\nLocal and broad spatial context\nStatic surface and time features'),
           (.352,'2. HYBRID FLOW MATCHING','Direct square-root precipitation\nResidual temperature, pressure, winds\nCoarse-residual humidity'),
           (.679,'3. ENSEMBLE SAMPLING','Shared-state tiled Heun solver\nIndependent member noise draws\nOptional paired spectral adjustment')]
    for x,title,body in boxes:
        fig.patches.append(Rectangle((x,.145),.296,.210,transform=fig.transFigure,facecolor='#eff6fa',edgecolor=LINE,lw=.7))
        fig.text(x+.013,.318,title,fontsize=8,color=BLUE,weight='bold')
        fig.text(x+.013,.279,body,fontsize=7.1,color=INK,linespacing=1.65,va='top')
    for x in (.321,.648):
        fig.patches.append(FancyArrowPatch((x,.25),(x+.029,.25),transform=fig.transFigure,arrowstyle='-|>',mutation_scale=9,color=NAVY,lw=.9))
    fig.text(.5,.071,'Area/channel-weighted velocity loss; 192² patches include halo supervision.',fontsize=7.2,ha='center',color=INK)
    fig.text(.5,.036,'GEOS replay: dynamical reference on the nominal 2–3 km grid. Predictions: learned downscaling.',fontsize=7.0,ha='center',color=MUTED)
    base_paper=out/'layout_paper.pdf';fig.savefig(base_paper);plt.close(fig)
    paper_records=compose(base_paper,out/'fig01_workflow_paper.pdf',paper,dpi)
    base.unlink();base_paper.unlink()
    atomic_json(out/'workflow_provenance.json',dict(time=entry['time'],case=case_id,case_selection=entry['reason'],focus_variable='t2m',
        checkpoint=metadata['checkpoint'],epoch=metadata['epoch'],weights=metadata['weights'],members=metadata['members'],
        steps=metadata['steps'],post=metadata['post'],metrics_sha256=digest(root/'metrics.json'),
        renderer_sha256=digest(__file__),readme_panels=records,paper_panels=paper_records,
        note='Recomposed original vector PDF plots; no field values were inferred from images and no evaluation was rerun. High-resolution reference is GEOS replay, not HRRR.'))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--presentation',default='present_last_v4_1');p.add_argument('--case-id',default='20260307_1930')
    p.add_argument('--output',default='output/figures/workflow_geos_v4_1');p.add_argument('--dpi',type=int,default=300)
    print(make(**vars(p.parse_args())))


if __name__=='__main__':main()
