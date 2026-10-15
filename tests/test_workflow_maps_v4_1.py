"""The visual resolution comparison must preserve the actual source cells."""
import numpy as np
import pytest
from merraflow.workflow_maps_v4_1 import centers_to_edges,crop,save_bundle,load_bundle


def test_hrrr_is_rejected_in_the_geos_workflow(tmp_path):
    from merraflow.workflow_maps_v4_1 import render
    with pytest.raises(ValueError,match='HRRR'):
        render(dict(public_proxy=True),tmp_path)


def test_archived_vector_map_clips_retain_equal_extent_and_exclude_headings():
    pymupdf=pytest.importorskip('pymupdf')
    from pathlib import Path
    from merraflow.workflow_archive_v4_1 import source_clip
    path=Path('present_last_v4_1/cases/20260307_1930/zoomA_precip.pdf')
    if not path.exists():pytest.skip('Original archive plot is not installed')
    with pymupdf.open(path) as doc:
        clips=[pymupdf.Rect(source_clip(doc[0],i)) for i in range(4)]
        assert all(r.width==pytest.approx(clips[0].width) and r.height==pytest.approx(clips[0].height) for r in clips)
        assert all(r.y0>doc[0].search_for('Truth 2 km')[0].y0 for r in clips)


def test_sparse_colorbar_labels_preserve_source_scale_and_signed_winds():
    pymupdf=pytest.importorskip('pymupdf')
    from pathlib import Path
    from merraflow.workflow_archive_v4_1 import colorbar_spec
    root=Path('present_last_v4_1/cases/20260307_1930')
    if not root.exists():pytest.skip('Original archive plot is not installed')
    for field,expected in [('precip',['0.1','4','64']),('u10m',['-5','0','5']),
                           ('v10m',['-5','0','5']),('ps',['800','1000'])]:
        with pymupdf.open(root/f'conus_{field}.pdf') as doc:
            clip,ticks=colorbar_spec(doc[0],field)
            assert [label for _,label in ticks]==expected
            fractions=[fraction for fraction,_ in ticks]
            assert 0<fractions[0]<fractions[-1]<1
            assert fractions==sorted(fractions)
            assert pymupdf.Rect(clip).height<30  # Gradient only; no old fine-print labels.
            if field in ('u10m','v10m'):
                assert fractions[1]==pytest.approx(.5,abs=.001)


def test_lossless_preview_pdf_preserves_rgb_and_tex_source(tmp_path):
    pymupdf=pytest.importorskip('pymupdf')
    import io
    from PIL import Image
    import runpy
    helpers=runpy.run_path('scripts/embed_workflow_preview.py')
    pixels=np.arange(6*9*3,dtype='uint8').reshape(6,9,3)
    png=tmp_path/'figure.png';Image.fromarray(pixels).save(png)
    pdf=helpers['png_pdf'](png).encode('ascii')
    with pymupdf.open(stream=pdf,filetype='pdf') as doc:
        image=doc.extract_image(doc[0].get_images()[0][0])['image']
        np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(image))),pixels)
    tex=tmp_path/'paper.tex';tex.write_text('\\documentclass{article}\n\\begin{document}Paper\\end{document}\n')
    helpers['embed'](tex,png);first=tex.read_text();helpers['embed'](tex,png)
    assert tex.read_text()==first and first.count(helpers['START'])==1
    assert '\\begin{document}Paper' in first


def test_native_cell_edges_and_identical_extent_preserve_resolution():
    lon=np.arange(-112,-99,.3125);lat=np.arange(33,44,.25)
    values=np.arange(len(lon)*len(lat)).reshape(len(lat),len(lon))
    edges=centers_to_edges(lon)
    np.testing.assert_allclose(np.diff(edges),.3125)
    np.testing.assert_allclose((edges[:-1]+edges[1:])/2,lon)
    _,_,coarse=crop(lon,lat,values,(-110,-102,35,41.5))
    fine_lon=np.arange(-112,-99,.03125);fine_lat=np.arange(33,44,.025)
    _,_,fine=crop(fine_lon,fine_lat,np.zeros((len(fine_lat),len(fine_lon))),(-110,-102,35,41.5))
    assert fine.shape[0]>8*coarse.shape[0] and fine.shape[1]>8*coarse.shape[1]
    assert np.isin(coarse,values).all()  # No interpolated coarse values introduced.
    with pytest.raises(ValueError):centers_to_edges(lon[::-1])


def test_workflow_bundle_preserves_fields_native_grids_and_source_labels(tmp_path):
    lat,lon=np.meshgrid(np.linspace(34,44,8),np.linspace(-112,-100,9),indexing='ij')
    case=dict(hr=np.ones((6,8,9),dtype='float32'),lat=lat,lon=lon,time='2026-02-03T00:30:00',
        coarse=np.zeros((6,8,9),dtype='float32'),members_t2m=np.ones((2,8,9),dtype='float32'),
        native=dict(t2m=dict(values=np.arange(12).reshape(3,4),lat=np.arange(3),lon=np.arange(4))),
        public_proxy=False,reference_label='HWT',footer='Saved project data')
    path=tmp_path/'workflow_data.npz';save_bundle(path,case);loaded=load_bundle(path)
    for name in ('hr','coarse','lat','lon','members_t2m'):np.testing.assert_array_equal(loaded[name],case[name])
    assert loaded['public_proxy'] is False and loaded['reference_label']=='HWT'
    np.testing.assert_array_equal(loaded['native']['t2m']['values'],case['native']['t2m']['values'])
