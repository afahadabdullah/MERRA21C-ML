"""The visual resolution comparison must preserve the actual source cells."""
import numpy as np
import pytest
from merraflow.workflow_maps_v4_1 import centers_to_edges,crop,save_bundle,load_bundle


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
