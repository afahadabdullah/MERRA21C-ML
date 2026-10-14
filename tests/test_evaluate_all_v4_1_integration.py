"""Synthetic real-model sampling: full enumeration, shards, resume and filtered pairing."""
from copy import deepcopy
import json
import pytest
import numpy as np
pytest.importorskip('torch')
from test_v4 import setup  # noqa: F401
from test_v4_1 import packed_cfg  # noqa: F401
from test_evaluate_v4_1 import trained  # noqa: F401
from merraflow.evaluate_all_v4_1 import initialize,run,aggregate,read_manifest,digest


def test_all_hours_shards_resume_and_frozen_checkpoint(trained,tmp_path):
    out=tmp_path/'all'
    initialize(deepcopy(trained),out,members=2,steps=2,post='both',spectral_count=2)
    manifest=read_manifest(out)
    assert manifest['entries'] and manifest['excluded']
    assert manifest['spectral_reference']['times']
    frozen=digest(out/'checkpoint.pt')
    with pytest.raises(FileExistsError):initialize(trained,out,members=2,steps=2,post='none')
    run(out,shard_index=0,shards=2,batch=2,threads=1,device='cpu')
    if len(manifest['entries'])>1:
        with pytest.raises(ValueError,match='Incomplete'):aggregate(out,draws=4)
    before={p.name:p.read_bytes() for p in (out/'cases').glob('*.json')}
    run(out,shard_index=0,shards=2,batch=2,threads=1,device='cpu')
    assert before=={p.name:p.read_bytes() for p in (out/'cases').glob('*.json')}
    run(out,shard_index=1,shards=2,batch=2,threads=1,device='cpu')
    summary=aggregate(out,draws=4)
    assert summary['complete'] and summary['hours']==len(manifest['entries'])
    assert set(summary['sources'])=={'coarse','regression','raw','spectral'}
    assert digest(out/'checkpoint.pt')==frozen
    assert manifest['settings']['saved_cases']
    assert digest(out/'fields'/'grid.npz')==manifest['saved_grid']['sha256']
    for path in (out/'cases').glob('*.json'):
        r=json.loads(path.read_text())
        assert len(r['seeds'])==2 and r['run_id']==manifest['run_id']
        assert r['sources']['raw']['members']==r['sources']['spectral']['members']==2
        if r['id'] in manifest['settings']['saved_cases']:
            for name,item in r['saved_fields'].items():
                assert digest(out/item['path'])==item['sha256']
                with np.load(out/item['path'],allow_pickle=False) as f:
                    assert f['ensemble'].dtype==np.float32 and len(f['ensemble'])==2
                    assert f['seeds'].tolist()==r['seeds']
    # Render the full scientific figure suite using only the saved outputs.
    from merraflow.paper_figures_v4_1 import make,load_results
    import os
    figure_out=os.path.join(os.environ['MERRAFLOW_PAPER_QA'],manifest['run_id'][:12]) if os.environ.get('MERRAFLOW_PAPER_QA') else str(tmp_path/'paper')
    make(out,figure_out,dpi=72)
    assert os.path.exists(os.path.join(figure_out,'fig01_workflow.pdf'))
    assert json.loads(open(os.path.join(figure_out,'plot_manifest.json')).read())['run_id']==manifest['run_id']
    assert os.path.exists(os.path.join(figure_out,'fig08_filtering.svg'))
    assert os.path.exists(os.path.join(figure_out,'table01_all_variables.csv'))
    completion=json.loads((out/'completion.json').read_text())
    completion['complete']=False;(out/'completion.json').write_text(json.dumps(completion))
    with pytest.raises(ValueError,match='complete'):load_results(out)
    completion['complete']=True;(out/'completion.json').write_text(json.dumps(completion))
    # Corrupt a saved map: resumed workers must reject it, not silently skip it.
    map_path=out/'fields'/f'{manifest["settings"]["saved_cases"][0]}_raw.npz'
    original_map=map_path.read_bytes();map_path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='fields changed'):run(out,shards=1,threads=1,device='cpu')
    map_path.write_bytes(original_map)
    # Evaluator cannot mix code versions on a resumed run.
    import merraflow.evaluate_all_v4_1 as evaluation
    original=evaluation.source_digest
    try:
        evaluation.source_digest=lambda:'other-code'
        with pytest.raises(ValueError,match='Code/checkpoint changed'):run(out,device='cpu')
    finally:evaluation.source_digest=original
