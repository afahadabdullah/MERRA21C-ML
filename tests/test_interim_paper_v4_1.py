"""Synthetic QA only: interim scope, parent immutability, redraw and manuscript population."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import shutil

import numpy as np
import pytest
import yaml

from test_evaluate_all_v4_1 import records_for_scores
from merraflow.evaluate_all_v4_1 import canonical,atomic_json,read_case,aggregate

BUILDER=runpy.run_path('scripts/build_interim_paper_v4_1.py')


@pytest.fixture
def parent(tmp_path):
    root=tmp_path/'parent';(root/'cases').mkdir(parents=True)
    records=records_for_scores()
    entries=[{k:r[k] for k in ('id','time')} for r in records]
    entries.append(dict(id='20260203_0230',time='2026-02-03T02:30:00'))
    m=dict(version=1,entries=entries,settings=dict(post='none',members=3,steps=64,weights='ema',saved_cases=[]),
           excluded=[dict(id='excluded',reason='history')],checkpoint_sha256='fixture',
           config=yaml.safe_load(Path('configs/discover_v4_1.yaml').read_text()))
    m['run_id']=hashlib.sha256(canonical(m).encode()).hexdigest();atomic_json(root/'manifest.json',m)
    for record in records:
        record.update(run_id=m['run_id'],epoch=243)
        atomic_json(root/'cases'/f'{record["id"]}.json',record)
    return root,m,records


def test_snapshot_remains_partial_and_leaves_parent_unchanged(parent,tmp_path):
    root,m,records=parent
    before={p.relative_to(root):p.read_bytes() for p in root.rglob('*.json')}
    out=BUILDER['snapshot'](root,tmp_path/'snapshot',False)
    manifest,meta,summary=BUILDER['load_snapshot'](out)
    assert summary['hours']==2 and summary['expected_hours']==3 and not summary['complete']
    assert summary['scope']=='interim' and not summary['bootstrap']['comparisons']
    assert manifest['run_id']==m['run_id'] and len(manifest['entries'])==3
    assert [e['id'] for e in meta['entries']]==[r['id'] for r in records]
    assert before=={p.relative_to(root):p.read_bytes() for p in root.rglob('*.json')}
    for record in records:
        assert (out/'cases'/f'{record["id"]}.json').read_bytes()==(root/'cases'/f'{record["id"]}.json').read_bytes()
    with pytest.raises(ValueError,match='Incomplete'):aggregate(out,draws=4)
    with pytest.raises(FileExistsError):BUILDER['snapshot'](root,out,False)
    with pytest.raises(ValueError,match='separate'):BUILDER['snapshot'](root,root/'snapshot',False)


def test_snapshot_freezes_membership_before_later_completions(parent,tmp_path,monkeypatch):
    root,m,records=parent
    import merraflow.evaluate_all_v4_1 as evaluation
    def arrived(path,manifest,entry):
        extra=deepcopy(records[0]);extra.update(m['entries'][-1]);atomic_json(root/'cases'/f'{extra["id"]}.json',extra)
        return read_case(path,manifest,entry)
    monkeypatch.setattr(evaluation,'read_case',arrived)
    out=BUILDER['snapshot'](root,tmp_path/'snapshot',False)
    assert len(list((root/'cases').glob('*.json')))==3
    assert BUILDER['load_snapshot'](out)[2]['hours']==2


def test_invalid_or_changed_data_cannot_be_silently_published(parent,tmp_path):
    root,m,records=parent
    out=BUILDER['snapshot'](root,tmp_path/'snapshot',False)
    summary=json.loads((out/'summary.json').read_text());summary['hours']=3;atomic_json(out/'summary.json',summary)
    with pytest.raises(ValueError,match='summary changed'):BUILDER['load_snapshot'](out)
    record=records[0];record['run_id']='foreign';atomic_json(root/'cases'/f'{record["id"]}.json',record)
    with pytest.raises(ValueError,match='identity'):BUILDER['snapshot'](root,tmp_path/'bad',False)


def test_render_and_populate_paper_from_saved_snapshot(parent,tmp_path):
    root,m,records=parent
    out=BUILDER['snapshot'](root,tmp_path/'snapshot',False)
    BUILDER['render'](out,72)
    paper=tmp_path/'manuscript'/'paper.tex';paper.parent.mkdir()
    shutil.copyfile('manuscript/MERRA21C_ML_arxiv.tex',paper)
    BUILDER['update_paper'](out,paper)
    BUILDER['update_paper'](out,paper)  # Safe repeat; no duplicate blocks or stale fields.
    source=paper.read_text()
    assert 'Interim snapshot: 2 of 3 eligible test hours' in source
    assert r'\section{Interim results}' in source
    assert '20.4\\%' not in source and 'three-case processed ensemble' not in source
    assert not any(line.startswith(('\\figslot{','\\tableslot{')) for line in source.splitlines())
    assert source.count('\\label{tab:scores}')==1 and source.count('\\label{fig:skill}')==1
    assert (out/'paper_figures'/'table01_all_variables.csv').exists()
    import pymupdf
    with pymupdf.open(out/'paper_figures'/'fig02_skill.pdf') as doc:
        assert 'INTERIM: 2/3' in doc[0].get_text()
    preview=BUILDER['inline_pdf'](out/'paper_figures'/'fig02_skill.pdf')
    with pymupdf.open(stream=preview.encode('ascii'),filetype='pdf') as doc:
        assert len(doc)==1 and doc[0].get_images()
        assert len(set(doc[0].get_pixmap().samples))>100
    archive=BUILDER['bundle'](out,paper)
    import zipfile
    with zipfile.ZipFile(archive) as z:
        assert 'manuscript/paper.tex' in z.namelist()
        assert 'analysis/summary.json' in z.namelist()
    from merraflow.paper_figures_v4_1 import load_results
    with pytest.raises(ValueError,match='complete'):load_results(out)


def test_only_completed_predeclared_example_fields_are_copied_and_plotted(parent,tmp_path):
    root,m,records=parent
    from test_evaluate_all_v4_1 import fields
    from merraflow.evaluate_all_v4_1 import score_source,digest
    ensemble,truth,area=fields()
    m.pop('run_id');m['settings'].update(post='both',saved_cases=[records[0]['id'],m['entries'][-1]['id']])
    (root/'fields').mkdir()
    lat,lon=np.meshgrid(np.linspace(30,40,8),np.linspace(-110,-100,9),indexing='ij')
    np.savez(root/'fields'/'grid.npz',lat=lat,lon=lon,area=area)
    m['saved_grid']=dict(path='fields/grid.npz',sha256=digest(root/'fields'/'grid.npz'))
    m['run_id']=hashlib.sha256(canonical(m).encode()).hexdigest();atomic_json(root/'manifest.json',m)
    for i,record in enumerate(records):
        record.update(run_id=m['run_id'],seeds=[11,12,13])
        record['sources']['spectral']=score_source(ensemble*.95+i,truth,area)
        if i==0:
            record['saved_fields']={}
            for method in ('raw','spectral'):
                relative=f'fields/{record["id"]}_{method}.npz'
                arrays=dict(ensemble=ensemble if method=='raw' else ensemble*.95,
                            run_id=np.asarray(m['run_id']),seeds=np.array(record['seeds']))
                if method=='raw':arrays.update(truth=truth,coarse=truth+2,regression=truth+1)
                np.savez(root/relative,**arrays)
                record['saved_fields'][method]=dict(path=relative,sha256=digest(root/relative))
        atomic_json(root/'cases'/f'{record["id"]}.json',record)
    out=BUILDER['snapshot'](root,tmp_path/'snapshot')
    _,meta,s=BUILDER['load_snapshot'](out)
    assert meta['available_map_cases']==[records[0]['id']]
    assert s['hours']==2 and s['settings']['saved_cases']==m['settings']['saved_cases']
    maps=BUILDER['render'](out,72)
    assert len(maps)==14 and all(records[0]['id'] in name for name in maps)
    paper=tmp_path/'paper.tex';shutil.copyfile('manuscript/MERRA21C_ML_arxiv.tex',paper)
    BUILDER['update_paper'](out,paper)
    assert '\\section{Saved interim map examples}' in paper.read_text()
    assert (tmp_path/'figures'/'interim'/f'MERRAFlow_interim_fig09_{records[0]["id"]}_t2m.pdf').is_file()


def test_submission_stages_only_publishing_files_and_backs_up_paper(tmp_path):
    import os
    import subprocess
    project=tmp_path/'project';project.mkdir()
    live=project/'src'/'merraflow';live.mkdir(parents=True)
    (live/'worker.py').write_text('frozen evaluator\n')
    output=project/'evaluation';output.mkdir();(output/'manifest.json').write_text('{}')
    paper=project/'manuscript'/'MERRA21C_ML_arxiv.tex';paper.parent.mkdir()
    paper.write_text('local source before template update\n')
    tools=tmp_path/'bin';tools.mkdir()
    (tools/'git').write_text('''#!/usr/bin/env python3
import os,sys
from pathlib import Path
if sys.argv[1]=='rev-parse': print('fixture-publishing-commit')
elif sys.argv[1]=='show':
    path=sys.argv[2].split(':',1)[1]
    sys.stdout.buffer.write((Path(os.environ['QA_REPO'])/path).read_bytes())
else: raise SystemExit('Unexpected git mutation')
''')
    (tools/'sbatch').write_text('''#!/usr/bin/env python3
import json,os
from pathlib import Path
Path(os.environ['QA_SUBMISSION']).write_text(json.dumps(dict(builder=os.environ['INTERIM_BUILDER'],output=os.environ['OUTPUT'])))
print('123456;discover')
''')
    for executable in tools.iterdir():executable.chmod(0o755)
    environment=dict(os.environ,PATH=str(tools)+os.pathsep+os.environ['PATH'],
        QA_REPO=str(Path.cwd()),QA_SUBMISSION=str(tmp_path/'submitted.json'),
        PROJECT_DIR=str(project),OUTPUT=str(output))
    for name in ('KEEP_MANUSCRIPT','PUBLISHING_REF','MANUSCRIPT','SNAPSHOT_OUTPUT'):
        environment.pop(name,None)
    result=subprocess.run(['bash',str(Path('scripts/submit_interim_paper_v4_1.sh').resolve())],
                          env=environment,capture_output=True,text=True,check=True)
    submission=json.loads((tmp_path/'submitted.json').read_text())
    staged=Path(submission['builder']).parent
    assert 'Interim CPU paper job: 123456' in result.stdout
    assert submission['output']==str(output)
    assert (staged/'manuscript_before_template_update.tex').read_text()=='local source before template update\n'
    assert paper.read_bytes()==Path('manuscript/MERRA21C_ML_arxiv.tex').read_bytes()
    assert (staged/'build_interim_paper_v4_1.py').read_bytes()==Path('scripts/build_interim_paper_v4_1.py').read_bytes()
    assert (live/'worker.py').read_text()=='frozen evaluator\n'
