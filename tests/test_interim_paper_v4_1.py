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


def paper_template():
    """Minimal public QA fixture; never requires the ignored author manuscript."""
    blocks=('FIGURES','ABSTRACT','KEYPOINTS','SCOPE','TABLE_SETTINGS','TABLE_SCORES',
            'RESULTS','CONCLUSIONS','APPENDIX','STORM_METHOD')
    return (r'\documentclass{article}'+'\n'+r'\begin{document}'+'\n'
            +'\n'.join(f'% BEGIN AUTO {key}\n% placeholder\n% END AUTO {key}' for key in blocks)
            +'\n'+r'\end{document}'+'\n')


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
    figs=BUILDER['render'](out,72)
    paper=tmp_path/'manuscript'/'paper.tex';paper.parent.mkdir()
    paper.write_text(paper_template())
    BUILDER['update_paper'](out,paper,figs)
    BUILDER['update_paper'](out,paper)  # Safe repeat from figures.json; no duplicate blocks or stale fields.
    source=paper.read_text()
    assert '2 of 3 eligible held-out hours' in source and 'interim, completion-ordered' in source
    assert r'\section{Results}' in source and source.count(r'\section{Results}')==1
    assert not any(line.startswith(('\\figslot{','\\tableslot{')) for line in source.splitlines())
    for label in ('tab:scores','tab:settings','fig:scorecard','fig:temporal','fig:spectra','fig:precip','fig:calibration'):
        assert source.count('\\label{'+label+'}')==1, label
    assert 'fig:overview' not in source  # no saved maps in this snapshot
    assert (out/'paper_figures'/'table01_all_variables.csv').exists()
    assert (out/'paper_figures'/'figures.json').exists()
    assert (tmp_path/'manuscript'/'figures'/'auto'/'fig_scorecard.pdf').is_file()
    import pymupdf
    with pymupdf.open(out/'paper_figures'/'fig_scorecard.pdf') as doc:
        assert '2/3' in doc.metadata['subject'] and 'interim' in doc.metadata['subject']
    preview=BUILDER['inline_pdf'](out/'paper_figures'/'fig_scorecard.pdf')
    with pymupdf.open(stream=preview.encode('ascii'),filetype='pdf') as doc:
        assert len(doc)==1 and doc[0].get_images()
        assert len(set(doc[0].get_pixmap().samples))>100
    archive=BUILDER['bundle'](out,paper)
    import zipfile
    with zipfile.ZipFile(archive) as z:
        assert 'manuscript/paper.tex' in z.namelist()
        assert 'analysis/summary.json' in z.namelist()
        assert 'manuscript/figures/auto/fig_scorecard.pdf' in z.namelist()
    from merraflow.paper_figures_v4_1 import load_results
    with pytest.raises(ValueError,match='complete'):load_results(out)


def test_manuscript_without_auto_markers_is_rejected(parent,tmp_path):
    root,m,records=parent
    out=BUILDER['snapshot'](root,tmp_path/'snapshot',False)
    figs=BUILDER['render'](out,72)
    paper=tmp_path/'old.tex';paper.write_text('\\documentclass{article}\\begin{document}x\\end{document}\n')
    with pytest.raises(ValueError,match='AUTO'):BUILDER['update_paper'](out,paper,figs)


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
    nlat,nlon=np.arange(29.,42.,2.5),np.arange(-111.,-98.,3.125)
    native={f'native_{n}_{k}':v for n in ('t2m','u10m','v10m') for k,v in
            (('values',np.random.default_rng(1).uniform(270,280,(len(nlat),len(nlon))).astype('float32')),('lat',nlat),('lon',nlon))}
    for i,record in enumerate(records):
        record.update(run_id=m['run_id'],seeds=[11,12,13])
        record['sources']['spectral']=score_source(ensemble*.95+i,truth,area)
        if i==0:
            record['saved_fields']={}
            for method in ('raw','spectral'):
                relative=f'fields/{record["id"]}_{method}.npz'
                arrays=dict(ensemble=ensemble if method=='raw' else ensemble*.95,
                            run_id=np.asarray(m['run_id']),seeds=np.array(record['seeds']))
                if method=='raw':arrays.update(truth=truth,coarse=truth+2,regression=truth+1,**native)
                np.savez(root/relative,**arrays)
                record['saved_fields'][method]=dict(path=relative,sha256=digest(root/relative))
        atomic_json(root/'cases'/f'{record["id"]}.json',record)
    out=BUILDER['snapshot'](root,tmp_path/'snapshot')
    _,meta,s=BUILDER['load_snapshot'](out)
    assert meta['available_map_cases']==[records[0]['id']]
    assert s['hours']==2 and s['settings']['saved_cases']==m['settings']['saved_cases']
    figs=BUILDER['render'](out,72)
    assert [c['id'] for c in figs['cases']]==[records[0]['id']] and figs['cases'][0]['main']
    for kind in ('overview','zoom','ensemble','gain'):
        assert (out/'paper_figures'/f'{figs["cases"][0][kind]["stem"]}.pdf').is_file()
    paper=tmp_path/'paper.tex';paper.write_text(paper_template())
    BUILDER['update_paper'](out,paper,figs)
    text=paper.read_text()
    assert '\\label{fig:overview}' in text and '\\label{fig:zoom}' in text and '\\label{tab:spectral}' in text
    assert (tmp_path/'figures'/'auto'/f'fig_case_{records[0]["id"]}_overview.pdf').is_file()


def test_storm_case_study_from_storm_animation_netcdf(parent,tmp_path):
    xr=pytest.importorskip('xarray');pytest.importorskip('h5netcdf')
    root,m,records=parent
    rng=np.random.default_rng(0);T,H,W=5,40,48
    lat,lon=np.meshgrid(np.linspace(33,36,H),np.linspace(-90,-86,W),indexing='ij')
    data={}
    base=dict(t2m=285.,precip=0.,ps=1e5,u10m=2.,v10m=1.,q2m=.008)
    for name,b in base.items():
        f=lambda: (b+rng.gamma(.3,4,(T,H,W)) if name=='precip' else b+rng.normal(0,1e-3*abs(b)+.1,(T,H,W))).astype('float32')
        data[f'truth_{name}']=(('time','y','x'),f());data[f'geosfp_{name}']=(('time','y','x'),f())
        data[f'member_{name}']=(('time','run','member','y','x'),np.stack([f()[:,None],f()[:,None]],2).reshape(T,1,2,H,W))
    for k,v in (('lat',lat),('lon',lon),('elevation',np.full((H,W),100.))):
        data[k]=(('time','y','x'),np.repeat(v[None],T,0).astype('float32'))
    data['window_row0']=(('time',),np.zeros(T,'int32'));data['window_col0']=(('time',),np.zeros(T,'int32'))
    times=np.array([np.datetime64('2026-03-07T17:30')+np.timedelta64(k,'h') for k in range(T)])
    path=tmp_path/'storm_fields.nc'
    xr.Dataset(data,coords=dict(time=times,run=np.array(['flow'],dtype=object),member=[1,2]),
               attrs=dict(peak='2026-03-07T19:30:00',post='none',grid_km=3.)).to_netcdf(path,engine='h5netcdf')
    out=BUILDER['snapshot'](root,tmp_path/'snapshot',False)
    figs=BUILDER['render'](out,72,str(path))
    assert figs['storm']['peak'].startswith('2026-03-07T19:30') and figs['storm']['totals']
    paper=tmp_path/'paper.tex';paper.write_text(paper_template())
    BUILDER['update_paper'](out,paper,figs,inline=False)
    text=paper.read_text()
    assert '\\subsection{Storm case study}' in text and '\\label{fig:storm}' in text and 'filecontents' not in text.split('% END AUTO FIGURES')[0].split('% BEGIN AUTO FIGURES')[1]


@pytest.mark.parametrize('plots_only',[False,True])
def test_submission_stages_only_publishing_files_and_backs_up_paper(tmp_path,plots_only):
    import os
    import subprocess
    project=tmp_path/'project';project.mkdir()
    live=project/'src'/'merraflow';live.mkdir(parents=True)
    (live/'worker.py').write_text('frozen evaluator\n')
    output=project/'evaluation';output.mkdir();(output/'manifest.json').write_text('{}')
    paper=project/'manuscript'/'MERRA21C_ML_arxiv.tex';paper.parent.mkdir()
    paper.write_text(paper_template())
    if plots_only:paper.unlink()
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
    environment['PLOTS_ONLY']='1' if plots_only else '0'
    result=subprocess.run(['bash',str(Path('scripts/submit_interim_paper_v4_1.sh').resolve())],
                          env=environment,capture_output=True,text=True,check=True)
    submission=json.loads((tmp_path/'submitted.json').read_text())
    staged=Path(submission['builder']).parent
    assert 'Interim CPU paper job: 123456' in result.stdout
    assert submission['output']==str(output)
    if plots_only:
        assert not paper.exists()
        assert 'MERRAFlow_plots_metrics.zip' in result.stdout
    else:
        assert (staged/'manuscript_before_update.tex').read_text()==paper_template()
        assert paper.read_text()==paper_template()
    assert (staged/'build_interim_paper_v4_1.py').read_bytes()==Path('scripts/build_interim_paper_v4_1.py').read_bytes()
    assert (staged/'paper_plots_v4_1.py').read_bytes()==Path('scripts/paper_plots_v4_1.py').read_bytes()
    assert (staged/'conus_boundaries_50m.json.gz').read_bytes()==Path('src/merraflow/assets/conus_boundaries_50m.json.gz').read_bytes()
    assert (live/'worker.py').read_text()=='frozen evaluator\n'


def test_plots_only_cli_needs_no_manuscript(parent,tmp_path):
    import os,subprocess,sys,zipfile,csv
    root,_,_=parent
    out=tmp_path/'plots_only'
    source=tmp_path/'absent_manuscript.tex'
    subprocess.run([sys.executable,'scripts/build_interim_paper_v4_1.py',
                    '--evaluation',str(root),'--output',str(out),'--manuscript',str(source),
                    '--plots-only','--dpi','72'],check=True,capture_output=True,text=True)
    assert not source.exists()
    with (out/'paper_figures'/'score_matrix.csv').open() as stream:rows=list(csv.DictReader(stream))
    assert len(rows)==21 and rows[0]['hours']=='2' and rows[0]['expected_hours']=='3'
    with zipfile.ZipFile(out/'MERRAFlow_plots_metrics.zip') as archive:
        assert 'analysis/paper_figures/score_matrix.csv' in archive.namelist()
        assert not any(name.startswith('manuscript/') for name in archive.namelist())


def test_score_matrix_retains_zero_skill(parent,tmp_path):
    root,_,_=parent
    out=BUILDER['snapshot'](root,tmp_path/'snapshot',False)
    summary=BUILDER['load_snapshot'](out)[2]
    summary['sources']['raw']['fields']['t2m']['crpss_vs_coarse']=0.
    result=BUILDER['_plots']().fig_scorecard(summary,out,72,'zero_skill')
    assert result['crpss']['t2m']==0.
