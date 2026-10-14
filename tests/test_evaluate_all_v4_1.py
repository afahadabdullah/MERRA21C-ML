"""Whole-archive evaluation contracts and numerical checks; no real-data skill claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from merraflow.evaluate_all_v4_1 import (FIELDS, field_scores, score_source, aggregate_sources,
    fss_parts, paired_intervals, atomic_json, canonical, aggregate, plan_entries, lock,
    initialize, run, read_manifest, source_digest, saved_case_ids)


def test_map_examples_are_fixed_before_test_scoring():
    entries=[dict(id=str(i)) for i in range(8)]
    assert saved_case_ids(entries)==['0','3','7']
    assert saved_case_ids(entries,'none')==[]
    assert saved_case_ids(entries,'2,5')==['2','5']
    assert len(saved_case_ids(entries,'all'))==8
    for bad in ('9','2,2'):
        with pytest.raises(ValueError):saved_case_ids(entries,bad)
    with pytest.raises(ValueError):saved_case_ids([])


def fields(seed=8):
    rng=np.random.default_rng(seed)
    truth=rng.uniform(.1,3,(6,8,9)).astype('float32')
    ensemble=np.maximum(0,truth[None]+rng.normal(0,.2,(3,6,8,9))).astype('float32')
    return ensemble,truth,np.linspace(1,2,72).reshape(8,9)


def test_crps_fair_and_variance_against_pairwise_definition():
    e,t,a=fields()
    x=e[:,0]; y=t[0];m=len(x)
    s=field_scores(x,y,a)
    first=np.abs(x-y).mean(0)
    pair=np.abs(x[:,None]-x[None,:]).sum((0,1))
    weight=lambda z:float((z*a).sum()/a.sum())
    assert s['crps']==pytest.approx(weight(first-pair/(2*m*m)),abs=1e-6)
    assert s['fair_crps']==pytest.approx(weight(first-pair/(2*m*(m-1))),abs=1e-6)
    assert s['variance_unbiased']==pytest.approx(weight(x.var(0,ddof=1)),rel=1e-6)
    assert sum(s['rank_histogram'])==pytest.approx(1)


def test_scores_cover_all_fields_speed_and_dry_event_handling():
    e,t,a=fields();e[:,1]=0;t[1]=0
    scored=score_source(e,t,a)
    assert set(scored['fields'])==set(FIELDS)
    expected=field_scores(np.hypot(e[:,3],e[:,4]),np.hypot(t[3],t[4]),a,seed=6)
    assert scored['fields']['wind_speed']['crps']==expected['crps']
    dry=scored['precipitation']['1.0']
    assert dry['brier']==0 and dry['observed_pixels']==0
    assert sum(dry['reliability_area'])==pytest.approx(a.sum())
    assert dry['fss']['1']['ensemble_mean']['denominator']==0
    assert fss_parts(e[0,1],t[1],1,99,a) is None
    with pytest.raises(ValueError):score_source(e,t,a*0)
    e[0,0,0,0]=np.nan
    with pytest.raises(ValueError):score_source(e,t,a)


def records_for_scores():
    e,t,a=fields();result=[]
    for hour,offset in enumerate((0.,1.)):
        sources={name:score_source(x,t,a) for name,x in
                 [('coarse',t[None]+2),('regression',t[None]+1),('raw',e+offset)]}
        result.append(dict(id=f'20260203_{hour:02d}30',time=f'2026-02-03T{hour:02d}:30:00',
                           sources=sources,area_m2=float(a.sum()),sampling_seconds=1,total_seconds=2,epoch=3))
    return result


def test_aggregation_pools_mse_and_event_counts_not_rmse_ratios():
    records=records_for_scores();s=aggregate_sources(records)
    r=s['raw']['fields']['t2m'];xs=[i['sources']['raw']['fields']['t2m'] for i in records]
    assert r['rmse_pooled']==pytest.approx(np.sqrt(np.mean([x['mse'] for x in xs])))
    assert r['rmse_pooled']!=pytest.approx(r['rmse_hour_mean'])
    assert r['spread_skill_corrected']==pytest.approx(np.sqrt(r['variance_population']*2)/r['rmse_pooled'])
    q=s['raw']['precipitation']['1.0'];parts=[i['sources']['raw']['precipitation']['1.0'] for i in records]
    assert q['observed_pixels']==sum(x['observed_pixels'] for x in parts)
    n=sum(x['fss']['5']['members']['numerator'] for x in parts)
    d=sum(x['fss']['5']['members']['denominator'] for x in parts)
    assert q['fss']['5']['members']==pytest.approx(1-n/d)


def test_plan_audits_exclusions_and_order():
    class A:
        index={'entries':[dict(id='b',time='2',split='test'),dict(id='a',time='1',split='test'),
                          dict(id='c',time='3',split='test')]}
        def eligible(self,split):return self.index['entries'][:2]
        def has_target(self,e):return False
    entries,excluded=plan_entries(A(),'test')
    assert [e['id'] for e in entries]==['a','b']
    assert excluded==[dict(id='c',time='3',reason='missing hourly precipitation target')]


def manifest_for(tmp_path,records):
    (tmp_path/'cases').mkdir()
    m=dict(version=1,settings=dict(post='none',members=3),entries=[{k:r[k] for k in ('id','time')} for r in records],
           excluded=[],checkpoint_sha256='test')
    m['run_id']=hashlib.sha256(canonical(m).encode()).hexdigest()
    atomic_json(tmp_path/'manifest.json',m)
    for r in records:
        r['run_id']=m['run_id'];atomic_json(tmp_path/'cases'/f'{r["id"]}.json',r)
    return m


def test_aggregate_refuses_missing_foreign_and_damaged_hours(tmp_path):
    records=records_for_scores();m=manifest_for(tmp_path,records)
    missing=tmp_path/'cases'/f'{records[1]["id"]}.json';missing.unlink()
    with pytest.raises(ValueError,match='Incomplete'):aggregate(tmp_path,draws=4)
    assert not (tmp_path/'summary.json').exists()
    assert not json.loads((tmp_path/'completion.json').read_text())['complete']
    atomic_json(missing,records[1]);records[0]['run_id']='foreign'
    first=tmp_path/'cases'/f'{records[0]["id"]}.json';atomic_json(first,records[0])
    with pytest.raises(ValueError,match='identity'):aggregate(tmp_path,draws=4)
    records[0]['run_id']=m['run_id'];atomic_json(first,records[0])
    summary=aggregate(tmp_path,draws=4)
    assert summary['complete'] and summary['hours']==2
    assert summary['bootstrap']['status']=='insufficient_blocks'
    assert (tmp_path/'scorecard.csv').exists()
    assert (tmp_path/'report.md').exists()
    # Manifest edits cannot silently change the accepted run.
    m['settings']['members']=8;atomic_json(tmp_path/'manifest.json',m)
    with pytest.raises(ValueError,match='hash'):read_manifest(tmp_path)


def test_temporal_blocks_preserve_pairs_and_use_elapsed_hours():
    records=[]
    for i in range(7):
        values={k:dict(fields={f:dict(crps=2 if k=='raw' else 3,mse=4,bias=0) for f in FIELDS}) for k in ('raw','coarse','regression')}
        records.append(dict(time=f'2026-02-{1+i*3:02d}T00:30:00',sources=values))
    r=paired_intervals(records,draws=20,block_hours=72)
    assert r['blocks']==7
    assert r['comparisons']['raw_minus_coarse']['q2m']['crps']['interval_95']==[-1.,-1.]
    assert paired_intervals(records,20,72)==r


def test_atomic_json_and_duplicate_worker_lock(tmp_path):
    path=tmp_path/'case.json';atomic_json(path,dict(a=1))
    with pytest.raises(ValueError):atomic_json(path,dict(a=float('nan')))
    assert json.loads(path.read_text())==dict(a=1)
    with lock(tmp_path/'hour.lock'):
        with pytest.raises(RuntimeError,match='Locked'):
            with lock(tmp_path/'hour.lock'):pass
    assert not (tmp_path/'hour.lock').exists()


def test_discover_launcher_dependencies_and_resume(tmp_path):
    import os
    import subprocess
    import sys
    fake=tmp_path/'bin';fake.mkdir();log=tmp_path/'jobs.jsonl'
    sbatch=fake/'sbatch'
    sbatch.write_text('#!'+sys.executable+'\n'+'''import os,sys,json
from pathlib import Path
p=Path(os.environ['JOB_LOG'])
lines=p.read_text().splitlines() if p.exists() else []
job=101+len(lines)
with p.open('a') as f:f.write(json.dumps(dict(args=sys.argv[1:],stage=os.environ.get('STAGE'),shards=os.environ['SHARDS']))+'\\n')
print(str(job)+';cluster')
''')
    sbatch.chmod(0o755)
    script=Path('scripts/submit_evaluate_all_v4_1.sh').resolve()
    env=dict(os.environ,PATH=str(fake)+os.pathsep+os.environ['PATH'],JOB_LOG=str(log),
             PROJECT_DIR=str(tmp_path),OUTPUT=str(tmp_path/'results'),SHARDS='5',MAX_PARALLEL='2',RESUME='0')
    subprocess.run(['bash',str(script)],env=env,check=True,capture_output=True,text=True)
    jobs=[json.loads(x) for x in log.read_text().splitlines()]
    assert len(jobs)==3 and jobs[0]['stage']=='init'
    assert jobs[0]['args'][-1]=='scripts/slurm_evaluate_all_v4_1_cpu.sh'
    assert '--dependency=afterok:101' in jobs[1]['args'] and '--array=0-4%2' in jobs[1]['args']
    assert jobs[2]['stage']=='aggregate' and '--dependency=afterok:102' in jobs[2]['args']
    assert jobs[2]['args'][-1]=='scripts/slurm_evaluate_all_v4_1_cpu.sh'
    (tmp_path/'results').mkdir();(tmp_path/'results'/'manifest.json').write_text('{}')
    env['RESUME']='1'
    subprocess.run(['bash',str(script)],env=env,check=True,capture_output=True,text=True)
    jobs=[json.loads(x) for x in log.read_text().splitlines()]
    assert len(jobs)==5
    assert not any(x.startswith('--dependency') for x in jobs[3]['args'])
    assert '--dependency=afterok:104' in jobs[4]['args']
    env['SHARDS']='0'
    assert subprocess.run(['bash',str(script)],env=env,capture_output=True).returncode!=0
