from datetime import timedelta
from pathlib import Path
import json
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def plot_worker(rank, store, root):
    import merraflow.flow_progress_v2 as progress
    dist.init_process_group('gloo', init_method=f'file://{store}', rank=rank,
                            world_size=4, timeout=timedelta(seconds=60))
    group = dist.new_group(backend='gloo', timeout=timedelta(seconds=60))
    def render(*args):
        assert rank == 0
        path = Path(root)/'plots_v2/epoch_0005_case_v2.png'
        path.parent.mkdir(exist_ok=True)
        path.write_text('rendered once')
        return path
    progress.plot_flow_progress_v2 = render
    args = ({}, None, None, None, None, torch.device('cpu'), 5, root, group)
    result = progress.collective_flow_plot_v2(*args)
    assert result.read_text() == 'rendered once'
    def fail(*args):
        raise ValueError('injected plot error')
    progress.plot_flow_progress_v2 = fail
    # Resume skips an already saved plot, on every rank.
    assert progress.collective_flow_plot_v2(*args, skip_existing=True) == result
    try:
        progress.collective_flow_plot_v2(*args)
    except RuntimeError as exc:
        assert 'injected plot error' in str(exc)
    else:
        raise AssertionError('All ranks must receive rank-zero plotting errors')
    (Path(root)/f'rank_{rank}.json').write_text(json.dumps(dict(success=True)))
    dist.destroy_process_group()


def test_four_rank_plot_coordination_and_error_propagation(tmp_path):
    mp.spawn(plot_worker, args=(str(tmp_path/'store'), str(tmp_path)), nprocs=4, join=True)
    assert len(list(tmp_path.glob('rank_*.json'))) == 4
    assert len(list((tmp_path/'plots_v2').glob('*.png'))) == 1
