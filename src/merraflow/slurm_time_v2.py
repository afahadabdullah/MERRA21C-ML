"""Read a running Slurm job's remaining wall time without Slurm Python bindings."""
import os
import subprocess


def parse_time_left_v2(value):
    value = value.strip()
    if value == 'UNLIMITED':
        return None
    days = 0
    if '-' in value:
        day_text, value = value.split('-', 1)
        days = int(day_text)
    fields = value.split(':')
    if len(fields) not in (2, 3) or any(not part.isdigit() for part in fields):
        raise ValueError(f'Invalid Slurm time-left value: {value!r}')
    if len(fields) == 2:
        minutes, seconds = map(int, fields)
        hours = 0
    else:
        hours, minutes, seconds = map(int, fields)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f'Invalid Slurm time-left value: {value!r}')
    return days*86400+hours*3600+minutes*60+seconds


def flow_time_left_seconds():
    job_id = os.getenv('SLURM_JOB_ID')
    if not job_id:
        return None
    result = subprocess.run(['squeue', '--noheader', '--jobs', job_id, '--format=%L'],
                            check=True, capture_output=True, text=True)
    return parse_time_left_v2(result.stdout)
