"""Submission uses the immutable wet-evaluation checkpoint and can resume."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FullConusSubmissionTests(unittest.TestCase):
    def test_submits_config_checkpoint_and_output_without_gpu_work(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root/'checkpoint_direct_v2.pt'
            config = root/'config.yaml'
            checkpoint.write_bytes(b'fixture')
            config.write_text('version: fixture\n')
            binary = root/'bin'
            binary.mkdir()
            capture = root/'submission.json'
            sbatch = binary/'sbatch'
            sbatch.write_text(f'#!{sys.executable}\nimport json, os, sys\n'
                              'with open(os.environ["CAPTURE"], "w") as stream:\n'
                              '    json.dump({"args": sys.argv[1:], "checkpoint": os.environ["CHECKPOINT"], '
                              '"config": os.environ["CONFIG"], "output": os.environ["OUTPUT"]}, stream)\n'
                              'print("42")\n')
            sbatch.chmod(0o755)
            output = root/'full_conus'
            env = dict(os.environ, CHECKPOINT=str(checkpoint), CONFIG=str(config), OUTPUT=str(output),
                       CAPTURE=str(capture), PATH=f'{binary}:'+os.environ['PATH'])
            command = ['bash', 'scripts/submit_full_conus_precip_direct_v2.sh']
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            submitted = json.loads(capture.read_text())
            self.assertEqual(submitted['checkpoint'], str(checkpoint))
            self.assertEqual(submitted['config'], str(config))
            self.assertEqual(submitted['output'], str(output))
            self.assertEqual(submitted['args'][-1], 'scripts/slurm_full_conus_precip_direct_v2.sh')
            output.mkdir()
            (output/'20260223_0530_full_conus_direct_v2.png').write_bytes(b'finished')
            capture.unlink()
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('already exists', result.stdout)
            self.assertFalse(capture.exists())


if __name__ == '__main__':
    unittest.main()
