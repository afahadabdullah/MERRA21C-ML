import unittest
from unittest.mock import patch
from merraflow.slurm_time_v2 import flow_time_left_seconds, parse_time_left_v2


class SlurmTimeV2Tests(unittest.TestCase):
    def test_time_left_formats(self):
        self.assertEqual(parse_time_left_v2('39:59\n'), 2399)
        self.assertEqual(parse_time_left_v2('11:20:30'), 40830)
        self.assertEqual(parse_time_left_v2('1-02:03:04'), 93784)
        self.assertIsNone(parse_time_left_v2('UNLIMITED'))
        with self.assertRaises(ValueError):
            parse_time_left_v2('NOT_SET')

    @patch('merraflow.slurm_time_v2.subprocess.run')
    @patch.dict('os.environ', {'SLURM_JOB_ID': '12345'})
    def test_query_uses_current_job(self, run):
        run.return_value.stdout = '00:39:00\n'
        self.assertEqual(flow_time_left_seconds(), 2340)
        run.assert_called_once_with(
            ['squeue', '--noheader', '--jobs', '12345', '--format=%L'],
            check=True, capture_output=True, text=True)
