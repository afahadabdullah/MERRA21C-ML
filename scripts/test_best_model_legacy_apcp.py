#!/usr/bin/env python3
"""Run the five-member map diagnostic against the old, known-invalid APCP archive.

This wrapper is for historical debugging only. It never enables legacy APCP
targets in preparation or training.
"""
import sys
from test_best_model import main


if __name__ == '__main__':
    sys.argv.append('--legacy-apcp')
    main()
