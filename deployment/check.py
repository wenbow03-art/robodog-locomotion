#!/usr/bin/env python3
"""Portable deployment entrypoint: integrity check, or --model for CPU inference."""
import runpy
from pathlib import Path

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parent / 'lz_q1l_straight_20261105'
                       / 'tools' / 'verify_bundle.py'), run_name='__main__')
