"""Data roots for the authors' post-processing pipeline (Tier 3).

These scripts are published as run for the paper. They read OpenFOAM case results and write
intermediate JSON/pickle files; set the roots through environment variables:

DLAMR_ROOT   repository root (default: this clone)
DLAMR_CASES  directory holding the case results (default: <root>/cases). The scripts refer to
             cases by the authors' run names; the mapping to the public case directories is
             given in analysis/README.md.
DLAMR_EVAL   scratch directory for intermediate outputs (default: <root>/analysis/output/eval)
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get('DLAMR_ROOT', Path(__file__).resolve().parents[2]))
CR = Path(os.environ.get('DLAMR_CASES', ROOT / 'cases'))
EVAL = Path(os.environ.get('DLAMR_EVAL', ROOT / 'analysis' / 'output' / 'eval'))
EVAL.mkdir(parents=True, exist_ok=True)
