"""Put the repo root and scripts/ on sys.path.

The scripts are standalone entry points rather than an installed package, so
`import gc_treasury_overlays` only works if scripts/ is importable.  Doing it
here keeps a bare `pytest` working from the repo root.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
