#!/usr/bin/env python3
"""PolyStrata from a terminal.  The program is terminal/polystrata.py; this file keeps `python polystrata.py`
working from the checkout's root."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parent / "terminal" / "polystrata.py"), run_name="__main__")
