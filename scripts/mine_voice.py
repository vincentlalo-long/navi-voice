#!/usr/bin/env python3
"""
CLI wrapper forwarding to voice_dataset_miner.py.
"""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from voice_dataset_miner import main

if __name__ == "__main__":
    main()
