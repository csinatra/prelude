"""Load `.env` before any module in this package reads it (see pipeline/env.py)."""

from pipeline.env import load_env

load_env()
