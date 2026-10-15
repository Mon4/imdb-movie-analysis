"""Central place for filesystem paths used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
PLOTS_DIR = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
