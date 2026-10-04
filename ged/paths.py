"""Default locations inside the repository. Every script also accepts CLI overrides."""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = REPO_ROOT / "data"
ACTIVSG_CASE = DATA_DIR / "activsg2000" / "case_ACTIVSg2000.m"
GEOCODED_DIR = DATA_DIR / "activsg2000_geocoded"
RASTER_FILE = DATA_DIR / "raster" / "texas_raster_28ch.npz"

PROCESSED_DIR = DATA_DIR / "processed"
BACKBONE_DIR = PROCESSED_DIR / "backbone"          # 1,447-bus >=115 kV backbone CSVs
PATCH_1DEG_DIR = PROCESSED_DIR / "patches_1deg"    # 155 patches, 64x64 raster
PATCH_4DEG_DIR = PROCESSED_DIR / "patches_4deg"    # 128x128 raster
PATCH_7DEG_DIR = PROCESSED_DIR / "patches_7deg"    # 192x192 raster
GLOBAL_FILE = PROCESSED_DIR / "global" / "global.pt"  # full Texas, 256x256 raster
SPLITS_FILE = PROCESSED_DIR / "splits.json"

WEIGHTS_DIR = REPO_ROOT / "weights"
DEFAULT_WEIGHTS = WEIGHTS_DIR / "ged_texas.pt"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# Texas bounding box used for the full-state raster and for normalised positions.
TX_BBOX = {"lat_min": 25.84, "lat_max": 36.5, "lon_min": -106.65, "lon_max": -93.51}
