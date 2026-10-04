# Data

Everything needed to reproduce the paper is in this folder (about 3 MB). The
processed training and evaluation samples are rebuilt from these inputs by
`python scripts/01_prepare_data.py` in a few seconds; the rebuild is
byte-identical to the samples used in the paper.

| Path | Size | Content | Source and license |
|---|---|---|---|
| `activsg2000/case_ACTIVSg2000.m` | 0.7 MB | ACTIVSg2000 synthetic Texas case (MATPOWER) | Texas A&M University, CC BY 3.0 (`LICENSE_CC-BY-3.0.txt`, `README_TAMU.txt`) |
| `activsg2000_geocoded/{bus,branch,gen}.csv` | 0.4 MB | The case as CSV with bus coordinates geocoded from bus names | Derived from ACTIVSg2000 (CC BY 3.0) and the U.S. Census 2024 Gazetteer (public domain) |
| `raster/texas_raster_28ch.npz` | 1.1 MB | 28-channel geographic raster over Texas, 214 x 263 cells, values in [0, 1] | Derived from OpenStreetMap, U.S. Census, and U.S. EIA data (see below) |

## ACTIVSg2000

A. B. Birchfield, T. Xu, K. M. Gegner, K. S. Shetye, and T. J. Overbye, "Grid
structural characteristics as validation criteria for synthetic networks,"
IEEE Transactions on Power Systems, vol. 32, no. 4, pp. 3258-3265, 2017. The
case is entirely synthetic and contains no CEII. The original files are
distributed by the Texas A&M Electric Grid Test Case Repository
(https://electricgrids.engr.tamu.edu).

## Geocoding (optional step 0)

`activsg2000_geocoded/` was produced by `scripts/00_geocode_activsg.py`, which
matches each ACTIVSg2000 bus name to a U.S. Census place. To re-create it,
download the 2024 Gazetteer "Places" national file
(https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2024_Gazetteer/2024_Gaz_place_national.zip),
unzip it to `data/gazetteer/2024_Gaz_place_national.txt`, and run

```bash
python scripts/00_geocode_activsg.py --input data/activsg2000/case_ACTIVSg2000.m \
    --output-dir data/activsg2000_geocoded --region texas --force
```

79 of 2,000 names do not resolve; they are placed at the Texas centroid
(31.0, -99.0) and dropped by `01_prepare_data.py`. Many buses share a Census
place and therefore share coordinates.

## Geographic raster

`texas_raster_28ch.npz` holds `features` (float32, 214 x 263 x 28, row 0 =
southern edge), `channel_names`, and `bbox` = (lat_min, lon_min, lat_max,
lon_max) = (25.84, -106.65, 36.5, -93.51). Each channel is min-max normalised.

| Channels | Content | Source |
|---|---|---|
| 0-7 | road density, road-graph average degree, motorway / trunk / primary / secondary / tertiary / residential road share | OpenStreetMap |
| 8-12 | building density; residential / commercial / industrial / other building share | OpenStreetMap |
| 13 | population density | U.S. Census |
| 14-20 | land use: residential, commercial, industrial, farmland, forest, water, other | OpenStreetMap |
| 21-23 | distance to nearest power plant, nearest plant capacity, plant density | U.S. EIA plant inventory |
| 24-27 | sin / cos encodings of latitude and longitude | computed |

OpenStreetMap-derived channels: (c) OpenStreetMap contributors, available under
the Open Database License (ODbL 1.0). The pipeline that rasterised the raw
OpenStreetMap, Census and EIA sources is part of a larger code base and is not
included here; the released raster is the exact input used for all results.

## Processed layout (created by `01_prepare_data.py`)

```
processed/
  activsg2000_texas/   buses with state == TX (1,947 buses)
  backbone/            >= 115 kV, geocoded, largest component: 1,447 buses, 2,495 branches
  patches_1deg/        155 patches of 1 x 1 degree, stride 0.5, >= 8 buses, 64 x 64 raster
  splits.json          124 / 15 / 16 train / val / test (random, seed 42)
  patches_4deg/        13 patches of 4 x 4 degree, stride 2, >= 30 buses, 128 x 128 raster
  patches_7deg/        6 patches of 7 x 7 degree, stride 3, >= 60 buses, 192 x 192 raster
  global/global.pt     full Texas, 1,447 buses, 1,992 unique bus pairs, 256 x 256 raster
```

Each sample is a dict with `raster` (H, W, 28), `gt_pos` (N, 2) in [0, 1]^2
as (lon, lat), `gt_edge_index` (2, E), `gt_voltage` (N,) in kV,
`gt_edge_attrs` (E, 3) z-normalised log10 (r, x, b), `gt_bus_attrs`
(Pd, Qd, Pg, Pmax, type, tier), `bus_id` (N,) and `patch_meta`.
