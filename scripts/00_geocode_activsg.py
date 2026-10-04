#!/usr/bin/env python3
"""Parse the ACTIVSg2000 MATPOWER case and geocode bus names (optional step 0).

The shipped data/activsg2000_geocoded/ CSVs were produced by this script. To
re-create them, download the U.S. Census 2024 Gazetteer "Places" file
(2024_Gaz_place_national.txt, see data/README.md) and run:

    python scripts/00_geocode_activsg.py \
        --input data/activsg2000/case_ACTIVSg2000.m \
        --output-dir data/activsg2000_geocoded \
        --region texas --gazetteer /path/to/2024_Gaz_place_national.txt --force

Each bus_name is matched to a Census place (trailing integers stripped, ties
broken by in-region state and land area). Unresolved names are placed at the
region centroid (31.0, -99.0) and later dropped by 01_prepare_data.py.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

LOG = logging.getLogger("parse_activsg")

# Region -> USPS whitelists (from task spec).
REGIONS: dict[str, set[str]] = {
    "texas": {"TX"},
    "wecc": {"WA", "OR", "CA", "NV", "ID", "MT", "WY", "UT", "CO", "AZ", "NM"},
    "nemid": {"NY", "NJ", "PA", "DE", "MD", "MA", "CT", "RI", "VT", "NH", "ME", "DC"},
    "eastern": {
        "IL", "IN", "OH", "KY", "TN", "NC", "SC", "GA", "FL", "AL", "MS", "AR", "LA",
        "MO", "MN", "WI", "MI", "IA", "NY", "NJ", "PA", "DE", "MD", "MA", "CT", "RI",
        "VT", "NH", "ME", "DC", "WV", "VA", "OK", "KS", "NE", "SD", "ND", "TX",
    },
}

LSAD_SUFFIX_RE = re.compile(r"\s+(CITY|TOWN|CDP|VILLAGE|BOROUGH|COMM|TWP|PLACE)\b.*$", re.IGNORECASE)
TRAILING_INT_RE = re.compile(r"\s+\d+\s*$")
DEFAULT_GAZ = Path(__file__).resolve().parent.parent / "data" / "gazetteer" / "2024_Gaz_place_national.txt"

# Region centroid (single fallback per region) used when a bus_name fails to geocode.
REGION_CENTROID: dict[str, tuple[float, float]] = {
    "texas":   (31.0, -99.0),
    "wecc":    (40.0, -115.0),
    "nemid":   (41.5,  -74.5),
    "eastern": (37.0,  -85.0),
}


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# --- MATPOWER parsing ------------------------------------------------------
def _strip_inline_comment(line: str) -> str:
    """Remove `;%comment` and `% ...` tail without breaking string-literal contents."""
    in_str = False
    out_chars: list[str] = []
    for ch in line:
        if ch == "'":
            in_str = not in_str
        if ch == "%" and not in_str:
            break
        out_chars.append(ch)
    return "".join(out_chars)


def _extract_matrix(text: str, name: str) -> list[list[float]]:
    """Extract numeric matrix `mpc.<name> = [ ... ];`. Robust to comments,
    multi-line rows separated by ';', and ACTIVSg2000-style rows separated only
    by newlines."""
    pat = re.compile(rf"mpc\.{re.escape(name)}\s*=\s*\[(.*?)\]\s*;", re.DOTALL)
    m = pat.search(text)
    if m is None:
        raise ValueError(f"matrix mpc.{name} not found")
    body = m.group(1)
    rows: list[list[float]] = []
    if ";" in body:
        buffer = ""
        for raw_line in body.splitlines():
            line = _strip_inline_comment(raw_line)
            if not line.strip():
                continue
            buffer += " " + line
            while ";" in buffer:
                chunk, buffer = buffer.split(";", 1)
                tokens = chunk.split()
                if not tokens:
                    continue
                try:
                    rows.append([float(tok) for tok in tokens])
                except ValueError:
                    LOG.warning("non-numeric row in mpc.%s skipped: %r", name, tokens[:6])
    else:
        for raw_line in body.splitlines():
            line = _strip_inline_comment(raw_line).strip()
            if not line:
                continue
            tokens = line.split()
            if not tokens:
                continue
            try:
                rows.append([float(tok) for tok in tokens])
            except ValueError:
                LOG.warning("non-numeric row in mpc.%s skipped: %r", name, tokens[:6])
    return rows


def _extract_str_cell(text: str, name: str) -> list[str]:
    """Extract `mpc.<name> = { 'a'; 'b'; ... };` as Python list of strings."""
    pat = re.compile(rf"mpc\.{re.escape(name)}\s*=\s*\{{(.*?)\}}\s*;", re.DOTALL)
    m = pat.search(text)
    if m is None:
        return []
    body = m.group(1)
    return [s.strip() for s in re.findall(r"'([^']*)'", body)]


# --- Gazetteer geocoding ---------------------------------------------------
def _norm(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().upper()


def load_gazetteer(path: Path) -> dict[str, list[tuple[str, float, float, float]]]:
    """Build {normalized_name: [(USPS, lat, lon, ALAND_SQMI), ...]}.
    Stores both the LSAD-suffixed name and the suffix-stripped variant so that
    'PORTLAND CITY' and 'PORTLAND' both resolve.
    """
    df = pd.read_csv(path, sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    idx: dict[str, list[tuple[str, float, float, float]]] = defaultdict(list)
    for _, row in df.iterrows():
        try:
            usps = row["USPS"].strip()
            lat = float(row["INTPTLAT"])
            lon = float(row["INTPTLONG"])
            sqmi = float(row["ALAND_SQMI"])
        except (ValueError, KeyError):
            continue
        full_name = row["NAME"].strip()
        for variant in {_norm(full_name), _norm(LSAD_SUFFIX_RE.sub("", full_name))}:
            if variant:
                idx[variant].append((usps, lat, lon, sqmi))
    LOG.info("gazetteer: %d unique normalized name keys", len(idx))
    return idx


def geocode(
    bus_name: str,
    gaz: dict[str, list[tuple[str, float, float, float]]],
    region_states: set[str],
) -> tuple[float, float, str] | None:
    """Return (lat, lon, USPS) or None. Region-aware tie-break + ALAND_SQMI ranking.

    Lookup ladder: try the original name, then strip trailing-int disambiguators
    one by one (longest match first), then suffix-stripped variants.
    """
    raw = bus_name.strip()
    candidates: list[tuple[str, float, float, float]] | None = None
    # Try progressively shorter prefixes by stripping one trailing int at a time.
    cur = raw
    seen: set[str] = set()
    while cur and cur not in seen:
        seen.add(cur)
        candidates = gaz.get(_norm(cur)) or gaz.get(_norm(LSAD_SUFFIX_RE.sub("", cur)))
        if candidates:
            break
        new = TRAILING_INT_RE.sub("", cur).strip()
        if new == cur:
            break
        cur = new
    if not candidates:
        return None
    in_region = [c for c in candidates if c[0] in region_states]
    pool = in_region if in_region else candidates
    pool_sorted = sorted(pool, key=lambda c: c[3], reverse=True)
    usps, lat, lon, _ = pool_sorted[0]
    return lat, lon, usps


# --- Frame builders --------------------------------------------------------
_BUS_RAW_COLS = ["bus_i", "type", "Pd", "Qd", "Gs", "Bs", "area", "Vm", "Va", "baseKV"]
_BUS_OUT_COLS = ["bus_id", "type", "Pd", "Qd", "Vm", "Va", "baseKV"]
_BRANCH_COLS = ["from_bus", "to_bus", "r", "x", "b", "rateA", "rateB", "rateC",
                "tap", "shift", "status", "angmin", "angmax"]
_GEN_RAW_COLS = ["bus", "Pg", "Qg", "Qmax", "Qmin", "Vg", "mBase", "status", "Pmax", "Pmin"]


def build_bus_df(bus_rows: list[list[float]], names: list[str], coords: list[tuple[float, float, str]]) -> pd.DataFrame:
    df = pd.DataFrame([r[: len(_BUS_RAW_COLS)] for r in bus_rows], columns=_BUS_RAW_COLS)
    out = df[["bus_i", "type", "Pd", "Qd", "Vm", "Va", "baseKV"]].rename(columns={"bus_i": "bus_id"})
    out["bus_id"] = out["bus_id"].astype(int)
    n = len(out)
    if len(names) != n:
        LOG.warning("bus_name count (%d) != bus row count (%d)", len(names), n)
    out["bus_name"] = (names + [""] * n)[:n]
    padded = (coords + [(float("nan"), float("nan"), "")] * n)[:n]
    out["lat"] = [c[0] for c in padded]
    out["lon"] = [c[1] for c in padded]
    out["state"] = [c[2] for c in padded]
    return out[_BUS_OUT_COLS + ["bus_name", "lat", "lon", "state"]]


def build_branch_df(rows: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame([r[:13] for r in rows], columns=_BRANCH_COLS)
    for col in ("from_bus", "to_bus", "status"):
        df[col] = df[col].astype(int)
    return df


def build_gen_df(rows: list[list[float]], gentype: list[str], genfuel: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([r[:10] for r in rows], columns=_GEN_RAW_COLS).rename(columns={"bus": "bus_id"})
    df["bus_id"] = df["bus_id"].astype(int)
    df["status"] = df["status"].astype(int)
    n = len(df)
    df["gentype"] = (gentype + [""] * n)[:n]
    df["genfuel"] = (genfuel + [""] * n)[:n]
    return df


# --- Driver ----------------------------------------------------------------
def parse_one(input_path: Path, out_dir: Path, region: str, gaz_path: Path, force: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    expected = [out_dir / f for f in ("bus.csv", "branch.csv", "gen.csv", "metadata.json")]
    if not force and all(p.exists() for p in expected):
        LOG.info("outputs already present in %s: skip (use --force to override)", out_dir)
        return

    LOG.info("parsing %s (region=%s)", input_path.name, region)
    text = input_path.read_text()

    bus_rows = _extract_matrix(text, "bus")
    branch_rows = _extract_matrix(text, "branch")
    gen_rows = _extract_matrix(text, "gen")
    bus_names = _extract_str_cell(text, "bus_name")
    gentype = _extract_str_cell(text, "gentype")
    genfuel = _extract_str_cell(text, "genfuel")
    LOG.info("rows: bus=%d branch=%d gen=%d names=%d", len(bus_rows), len(branch_rows), len(gen_rows), len(bus_names))

    gaz = load_gazetteer(gaz_path)
    region_set = REGIONS[region]
    coords: list[tuple[float, float, str]] = []
    matches = 0
    unresolved: list[str] = []
    for n in bus_names:
        hit = geocode(n, gaz, region_set)
        if hit is not None:
            coords.append(hit)
            matches += 1
        else:
            lat, lon = REGION_CENTROID[region]
            coords.append((lat, lon, sorted(region_set)[0]))
            unresolved.append(n)
    match_rate = matches / max(1, len(bus_names))
    LOG.info("geocode match rate: %.4f (%d / %d): %d unresolved", match_rate, matches, len(bus_names), len(unresolved))
    if unresolved:
        sample = unresolved[:5]
        LOG.warning("unresolved sample: %s%s", sample, " ..." if len(unresolved) > 5 else "")

    bus_df = build_bus_df(bus_rows, bus_names, coords)
    branch_df = build_branch_df(branch_rows)
    gen_df = build_gen_df(gen_rows, gentype, genfuel)

    bus_df.to_csv(out_dir / "bus.csv", index=False)
    branch_df.to_csv(out_dir / "branch.csv", index=False)
    gen_df.to_csv(out_dir / "gen.csv", index=False)
    metadata = {
        "source_file": input_path.name,
        "region": region,
        "bus_count": int(len(bus_df)),
        "branch_count": int(len(branch_df)),
        "gen_count": int(len(gen_df)),
        "geocode_match_rate": round(match_rate, 6),
        "unresolved_count": len(unresolved),
        "states_covered": sorted(set(bus_df["state"].dropna().tolist()) - {""}),
        "schema_version": 1,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[parse_activsg] {input_path.name}: bus={len(bus_df)} match_rate={match_rate:.4f} -> {out_dir}", file=sys.stderr)


def main(argv: Iterable[str] | None = None) -> int:
    setup_logging()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--region", choices=sorted(REGIONS), required=True)
    p.add_argument("--gazetteer", type=Path, default=DEFAULT_GAZ)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    t0 = time.time()
    parse_one(args.input, args.output_dir, args.region, args.gazetteer, args.force)
    LOG.info("done in %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
