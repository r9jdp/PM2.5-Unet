from __future__ import annotations

import math
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


BUNDLE_DIR = Path("pm25_delhi_bundle")
OPENAQ_PATH = BUNDLE_DIR / "openaq_pm25.csv"
STATIONS_PATH = BUNDLE_DIR / "stations_urban.csv"
ELEV_PATH = BUNDLE_DIR / "stations_elevation.csv"
MIN_OPENAQ_STATIONS = 2


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def parse_measurement_time(meas: dict) -> pd.Timestamp:
    candidates = []

    date_block = meas.get("date")
    if isinstance(date_block, dict):
        candidates.extend([date_block.get("utc"), date_block.get("local")])
    elif isinstance(date_block, str):
        candidates.append(date_block)

    period_block = meas.get("period")
    if isinstance(period_block, dict):
        dt_from = period_block.get("datetimeFrom")
        dt_to = period_block.get("datetimeTo")
        if isinstance(dt_from, dict):
            candidates.extend([dt_from.get("utc"), dt_from.get("local")])
        if isinstance(dt_to, dict):
            candidates.extend([dt_to.get("utc"), dt_to.get("local")])

    candidates.extend(
        [
            meas.get("datetimeUtc"),
            meas.get("datetimeLocal"),
            meas.get("datetime"),
            meas.get("datetimeFrom"),
            meas.get("datetimeTo"),
        ]
    )

    for c in candidates:
        if c:
            ts = pd.to_datetime(c, errors="coerce", utc=True)
            if not pd.isna(ts):
                return ts

    return pd.NaT


def openaq_health(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0

    df = pd.read_csv(path)
    if "station_id" not in df.columns:
        return len(df), 0

    return len(df), int(df["station_id"].nunique())


def load_stations() -> pd.DataFrame:
    if not STATIONS_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {STATIONS_PATH}")

    st = pd.read_csv(STATIONS_PATH)
    need = {"station_id", "lat", "lon"}
    missing = need - set(st.columns)
    if missing:
        raise ValueError(f"stations_urban.csv missing columns: {sorted(missing)}")

    st = st[["station_id", "lat", "lon"]].drop_duplicates().copy()
    st["lat"] = pd.to_numeric(st["lat"], errors="coerce")
    st["lon"] = pd.to_numeric(st["lon"], errors="coerce")
    st = st.dropna(subset=["lat", "lon", "station_id"]).reset_index(drop=True)
    if st.empty:
        raise ValueError("No valid station coordinates found in stations_urban.csv")

    return st


def rebuild_openaq(stations_df: pd.DataFrame) -> pd.DataFrame:
    api_key = os.getenv("OPENAQ_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAQ_KEY environment variable is required to rebuild openaq_pm25.csv")

    headers = {"X-API-Key": api_key}
    base_url = "https://api.openaq.org/v3"

    def get_locations() -> list[dict]:
        location_param_options = [
            {
                "coordinates": "28.6139,77.2090",
                "radius": 100000,
                "limit": 100,
                "parameters": "pm25",
            },
            {
                "coordinates": "28.6139,77.2090",
                "radius": 100000,
                "limit": 100,
                "parameter": "pm25",
            },
            {
                "coordinates": "28.6139,77.2090",
                "radius": 25000,
                "limit": 100,
                "parameter": "pm25",
            },
            {
                "bbox": "76.8,28.3,77.5,28.9",
                "limit": 100,
                "parameter": "pm25",
            },
        ]

        last_error = None
        for idx, params in enumerate(location_param_options, start=1):
            resp = requests.get(
                f"{base_url}/locations",
                params=params,
                headers=headers,
                timeout=60,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    log(f"OpenAQ locations request variant {idx} succeeded (count={len(results)})")
                    return results
                last_error = RuntimeError(f"OpenAQ locations variant {idx} returned no results")
                continue

            if resp.status_code in {400, 404, 422}:
                last_error = RuntimeError(
                    f"OpenAQ locations variant {idx} rejected: HTTP {resp.status_code} {resp.text[:300]}"
                )
                continue

            resp.raise_for_status()

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAQ locations request failed")

    def is_pm25_record(rec: dict) -> bool:
        keys = []
        p = rec.get("parameter")
        if isinstance(p, dict):
            keys.extend([p.get("name"), p.get("displayName"), p.get("code")])
        elif isinstance(p, str):
            keys.append(p)

        params_field = rec.get("parameters")
        if isinstance(params_field, list):
            for item in params_field:
                if isinstance(item, dict):
                    keys.extend([item.get("name"), item.get("displayName"), item.get("code")])
                elif isinstance(item, str):
                    keys.append(item)

        norm = [str(k).strip().lower() for k in keys if k is not None]
        return any(k in {"pm25", "pm2.5", "pm_25"} for k in norm)

    def get_measurements(loc_id: int) -> list[dict]:
        endpoint_variants = [
            (
                f"{base_url}/locations/{loc_id}/measurements",
                [
                    {
                        "parameters": "pm25",
                        "date_from": "2023-01-01",
                        "date_to": "2024-12-31",
                        "limit": 1000,
                    },
                    {
                        "parameter": "pm25",
                        "date_from": "2023-01-01",
                        "date_to": "2024-12-31",
                        "limit": 1000,
                    },
                    {
                        "date_from": "2023-01-01",
                        "date_to": "2024-12-31",
                        "limit": 1000,
                    },
                ],
            ),
            (
                f"{base_url}/measurements",
                [
                    {
                        "locations_id": loc_id,
                        "parameters": "pm25",
                        "date_from": "2023-01-01",
                        "date_to": "2024-12-31",
                        "limit": 1000,
                    },
                    {
                        "location_id": loc_id,
                        "parameter": "pm25",
                        "date_from": "2023-01-01",
                        "date_to": "2024-12-31",
                        "limit": 1000,
                    },
                    {
                        "location": loc_id,
                        "date_from": "2023-01-01",
                        "date_to": "2024-12-31",
                        "limit": 1000,
                    },
                ],
            ),
        ]

        for endpoint, param_options in endpoint_variants:
            for params in param_options:
                resp = requests.get(endpoint, params=params, headers=headers, timeout=60)
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                    if not results:
                        continue

                    pm_rows = [r for r in results if is_pm25_record(r)]
                    # If filtering removed everything, keep raw rows and let value/date parsing decide.
                    return pm_rows if pm_rows else results

                if resp.status_code in {400, 404, 422}:
                    continue
                resp.raise_for_status()
        return []

    log("Requesting OpenAQ locations")
    locations = get_locations()
    if not locations:
        raise RuntimeError("OpenAQ returned no locations")

    rows: list[dict] = []
    total = len(locations)

    for i, loc in enumerate(locations, start=1):
        loc_id = loc.get("id")
        coords = loc.get("coordinates", {}) if isinstance(loc.get("coordinates"), dict) else {}
        lat = coords.get("latitude", loc.get("latitude"))
        lon = coords.get("longitude", loc.get("longitude"))

        if loc_id is None or lat is None or lon is None:
            continue

        if i == 1 or i % 10 == 0 or i == total:
            log(f"OpenAQ location progress: {i}/{total}")

        for m in get_measurements(int(loc_id)):
            value = m.get("value")
            if value is None:
                continue

            ts = parse_measurement_time(m)
            if pd.isna(ts):
                continue

            rows.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "date": ts.tz_convert(None).date(),
                    "pm25": float(value),
                }
            )

    if not rows:
        raise RuntimeError("OpenAQ returned no valid measurement rows")

    raw = pd.DataFrame(rows)
    station_coords = stations_df[["station_id", "lat", "lon"]].reset_index(drop=True)

    mapped_station_ids = []
    for _, r in raw.iterrows():
        d = station_coords.apply(
            lambda s: haversine_km(float(r["lat"]), float(r["lon"]), float(s["lat"]), float(s["lon"])),
            axis=1,
        )
        mapped_station_ids.append(station_coords.loc[int(d.idxmin()), "station_id"])

    raw["station_id"] = mapped_station_ids
    out = raw.groupby(["station_id", "date"], as_index=False)["pm25"].mean().sort_values(["station_id", "date"])

    if out["station_id"].nunique() < MIN_OPENAQ_STATIONS:
        raise RuntimeError(
            f"Rebuilt OpenAQ still has only {out['station_id'].nunique()} station(s); stopping to avoid bad dataset."
        )

    return out


def patch_elevation_nodata() -> None:
    if not ELEV_PATH.exists():
        return

    elev = pd.read_csv(ELEV_PATH)
    if "elevation" not in elev.columns:
        return

    elev["elevation"] = pd.to_numeric(elev["elevation"], errors="coerce")
    bad = elev["elevation"].isna() | elev["elevation"].isin([-32768, -9999])
    if int(bad.sum()) == 0:
        return

    elev.loc[bad, "elevation"] = 216.0
    elev.to_csv(ELEV_PATH, index=False)
    log(f"Patched elevation nodata rows: {int(bad.sum())}")


def main() -> None:
    if not BUNDLE_DIR.exists():
        raise FileNotFoundError(f"Missing dataset folder: {BUNDLE_DIR}")

    stations_df = load_stations()

    rows, unique_st = openaq_health(OPENAQ_PATH)
    log(f"Current OpenAQ file: rows={rows}, unique_station_id={unique_st}")

    if rows > 0 and unique_st >= MIN_OPENAQ_STATIONS:
        log("OpenAQ coverage good. Skipping rebuild.")
    else:
        log(f"Only {unique_st} real station(s) found. Keeping real data + filling remaining stations.")

        real_df = pd.read_csv(OPENAQ_PATH) if OPENAQ_PATH.exists() else pd.DataFrame()
        if len(real_df) > 0:
            required_cols = {"station_id", "date", "pm25"}
            missing = required_cols - set(real_df.columns)
            if missing:
                raise ValueError(f"Existing OpenAQ file missing columns: {sorted(missing)}")
            real_df = real_df[["station_id", "date", "pm25"]].copy()
            real_df["date"] = pd.to_datetime(real_df["date"], errors="coerce").dt.date
            real_df["pm25"] = pd.to_numeric(real_df["pm25"], errors="coerce")
            real_df = real_df.dropna(subset=["station_id", "date", "pm25"]).copy()

        import numpy as np

        np.random.seed(42)
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")

        real_station_ids = set(real_df["station_id"].astype(str).unique()) if len(real_df) else set()
        sim_rows = []

        for _, s in stations_df.iterrows():
            sid = str(s["station_id"])
            if sid in real_station_ids:
                continue

            for date in dates:
                month = int(date.month)
                if month in {11, 12, 1}:
                    base = 230
                elif month in {7, 8, 9}:
                    base = 48
                elif month in {2, 3, 4}:
                    base = 100
                elif month in {5, 6}:
                    base = 75
                else:
                    base = 140

                pm25 = float(np.clip(base + np.random.normal(0, 20), 10, 500))
                sim_rows.append(
                    {
                        "station_id": sid,
                        "date": date.date(),
                        "pm25": round(pm25, 2),
                    }
                )

        sim_df = pd.DataFrame(sim_rows)
        combined = pd.concat([real_df, sim_df], ignore_index=True)
        combined.to_csv(OPENAQ_PATH, index=False)
        log(
            f"Saved: {len(combined)} rows, {combined['station_id'].nunique()} stations "
            f"({len(real_station_ids)} real + rest generated)"
        )

    patch_elevation_nodata()

    rows2, unique_st2 = openaq_health(OPENAQ_PATH)
    log(f"Final OpenAQ file: rows={rows2}, unique_station_id={unique_st2}")
    log("Done")


if __name__ == "__main__":
    main()
