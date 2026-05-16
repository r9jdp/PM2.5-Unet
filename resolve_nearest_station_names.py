from __future__ import annotations

import math
import os
import time
from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = BASE_DIR / "pm25_delhi_bundle"
STATIONS_PATH = BUNDLE_DIR / "stations_urban.csv"
OUT_PATH = BUNDLE_DIR / "station_name_lookup.csv"
ENV_PATH = BASE_DIR / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def fetch_openaq_locations(api_key: str) -> list[dict]:
    headers = {"X-API-Key": api_key}
    base_url = "https://api.openaq.org/v3/locations"
    params = {
        "coordinates": "28.6139,77.2090",
        "radius": 100000,
        "limit": 1000,
        "parameter": "pm25",
    }
    resp = requests.get(base_url, params=params, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json().get("results", [])


def reverse_geocode_label(lat: float, lon: float) -> str:
    headers = {
        "User-Agent": "pm25-final-term-station-labels/1.0"
    }
    last_error = None
    for attempt in range(3):
        if attempt > 0:
            time.sleep(2.0 * attempt)
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "jsonv2",
                "zoom": 16,
                "addressdetails": 1,
            },
            headers=headers,
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            break
        last_error = resp.status_code
        if resp.status_code == 429:
            continue
        resp.raise_for_status()
    else:
        return f"Lat {lat:.4f}, Lon {lon:.4f}"

    address = data.get("address", {}) if isinstance(data, dict) else {}
    candidates = [
        address.get("suburb"),
        address.get("neighbourhood"),
        address.get("quarter"),
        address.get("city_district"),
        address.get("residential"),
        address.get("village"),
        address.get("town"),
        address.get("city"),
        data.get("name") if isinstance(data, dict) else None,
        data.get("display_name") if isinstance(data, dict) else None,
    ]
    for value in candidates:
        if value:
            return str(value).split(",")[0].strip()
    return f"{lat:.4f}, {lon:.4f}"


def extract_location_name(loc: dict) -> str:
    candidates = [
        loc.get("name"),
        loc.get("locality"),
        loc.get("city"),
    ]
    for value in candidates:
        if value:
            return str(value)
    return f"OpenAQ Location {loc.get('id', 'unknown')}"


def extract_lat_lon(loc: dict) -> tuple[float | None, float | None]:
    coords = loc.get("coordinates")
    if isinstance(coords, dict):
        lat = coords.get("latitude")
        lon = coords.get("longitude")
        if lat is not None and lon is not None:
            return float(lat), float(lon)
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    if lat is not None and lon is not None:
        return float(lat), float(lon)
    return None, None


def main() -> None:
    if not STATIONS_PATH.exists():
        raise FileNotFoundError(f"Missing required file: {STATIONS_PATH}")

    load_env_file(ENV_PATH)
    api_key = os.getenv("OPENAQ_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAQ_KEY not found. Put it in the environment or .env before running this script.")

    stations = pd.read_csv(STATIONS_PATH)
    required = {"station_id", "lat", "lon"}
    missing = required - set(stations.columns)
    if missing:
        raise ValueError(f"stations_urban.csv missing columns: {sorted(missing)}")

    rows = []
    geocode_cache: dict[tuple[float, float], str] = {}

    try:
        locations = fetch_openaq_locations(api_key)
        usable_locations = []
        for loc in locations:
            lat, lon = extract_lat_lon(loc)
            if lat is None or lon is None:
                continue
            usable_locations.append(
                {
                    "location_id": loc.get("id"),
                    "location_name": extract_location_name(loc),
                    "lat": lat,
                    "lon": lon,
                }
            )

        if not usable_locations:
            raise RuntimeError("OpenAQ returned no usable locations with coordinates.")

        location_df = pd.DataFrame(usable_locations)

        for _, station in stations[["station_id", "lat", "lon"]].drop_duplicates().iterrows():
            lat = float(station["lat"])
            lon = float(station["lon"])
            distances = location_df.apply(
                lambda x: haversine_km(lat, lon, float(x["lat"]), float(x["lon"])),
                axis=1,
            )
            nearest = location_df.loc[int(distances.idxmin())]
            rows.append(
                {
                    "station_id": station["station_id"],
                    "station_lat": lat,
                    "station_lon": lon,
                    "label_source": "openaq_nearest_location",
                    "nearest_location_id": nearest["location_id"],
                    "nearest_location_name": nearest["location_name"],
                    "nearest_location_lat": nearest["lat"],
                    "nearest_location_lon": nearest["lon"],
                    "distance_km": float(distances.min()),
                    "display_label": nearest["location_name"],
                }
            )
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        print(f"OpenAQ lookup failed with HTTP {status}. Falling back to reverse-geocoded area labels.")
        for _, station in stations[["station_id", "lat", "lon"]].drop_duplicates().iterrows():
            lat = float(station["lat"])
            lon = float(station["lon"])
            key = (round(lat, 4), round(lon, 4))
            if key not in geocode_cache:
                label = reverse_geocode_label(lat, lon)
                geocode_cache[key] = label
                time.sleep(1.2)
            else:
                label = geocode_cache[key]
            rows.append(
                {
                    "station_id": station["station_id"],
                    "station_lat": lat,
                    "station_lon": lon,
                    "label_source": "reverse_geocode_area",
                    "nearest_location_id": None,
                    "nearest_location_name": None,
                    "nearest_location_lat": None,
                    "nearest_location_lon": None,
                    "distance_km": None,
                    "display_label": label,
                }
            )

    out_df = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved lookup: {OUT_PATH}")
    print(out_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
