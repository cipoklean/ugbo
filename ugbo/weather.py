"""Rain facts for one farm: the last 14 days and the next 7.

Primary source is Open-Meteo (free, no key). If it is down or rate-limited,
we fall back to MET Norway for the forecast and NASA POWER for past rain.
All three are free and keyless, so Ugbo costs nothing to run.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass, field

import requests

UA = {"User-Agent": "ugbo/0.1 (+https://github.com/cipoklean/ugbo)"}
TIMEOUT = 15
DEFAULT_ET0 = 4.5  # mm/day, a typical Nigerian figure when no ET0 is available


@dataclass
class Place:
    name: str
    lat: float
    lon: float
    state: str = ""


@dataclass
class RainFacts:
    place: Place
    past: dict[str, float]          # date -> mm, last ~14 days
    forecast: dict[str, float]      # date -> mm, next ~7 days
    et0: float = DEFAULT_ET0        # mean daily evapotranspiration, mm
    source: str = ""
    tmax: dict[str, float] = field(default_factory=dict)

    @property
    def past_total(self) -> float:
        return round(sum(self.past.values()), 1)

    @property
    def forecast_total(self) -> float:
        return round(sum(self.forecast.values()), 1)

    @property
    def water_balance(self) -> float:
        """Rain minus evaporation over the past window. Negative = drying soil."""
        return round(self.past_total - self.et0 * len(self.past), 1)


def geocode(town: str) -> Place:
    r = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": town, "count": 1, "countryCode": "NG"},
        headers=UA, timeout=TIMEOUT,
    )
    r.raise_for_status()
    hits = r.json().get("results") or []
    if not hits:
        raise ValueError(f"Could not find a town called {town!r} in Nigeria")
    h = hits[0]
    return Place(h["name"], h["latitude"], h["longitude"], h.get("admin1", ""))


def _open_meteo(p: Place) -> RainFacts:
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": p.lat, "longitude": p.lon,
            "daily": "precipitation_sum,et0_fao_evapotranspiration,temperature_2m_max",
            "past_days": 14, "forecast_days": 7, "timezone": "Africa/Lagos",
        },
        headers=UA, timeout=TIMEOUT,
    )
    r.raise_for_status()
    d = r.json()
    if d.get("error"):
        raise RuntimeError(d.get("reason"))
    d = d["daily"]
    today = dt.date.today().isoformat()
    past, fc, tmax, et = {}, {}, {}, []
    for day, rain, e, t in zip(d["time"], d["precipitation_sum"],
                               d["et0_fao_evapotranspiration"], d["temperature_2m_max"]):
        rain = rain or 0.0
        (past if day < today else fc)[day] = round(rain, 1)
        if day < today and e is not None:
            et.append(e)
        if t is not None:
            tmax[day] = t
    et0 = round(sum(et) / len(et), 2) if et else DEFAULT_ET0
    return RainFacts(p, past, fc, et0, "Open-Meteo", tmax)


def _met_no_forecast(p: Place) -> dict[str, float]:
    r = requests.get(
        "https://api.met.no/weatherapi/locationforecast/2.0/compact",
        params={"lat": round(p.lat, 2), "lon": round(p.lon, 2)},
        headers=UA, timeout=TIMEOUT,
    )
    r.raise_for_status()
    daily: dict[str, float] = defaultdict(float)
    # met.no gives hourly steps for ~2.5 days, then 6-hourly steps. Each step's
    # "next_1_hours" (or, later, "next_6_hours") block covers the gap to the
    # next step, so summing the finest block per step never double counts.
    for step in r.json()["properties"]["timeseries"]:
        t = dt.datetime.fromisoformat(step["time"].replace("Z", "+00:00"))
        local = (t + dt.timedelta(hours=1)).date().isoformat()  # WAT = UTC+1
        data = step["data"]
        block = data.get("next_1_hours") or data.get("next_6_hours")
        if block:
            daily[local] += block["details"].get("precipitation_amount", 0.0)
    return {k: round(v, 1) for k, v in sorted(daily.items())[:7]}


def _nasa_power_past(p: Place, days: int = 14) -> dict[str, float]:
    end = dt.date.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days=days + 4)  # POWER lags a few days
    r = requests.get(
        "https://power.larc.nasa.gov/api/temporal/daily/point",
        params={"parameters": "PRECTOTCORR", "community": "AG",
                "longitude": p.lon, "latitude": p.lat,
                "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"),
                "format": "JSON"},
        headers=UA, timeout=30,
    )
    r.raise_for_status()
    raw = r.json()["properties"]["parameter"]["PRECTOTCORR"]
    vals = {f"{k[:4]}-{k[4:6]}-{k[6:]}": round(v, 1) for k, v in raw.items() if v >= 0}
    return dict(sorted(vals.items())[-days:])


def rain_facts(place: Place) -> RainFacts:
    try:
        return _open_meteo(place)
    except Exception:
        pass
    fc = _met_no_forecast(place)
    try:
        past = _nasa_power_past(place)
    except Exception:
        past = {}
    return RainFacts(place, past, fc, DEFAULT_ET0, "MET Norway + NASA POWER")
