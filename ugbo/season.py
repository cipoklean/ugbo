"""Plain-rule crop calendar for Nigeria, plus this week's field signals.

These are rules of thumb drawn from common extension advice (IITA, FAO,
state ADPs). They are deliberately simple and transparent: the rules decide
WHAT the facts are, the language model only decides HOW to say them. That
keeps the model from inventing planting dates.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from .weather import RainFacts

# Zones by latitude. Rainy season gets shorter the further north you go.
ZONES = {
    "south": "Southern Nigeria (two rainy seasons, Mar to Nov, short 'August break')",
    "middle": "Middle Belt (one long rainy season, Apr to Oct)",
    "north": "Northern Nigeria (one short rainy season, Jun to Sep)",
}


def zone_for(lat: float) -> str:
    if lat < 8.0:
        return "south"
    if lat < 10.5:
        return "middle"
    return "north"


# months (1-12) when each job is usually right, per zone
CALENDAR: dict[str, dict[str, dict[str, list[int]]]] = {
    "maize": {
        "south": {"plant": [3, 4, 8, 9], "harvest": [6, 7, 11, 12]},
        "middle": {"plant": [4, 5, 6], "harvest": [8, 9, 10]},
        "north": {"plant": [6, 7], "harvest": [9, 10]},
    },
    "cassava": {
        "south": {"plant": [4, 5, 6, 7, 9, 10], "harvest": list(range(1, 13))},
        "middle": {"plant": [5, 6, 7], "harvest": list(range(1, 13))},
        "north": {"plant": [6, 7], "harvest": [11, 12, 1, 2, 3]},
    },
    "yam": {
        "south": {"plant": [2, 3, 4], "harvest": [8, 9, 10, 11, 12]},
        "middle": {"plant": [3, 4, 5], "harvest": [9, 10, 11, 12]},
        "north": {"plant": [5, 6], "harvest": [10, 11, 12]},
    },
    "rice": {
        "south": {"plant": [5, 6, 7], "harvest": [9, 10, 11]},
        "middle": {"plant": [5, 6, 7], "harvest": [10, 11]},
        "north": {"plant": [6, 7], "harvest": [10, 11]},
    },
    "cowpea": {
        "south": {"plant": [8, 9], "harvest": [11, 12]},
        "middle": {"plant": [7, 8], "harvest": [10, 11]},
        "north": {"plant": [7, 8], "harvest": [10, 11]},
    },
    "sorghum": {
        "south": {"plant": [], "harvest": []},
        "middle": {"plant": [5, 6], "harvest": [10, 11]},
        "north": {"plant": [6, 7], "harvest": [10, 11, 12]},
    },
    "groundnut": {
        "south": {"plant": [4, 5], "harvest": [7, 8]},
        "middle": {"plant": [5, 6], "harvest": [9, 10]},
        "north": {"plant": [6, 7], "harvest": [9, 10]},
    },
    "tomato": {
        "south": {"plant": [3, 4, 9, 10], "harvest": [6, 7, 12, 1]},
        "middle": {"plant": [9, 10, 11], "harvest": [12, 1, 2, 3]},
        "north": {"plant": [10, 11], "harvest": [1, 2, 3, 4]},  # dry-season, irrigated
    },
    "pepper": {
        "south": {"plant": [3, 4, 5], "harvest": [7, 8, 9, 10]},
        "middle": {"plant": [5, 6], "harvest": [9, 10, 11]},
        "north": {"plant": [6, 7, 10], "harvest": [9, 10, 11, 1, 2]},
    },
    "okra": {
        "south": {"plant": [3, 4, 5, 8, 9], "harvest": [5, 6, 7, 10, 11]},
        "middle": {"plant": [4, 5, 6], "harvest": [6, 7, 8, 9]},
        "north": {"plant": [6, 7], "harvest": [8, 9, 10]},
    },
}

CROPS = sorted(CALENDAR)


@dataclass
class Signals:
    zone: str
    month: int
    heavy_rain_days: list[str]     # >= 25 mm forecast
    dry_days_ahead: int            # forecast days with < 1 mm
    longest_dry_spell_past: int    # consecutive past days < 1 mm
    soil: str                      # "wet", "moist", "drying", "dry"
    crop_jobs: dict[str, list[str]]  # crop -> ["plant", "harvest"]
    flags: list[str]               # plain-English facts the model must respect


def _longest_dry(series: list[float]) -> int:
    best = run = 0
    for mm in series:
        run = run + 1 if mm < 1.0 else 0
        best = max(best, run)
    return best


def signals(facts: RainFacts, crops: list[str], today: dt.date | None = None) -> Signals:
    today = today or dt.date.today()
    zone = zone_for(facts.place.lat)
    month = today.month

    heavy = [d for d, mm in facts.forecast.items() if mm >= 25]
    dry_ahead = sum(1 for mm in facts.forecast.values() if mm < 1.0)
    dry_past = _longest_dry(list(facts.past.values()))

    wb = facts.water_balance
    if wb > 20:
        soil = "wet"
    elif wb > -15:
        soil = "moist"
    elif wb > -40:
        soil = "drying"
    else:
        soil = "dry"

    jobs: dict[str, list[str]] = {}
    for c in crops:
        cal = CALENDAR.get(c, {}).get(zone, {})
        jobs[c] = [job for job in ("plant", "harvest") if month in cal.get(job, [])]

    flags: list[str] = []
    if heavy:
        flags.append(f"Heavy rain (25 mm or more) expected on {', '.join(heavy)}. "
                     "Do not apply fertilizer or spray chemicals the day before; it will wash off. "
                     "Clear drainage channels and ridges before then.")
    if dry_ahead >= 5:
        flags.append(f"{dry_ahead} of the next 7 days look dry. Good for harvesting, drying "
                     "grain and weeding; hold off planting seeds unless you can water.")
    if dry_past >= 7:
        flags.append(f"There was a dry spell of {dry_past} days recently. Check young plants for wilting.")
    if soil in ("dry", "drying") and any("plant" in j for j in jobs.values()):
        flags.append("Soil is drying out. Plant only after a soaking rain of about 20 mm or more.")
    if soil == "wet":
        flags.append("Soil is very wet. Watch for waterlogging and fungal disease; harvest roots before they rot.")
    for c, j in jobs.items():
        if not j:
            flags.append(f"This is not the usual time to plant or harvest {c} in this zone; "
                         "focus on weeding, pest checks and care.")

    return Signals(zone, month, heavy, dry_ahead, dry_past, soil, jobs, flags)
