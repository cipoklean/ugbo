import datetime as dt

from ugbo.season import signals, zone_for
from ugbo.weather import Place, RainFacts


def facts(lat, past, fc):
    days = [f"2026-10-{i:02d}" for i in range(1, 15)]
    fdays = [f"2026-10-{i:02d}" for i in range(15, 22)]
    return RainFacts(Place("X", lat, 7.0, "S"), dict(zip(days, past)), dict(zip(fdays, fc)), 4.5)


def test_zones():
    assert zone_for(6.8) == "south"
    assert zone_for(9.0) == "middle"
    assert zone_for(12.0) == "north"


def test_heavy_rain_flag():
    s = signals(facts(6.8, [5] * 14, [0, 30, 0, 0, 0, 0, 0]), ["maize"], dt.date(2026, 10, 11))
    assert s.heavy_rain_days == ["2026-10-16"]
    assert any("fertilizer" in f for f in s.flags)


def test_dry_week_and_dry_soil():
    s = signals(facts(6.8, [0] * 14, [0] * 7), ["maize"], dt.date(2026, 9, 1))
    assert s.soil == "dry"
    assert s.dry_days_ahead == 7
    assert s.longest_dry_spell_past == 14
    assert "plant" in s.crop_jobs["maize"]
    assert any("soaking rain" in f for f in s.flags)


def test_off_season_crop():
    s = signals(facts(12.0, [5] * 14, [5] * 7), ["sorghum"], dt.date(2026, 2, 1))
    assert s.crop_jobs["sorghum"] == []
    assert any("not the usual time" in f for f in s.flags)
