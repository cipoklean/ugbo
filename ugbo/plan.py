"""Glue: town + crops + language (+ optional question) -> field plan."""
from __future__ import annotations

from dataclasses import dataclass

from . import llm
from .season import Signals, signals
from .weather import RainFacts, geocode, rain_facts


@dataclass
class Plan:
    text: str
    facts: RainFacts
    signals: Signals
    prompt: str


def make_plan(town: str, crops: list[str], lang: str = "pcm", question: str = "") -> Plan:
    if not crops:
        raise ValueError("Pick at least one crop")
    place = geocode(town)
    facts = rain_facts(place)
    sig = signals(facts, crops)
    prompt = llm.build_prompt(facts, sig, crops, lang, question)
    try:
        text = llm.generate(prompt)
    except Exception as e:  # e.g. ZeroGPU quota used up: still answer from the rules
        print(f"Gemma unavailable, using rule-based plan: {e}", flush=True)
        text = rule_plan(facts, sig, crops)
    return Plan(text, facts, sig, prompt)


def rule_plan(facts: RainFacts, sig: Signals, crops: list[str]) -> str:
    """A plain plan built only from the rules, used when the model can't run."""
    parts = [f"Here is your week in {facts.place.name}.",
             f"Rain in the last {len(facts.past)} days: {facts.past_total:.0f} mm. "
             f"Expected in the next {len(facts.forecast)} days: {facts.forecast_total:.0f} mm. "
             f"The soil is {sig.soil}."]
    for c, jobs in sig.crop_jobs.items():
        if jobs:
            parts.append(f"For {c}, this is the time to {' and '.join(jobs)}.")
    parts += sig.flags
    if sig.heavy_rain_days:
        today = "Clear your drainage channels before the heavy rain."
    elif any("plant" in j for j in sig.crop_jobs.values()) and sig.soil in ("wet", "moist"):
        today = "The soil is ready, so plant today."
    elif sig.dry_days_ahead >= 5:
        today = "Use the dry days to weed and harvest."
    else:
        today = "Walk your farm and check for weeds and pests."
    parts.append(f"Most important job today: {today}")
    return " ".join(parts)


def facts_summary(p: Plan) -> str:
    f, s = p.facts, p.signals
    return (f"{f.place.name}, {f.place.state} ({s.zone} zone). "
            f"Rain last {len(f.past)} days: {f.past_total} mm. "
            f"Next {len(f.forecast)} days: {f.forecast_total} mm. "
            f"Soil: {s.soil}. Source: {f.source}.")
