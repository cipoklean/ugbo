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
    return Plan(llm.generate(prompt), facts, sig, prompt)


def facts_summary(p: Plan) -> str:
    f, s = p.facts, p.signals
    return (f"{f.place.name}, {f.place.state} ({s.zone} zone). "
            f"Rain last {len(f.past)} days: {f.past_total} mm. "
            f"Next {len(f.forecast)} days: {f.forecast_total} mm. "
            f"Soil: {s.soil}. Source: {f.source}.")
