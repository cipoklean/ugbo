"""Glue: town + crops + language (+ optional question) -> field plan."""
from __future__ import annotations

import re
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
        bad = check(text, prompt)
        if bad:
            print(f"Gemma output failed the fact check ({bad}); using rule-based plan", flush=True)
            text = rule_plan(facts, sig, crops)
    except Exception as e:  # e.g. ZeroGPU quota used up: still answer from the rules
        print(f"Gemma unavailable, using rule-based plan: {e}", flush=True)
        text = rule_plan(facts, sig, crops)
    return Plan(text, facts, sig, prompt)


_NUM = re.compile(r"\d+(?:\.\d+)?")


def check(text: str, prompt: str) -> str:
    """Every number the model says must come from the facts it was given.

    The rules own the facts; if Gemma invents a rainfall figure, a date or a
    quantity, its words are thrown away and the templated plan is used.
    Returns "" when the text passes, or the first problem found.
    """
    allowed = {float(n) for n in _NUM.findall(prompt)}
    allowed |= {round(a) for a in allowed} | set(range(0, 13))  # small counts, months
    for n in _NUM.findall(text):
        v = float(n)
        if v not in allowed and round(v) not in allowed:
            return f"number {n} is not in the facts"
    return ""


def rule_plan(facts: RainFacts, sig: Signals, crops: list[str]) -> str:
    """A plain plan built only from the rules, used when the model can't run."""
    parts = [f"Here is your week in {facts.place.name}.",
             f"Rain in the last {len(facts.past)} days: {facts.past_total:.0f} mm. "
             f"Expected in the next {len(facts.forecast)} days: {facts.forecast_total:.0f} mm. "
             f"The soil is {sig.soil}. How sure: {sig.confidence}."]
    for c, jobs in sig.crop_jobs.items():
        if jobs:
            parts.append(f"For {c}, this is the time to {' and '.join(jobs)}.")
    parts += [f for f in sig.flags if not f.startswith("How sure")]
    if sig.heavy_rain_days:
        today = "Clear your drainage channels before the heavy rain."
    elif sig.confidence == "not sure":
        today = "Wait for the rains to settle before you plant. Prepare your ridges instead."
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
