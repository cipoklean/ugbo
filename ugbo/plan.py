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
            text = rule_plan(facts, sig, crops, lang)
    except Exception as e:  # e.g. ZeroGPU quota used up: still answer from the rules
        print(f"Gemma unavailable, using rule-based plan: {e}", flush=True)
        text = rule_plan(facts, sig, crops, lang)
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


def rule_plan(facts: RainFacts, sig: Signals, crops: list[str], lang: str = "en") -> str:
    """A plain plan built only from the rules, used when the model can't run.

    Pidgin is written by hand; Hausa, Yoruba and Igbo are machine-translated
    from the English plan (Google Translate), falling back to English.
    """
    if lang == "pcm":
        return _pidgin_plan(facts, sig)
    text = _english_plan(facts, sig)
    if lang in ("ha", "yo", "ig"):
        return translate(text, lang)
    return text


def translate(text: str, lang: str) -> str:
    try:
        from deep_translator import GoogleTranslator
        out = GoogleTranslator(source="en", target=lang).translate(text)
        return out or text
    except Exception as e:
        print(f"Translation unavailable, answering in English: {e}", flush=True)
        return text


_SOIL_PCM = {"wet": "don wet well well", "moist": "get small water", "drying": "dey dry small small",
             "dry": "don dry"}
_SURE_PCM = {"sure": "I sure", "fairly sure": "I sure small", "not sure": "I no too sure"}
_JOB_PCM = {"plant": "plant", "harvest": "harvest"}


def _pidgin_plan(facts: RainFacts, sig: Signals) -> str:
    parts = [f"See your week for {facts.place.name}.",
             f"Rain wey fall for the last {len(facts.past)} days: {facts.past_total:.0f} mm. "
             f"Rain wey go fall for the next {len(facts.forecast)} days: {facts.forecast_total:.0f} mm. "
             f"The soil {_SOIL_PCM.get(sig.soil, sig.soil)}. How sure: {_SURE_PCM.get(sig.confidence, sig.confidence)}."]
    for c, jobs in sig.crop_jobs.items():
        if jobs:
            parts.append(f"For your {c}, na time to {' and '.join(_JOB_PCM[j] for j in jobs)}.")
        else:
            parts.append(f"E never reach time to plant or harvest {c} for this side; "
                         "just weed am, check for insect, and take care of am.")
    if sig.heavy_rain_days:
        parts.append(f"Heavy rain dey come on {', '.join(sig.heavy_rain_days)}. No put fertilizer "
                     "or spray chemical the day before, rain go wash am comot.")
    if sig.dry_days_ahead >= 5:
        parts.append(f"{sig.dry_days_ahead} out of the next 7 days go dry. E good for harvest, "
                     "to dry your grain and to weed; no plant seed unless you fit water am.")
    if sig.longest_dry_spell_past >= 7:
        parts.append(f"Rain no fall for {sig.longest_dry_spell_past} days recently. "
                     "Check your small plants make sure dem no dey die.")
    if sig.soil == "wet":
        parts.append("The ground wet too much. Watch for water wey dey stand and for disease; "
                     "harvest your root crops before dem rot.")
    if sig.confidence == "not sure":
        parts.append("The rain never settle well. E fit be false start, so wait small before you plant.")
    if sig.heavy_rain_days:
        today = "Clear your gutter and ridges before the heavy rain."
    elif sig.confidence == "not sure":
        today = "No plant yet. Prepare your ridges first."
    elif any("plant" in j for j in sig.crop_jobs.values()) and sig.soil in ("wet", "moist"):
        today = "The soil ready, so plant today."
    elif sig.dry_days_ahead >= 5:
        today = "Use the dry days to weed and harvest."
    else:
        today = "Waka round your farm, check for weed and insect."
    parts.append(f"The main work for today: {today}")
    parts.append("Remember say na weather model give this rain data, no be gauge for your farm.")
    return " ".join(parts)


def _english_plan(facts: RainFacts, sig: Signals) -> str:
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
