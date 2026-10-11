"""Gemma turns the rule-based facts into a short field plan in the farmer's language.

Swappable backends, all running open-weight Gemma:
  * local    - Gemma loaded with transformers in-process (default on a
               Hugging Face ZeroGPU Space; also works on any CUDA machine)
  * ollama   - fully local, e.g. `ollama pull gemma3:4b` (works offline)
  * hf       - Hugging Face Inference router (OpenAI-compatible), HF_TOKEN
  * aistudio - Google AI Studio hosted Gemma (free tier), GEMINI_API_KEY

Pick with UGBO_LLM=local|ollama|hf|aistudio, or leave unset to auto-detect.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

import requests

from .gpu import gpu

from .season import ZONES, Signals
from .weather import RainFacts

LANGS = {
    "en": "simple Nigerian English",
    "pcm": "Nigerian Pidgin",
    "ha": "Hausa",
    "yo": "Yoruba",
    "ig": "Igbo",
}

SYSTEM = (
    "You are Ugbo, a farm extension officer who talks to smallholder farmers in Nigeria. "
    "Speak warmly and plainly, like a trusted neighbour. Your answer will be read aloud, "
    "so: no markdown, no bullet symbols, no emojis, short sentences, at most 120 words. "
    "Use ONLY the facts given. Never invent dates, rainfall numbers, prices or chemicals. "
    "End by telling the farmer the single most important job to do in the field today."
)


def build_prompt(facts: RainFacts, sig: Signals, crops: list[str], lang: str,
                 question: str = "") -> str:
    jobs = "; ".join(f"{c}: {', '.join(j) if j else 'care only'}" for c, j in sig.crop_jobs.items())
    lines = [
        f"Reply in {LANGS.get(lang, 'simple English')}.",
        f"Farm location: {facts.place.name}, {facts.place.state}. Zone: {ZONES[sig.zone]}.",
        f"Crops: {', '.join(crops)}. Usual jobs this month -> {jobs}.",
        f"Rain in last {len(facts.past)} days: {facts.past_total} mm. "
        f"Rain forecast next {len(facts.forecast)} days: {facts.forecast_total} mm "
        f"(by day: {json.dumps(facts.forecast)}).",
        f"Soil moisture estimate: {sig.soil}.",
        "Facts you must respect:",
        *[f"- {f}" for f in sig.flags],
    ]
    if question.strip():
        lines.append(f'The farmer asked: "{question.strip()}". Answer that first, then give the week plan.')
    else:
        lines.append("Give this week's field plan.")
    return "\n".join(lines)


def _backend() -> str:
    b = os.getenv("UGBO_LLM", "").lower()
    if b:
        return b
    if os.getenv("GEMINI_API_KEY"):
        return "aistudio"
    if os.getenv("SPACE_ID"):  # running on a Hugging Face Space
        return "local"
    if os.getenv("HF_TOKEN"):
        return "hf"
    return "ollama"


LOCAL_MODEL = os.getenv("UGBO_LOCAL_MODEL", "unsloth/gemma-3-12b-it")  # ungated mirror of google/gemma-3-12b-it


@lru_cache(maxsize=1)
def _local():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(LOCAL_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(
        LOCAL_MODEL, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    return processor, model.eval()


@gpu(duration=90)
def _generate_local(prompt: str) -> str:
    import torch

    processor, model = _local()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt").to(model.device)
    n = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=350, do_sample=False)
    return processor.decode(out[0][n:], skip_special_tokens=True).strip()


def generate(prompt: str) -> str:
    b = _backend()
    if b == "local":
        return _generate_local(prompt)
    if b == "ollama":
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        model = os.getenv("UGBO_MODEL", "gemma3:4b")
        r = requests.post(f"{host}/api/chat", json={
            "model": model, "stream": False, "options": {"temperature": 0.3},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt}],
        }, timeout=180)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()

    if b == "hf":
        model = os.getenv("UGBO_MODEL", "google/gemma-3-27b-it")
        r = requests.post("https://router.huggingface.co/v1/chat/completions",
                          headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
                          json={"model": model, "temperature": 0.3, "max_tokens": 400,
                                "messages": [{"role": "system", "content": SYSTEM},
                                             {"role": "user", "content": prompt}]},
                          timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    if b == "aistudio":
        model = os.getenv("UGBO_MODEL", "gemma-3-27b-it")
        # Gemma on AI Studio has no system role, so the instructions lead the user turn
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": os.environ["GEMINI_API_KEY"]},
            json={"contents": [{"role": "user", "parts": [{"text": SYSTEM + "\n\n" + prompt}]}],
                  "generationConfig": {"temperature": 0.3, "maxOutputTokens": 400}},
            timeout=120)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()

    raise ValueError(f"Unknown UGBO_LLM backend {b!r}")
