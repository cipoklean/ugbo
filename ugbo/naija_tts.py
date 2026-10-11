"""Natural Nigerian voices with Naija TTS (PlotweaverAI/naija-tts, MIT).

A LoRA fine-tune of Resemble AI's Chatterbox Multilingual on Hausa, Yoruba,
Igbo and Nigerian English. It sounds far less robotic than the small MMS-TTS
models and gives Igbo a voice at all, but it is a 0.5B model that wants a GPU,
so Ugbo uses it on the Space and keeps MMS-TTS as the CPU/offline fallback.

`chatterbox-tts` pins old torch/transformers/gradio versions, so it is
installed with --no-deps (its real runtime deps are in requirements.txt).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from functools import lru_cache

import numpy as np

from .gpu import gpu

REPO = os.getenv("UGBO_NAIJA_TTS", "PlotweaverAI/naija-tts")
FILES = ["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
         "grapheme_mtl_merged_expanded_v1.json", "conds.pt"]

# Ugbo language code -> Naija TTS language id. Pidgin is read as Nigerian
# English, which shares its accent and most of its words. Yoruba stays on
# MMS-TTS: the model card reports 45% CER for Yoruba, against 9-15% for the rest.
LANG = {"pcm": "en-ng", "en": "en-ng", "ha": "ha", "ig": "ig"}
EXTRA_LANGS = {"ha": "Hausa", "yo": "Yoruba", "ig": "Igbo", "en-ng": "Nigerian English"}


def enabled() -> bool:
    return os.getenv("UGBO_TTS", "naija" if os.getenv("SPACE_ID") else "mms") == "naija"


def supports(lang: str) -> bool:
    return enabled() and lang in LANG


def _pip(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def ensure_installed() -> None:
    try:
        import torchaudio  # noqa: F401
    except Exception:
        import torch
        base, _, cuda = torch.__version__.partition("+")
        extra = ["--index-url", f"https://download.pytorch.org/whl/{cuda}"] if cuda else []
        _pip("--no-deps", f"torchaudio=={base}", *extra)
    try:
        import chatterbox  # noqa: F401
    except Exception:
        _pip("--no-deps", "chatterbox-tts==0.1.7")


@lru_cache(maxsize=1)
def _model():
    import torch
    from huggingface_hub import hf_hub_download

    ensure_installed()
    import chatterbox.mtl_tts as mtl

    for code, name in EXTRA_LANGS.items():
        mtl.SUPPORTED_LANGUAGES[code] = name
    path = None
    for f in FILES:
        path = hf_hub_download(REPO, f)
    device = "cuda" if (torch.cuda.is_available() or os.getenv("SPACE_ID")) else "cpu"
    return mtl.ChatterboxMultilingualTTS.from_local(os.path.dirname(path), device=device)


def _number_words(text: str) -> str:
    """Spell out small numbers and units so the voice reads them naturally."""
    ones = "zero one two three four five six seven eight nine ten eleven twelve thirteen " \
           "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    tens = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()

    def words(n: int) -> str:
        if n < 20:
            return ones[n]
        if n < 100:
            return tens[n // 10] + ("" if n % 10 == 0 else " " + ones[n % 10])
        if n < 1000:
            rest = n % 100
            return ones[n // 100] + " hundred" + ("" if not rest else " and " + words(rest))
        return str(n)

    text = re.sub(r"(\d)\s*°C", r"\1 degrees", text)
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    text = re.sub(r"(\d)\s*mm\b", r"\1 millimetres", text)
    text = re.sub(r"(\d+)\.(\d+)", lambda m: f"{words(int(m[1]))} point {' '.join(ones[int(d)] for d in m[2])}", text)
    return re.sub(r"\d+", lambda m: words(int(m[0])), text)


def _chunks(text: str, limit: int = 220) -> list[str]:
    """Join sentences into chunks short enough that Chatterbox won't cut them off."""
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")):
        s = s.strip(" -*•")
        if not s:
            continue
        if cur and len(cur) + len(s) > limit:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return out


@gpu(duration=60)
def speak(text: str, lang: str) -> tuple[int, np.ndarray] | None:
    import torch

    model = _model()
    pause = np.zeros(int(model.sr * 0.3), dtype=np.float32)
    waves = []
    for chunk in _chunks(_number_words(text)):
        with torch.no_grad():
            wav = model.generate(chunk, language_id=LANG[lang])
        waves += [wav.squeeze().float().cpu().numpy(), pause]
    if not waves:
        return None
    return model.sr, np.concatenate(waves)
