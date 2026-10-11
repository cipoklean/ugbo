"""Voice in, voice out, so farmers who can't read or type can still use Ugbo.

Speech-to-text: NaijaVox-2.0 (Apache-2.0, Whisper-large-v3 fine-tuned on
Yoruba, Hausa, Igbo, Nigerian Pidgin and Nigerian English).
Text-to-speech: Meta MMS-TTS (VITS), one small model per language.

Both are open-weight and run on a CPU, so no voice data leaves the machine.
Models are loaded lazily and cached on first use.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache

import numpy as np

from .gpu import device as _device, gpu

ASR_MODEL = os.getenv("UGBO_ASR_MODEL", "Axiveri/NaijaVox-2.0")
ASR_LANG_TOKEN = {"en": "<|en|>", "pcm": "<|pcm|>", "ha": "<|ha|>", "yo": "<|yo|>", "ig": "<|ig|>"}

# MMS-TTS has no public Igbo voice, so Igbo answers are shown as text only.
TTS_MODEL = {
    "en": "facebook/mms-tts-eng",
    "pcm": "facebook/mms-tts-pcm",
    "ha": "facebook/mms-tts-hau",
    "yo": "facebook/mms-tts-yor",
}


@lru_cache(maxsize=1)
def _asr():
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    processor = WhisperProcessor.from_pretrained(ASR_MODEL)
    on_gpu = torch.cuda.is_available() or bool(os.getenv("SPACE_ID"))
    dtype = torch.float16 if on_gpu else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
        ASR_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(_device()).eval()
    return processor, model


def _to_16k_mono(sr: int, audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    peak = np.abs(audio).max() or 1.0
    if peak > 1.0:  # int16 from the browser mic
        audio = audio / 32768.0
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio[: 16000 * 30]  # Whisper window: 30 s is plenty for a question


@gpu(duration=20)
def transcribe(sr: int, audio: np.ndarray, lang: str = "pcm") -> str:
    import torch

    processor, model = _asr()
    vocab = processor.tokenizer.get_vocab()
    audio = _to_16k_mono(sr, audio)
    feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features
    feats = feats.to(model.device, dtype=model.dtype)
    prompt = [vocab["<|startoftranscript|>"], vocab[ASR_LANG_TOKEN.get(lang, "<|en|>")],
              vocab["<|transcribe|>"], vocab["<|notimestamps|>"]]
    with torch.no_grad():
        out = model.generate(feats, decoder_input_ids=torch.tensor([prompt], device=model.device),
                             max_new_tokens=200)
    return processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()


@lru_cache(maxsize=4)
def _tts(lang: str):
    from transformers import AutoTokenizer, VitsModel

    name = TTS_MODEL[lang]
    return AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name)


def can_speak(lang: str) -> bool:
    return lang in TTS_MODEL


def speak(text: str, lang: str) -> tuple[int, np.ndarray] | None:
    """Return (sample_rate, waveform) or None when no voice exists for lang."""
    if not can_speak(lang):
        return None
    import torch

    tok, model = _tts(lang)
    chunks = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    pause = np.zeros(int(model.config.sampling_rate * 0.25), dtype=np.float32)
    waves = []
    for s in chunks:
        inputs = tok(s, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            continue
        with torch.no_grad():
            waves.append(model(**inputs).waveform[0].numpy())
        waves.append(pause)
    if not waves:
        return None
    return model.config.sampling_rate, np.concatenate(waves)
