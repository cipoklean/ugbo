---
title: Ugbo
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🌱 Ugbo — talk to your farm

*Ugbo* means "farm" in Igbo.

Ugbo is a voice-first weekly field plan for smallholder farmers in Nigeria.
A farmer taps one button and asks, in Pidgin, Hausa, Yorùbá, Igbo or English,
something like *"Rain don start, make I plant maize now?"* Ugbo checks the real
rain at their village, works out what the season allows for their crops, and
**speaks back** a short plan that ends with the one job to do in the field today.

The screen is the shortest part of the experience. Ask, listen, go to the farm.

## How it works

```
 🎤 voice ──► NaijaVox-2.0 (open ASR) ──► question text
                                              │
 town ──► Open-Meteo / MET Norway / NASA POWER ──► rain facts (14 days back, 7 ahead)
                                              │
 crops ──► plain-rule Nigerian crop calendar ──► this week's signals
                                              │
                         Gemma (open weights) ──► plan in the farmer's language
                                              │
                         MMS-TTS (open voice) ──► 🔊 spoken plan
```

* **Rules decide the facts, Gemma decides the words.** Planting windows,
  heavy-rain warnings and dry-spell checks come from transparent rules in
  `ugbo/season.py`. Gemma only explains them, so it can't invent a planting date.
* **Free to run.** Every data source is free and keyless.
* **Swappable model.** `UGBO_LLM=ollama` runs Gemma fully on your own machine;
  `hf` or `aistudio` use hosted Gemma when you don't have the hardware.

## Run it

```bash
pip install -r requirements.txt

# pick one Gemma backend
export UGBO_LLM=ollama            # local: ollama pull gemma3:4b
# export HF_TOKEN=...             # Hugging Face Inference (google/gemma-3-27b-it)
# export GEMINI_API_KEY=...       # Google AI Studio (gemma-3-27b-it)

python app.py                     # web app with mic
python -m ugbo.cli Nsukka maize cassava --lang pcm --ask "Make I plant maize now?"
pytest -q
```

Optional: `UGBO_MODEL` overrides the Gemma model name, `UGBO_ASR_MODEL` the speech model.

## Languages

| Language | Speak to Ugbo | Ugbo speaks back |
|---|---|---|
| Nigerian Pidgin | ✅ | ✅ |
| English | ✅ | ✅ |
| Hausa | ✅ | ✅ |
| Yorùbá | ✅ | ✅ |
| Igbo | ✅ (less accurate) | text only, since no open Igbo voice model exists yet |

## Open models used

* [NaijaVox-2.0](https://huggingface.co/Axiveri/NaijaVox-2.0) (Apache-2.0): Whisper-large-v3 fine-tuned for Nigerian languages
* [Gemma](https://ai.google.dev/gemma): open-weight LLM from Google
* [MMS-TTS](https://huggingface.co/facebook/mms-tts-pcm) (CC-BY-NC-4.0): Meta's Massively Multilingual Speech voices

## Limits

The crop calendar is a rule of thumb for three broad zones (South, Middle Belt,
North), not a replacement for your local extension officer. Rain data comes from
weather models, not a gauge on your farm.

## License

MIT for the code. Each model keeps its own license.
