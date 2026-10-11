---
title: Ugbo
emoji: 🌱
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
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

**Try it:** [huggingface.co/spaces/Cipoklean/ugbo](https://huggingface.co/spaces/Cipoklean/ugbo)

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
                         fact check ──► fallback to a templated plan if Gemma drifts
                                              │
                         phone's own voice (Web Speech API) ──► 🔊 spoken plan
```

* **Rules decide the facts, Gemma decides the words.** Planting windows,
  heavy-rain warnings and dry-spell checks come from transparent rules in
  `ugbo/season.py`. Gemma only explains them, so it can't invent a planting date.
* **Gemma's words are checked.** Every number in Gemma's answer must appear in
  the facts it was given. If Gemma invents a rainfall figure or a fertilizer
  rate, its answer is dropped and a templated plan built from the rules is used.
  The same template answers when the model is unavailable (e.g. GPU quota).
* **False starts are caught.** Early rain followed by a dry spell is the classic
  way a farmer loses a season. Ugbo only treats the rains as established after
  3+ rainy days in the last week, 20 mm+ in two weeks and no dry spell of 4+ days
  forecast. Until then it says "wait", and every plan says how sure it is.
* **Misheard questions are echoed.** Ugbo shows "You asked: ..." so the farmer
  can correct a bad transcription before acting on the answer.
* **Free to run.** Every data source is free and keyless.
* **Swappable model.** On the Hugging Face Space, Gemma 3 12B runs in-process on
  free ZeroGPU (`UGBO_LLM=local`). `ollama` runs Gemma fully on your own machine,
  and `hf` or `aistudio` use hosted Gemma when you don't have the hardware.

## Run it

```bash
pip install torch -r requirements.txt

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

| Language | Speak to Ugbo | Plan written in | Read aloud |
|---|---|---|---|
| Nigerian Pidgin | ✅ | ✅ | phone's Nigerian English voice |
| English | ✅ | ✅ | ✅ phone voice (en-NG where installed) |
| Hausa | ✅ | ✅ | phone's Hausa voice if installed, else English voice |
| Yorùbá | ✅ | ✅ | phone's Yorùbá voice if installed, else English voice |
| Igbo | ✅ (less accurate) | ✅ | phone's Igbo voice if installed, else English voice |

The web app reads plans aloud with the phone's own text-to-speech (the browser's
Web Speech API): it sounds natural on Android, works on cheap phones and costs no
GPU. The CLI can still speak with the server-side Naija TTS / MMS-TTS voices in
`ugbo/voice.py`; note MMS-TTS is **CC-BY-NC-4.0 (non-commercial)** and Naija TTS
audio carries Resemble's Perth watermark.

## Open models used

* [NaijaVox-2.0](https://huggingface.co/Axiveri/NaijaVox-2.0) (Apache-2.0): Whisper-large-v3 fine-tuned for Nigerian languages
* [Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it): open-weight LLM from Google (the Space loads the ungated `unsloth/gemma-3-12b-it` copy of the same weights)
* [Naija TTS](https://huggingface.co/PlotweaverAI/naija-tts) (MIT): Chatterbox Multilingual fine-tuned on Hausa, Yorùbá, Igbo and Nigerian English. Its audio carries Resemble's Perth watermark
* [MMS-TTS](https://huggingface.co/facebook/mms-tts-pcm) (CC-BY-NC-4.0): Meta's Massively Multilingual Speech voices

## Limits and what's not yet verified

* **Accuracy is not yet measured.** There is no word-error-rate evaluation of
  NaijaVox on farm questions yet (Igbo is known to be weaker), and the crop
  calendar has not been checked row by row against IITA / state ADP calendars.
  Help with either is the most useful contribution (see CONTRIBUTING.md).
* **Reach.** A web app needs data and a smartphone. A WhatsApp voice-note or
  phone-call (IVR) front end would reach far more farmers; that's the next step.

The crop calendar is a rule of thumb for three broad zones (South, Middle Belt,
North), not a replacement for your local extension officer. Rain data comes from
weather models, not a gauge on your farm.

## License

MIT for the code. Each model keeps its own license.
