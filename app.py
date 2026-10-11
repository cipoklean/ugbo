"""Ugbo web app. Speak or tap, hear your plan, put the phone away, go to the farm."""
from __future__ import annotations

import os

import gradio as gr

from ugbo import llm, voice
from ugbo.plan import facts_summary, make_plan
from ugbo.season import CROPS

if os.getenv("SPACE_ID"):
    # On a ZeroGPU Space, download and place models once at startup so a
    # farmer's request never waits on a multi-GB download.
    _, asr_model = voice._asr()
    asr_model.to("cuda")
    if llm._backend() == "local":
        _, lm = llm._local()
        lm.to("cuda")
    for code in voice.TTS_MODEL:
        voice._tts(code)
    try:
        from ugbo import naija_tts
        if naija_tts.enabled():
            naija_tts._model()
    except Exception as e:  # the MMS voices still work
        print(f"Naija TTS unavailable: {e}", flush=True)

LANG_CHOICES = [("Pidgin", "pcm"), ("English", "en"), ("Hausa", "ha"), ("Yorùbá", "yo"), ("Igbo", "ig")]


def run(town, crops, lang, mic, typed):
    question = (typed or "").strip()
    heard = ""
    if mic is not None and not question:
        sr, audio = mic
        try:
            heard = voice.transcribe(sr, audio, lang)
            question = heard
        except Exception as e:  # keep going without the question
            heard = f"(Could not understand the recording: {e})"
    try:
        plan = make_plan(town or "Nsukka", crops or ["maize"], lang, question)
    except Exception as e:
        return None, f"Sorry, something went wrong: {e}", heard, ""
    audio = None
    if voice.can_speak(lang):
        try:
            audio = voice.speak(plan.text, lang)
        except Exception:
            audio = None
    return audio, plan.text, heard, facts_summary(plan)


with gr.Blocks(title="Ugbo - talk to your farm") as demo:
    gr.Markdown("# 🌱 Ugbo\n**Ask about your farm by voice. Hear this week's plan. Then go to the field.**")
    with gr.Row():
        town = gr.Textbox(label="Town or village", value="Nsukka")
        lang = gr.Radio(LANG_CHOICES, value="pcm", label="Language")
    crops = gr.CheckboxGroup(CROPS, value=["maize", "cassava"], label="What do you grow?")
    mic = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Tap and ask your question (optional)")
    typed = gr.Textbox(label="...or type it (optional)", placeholder="Rain don start, make I plant maize now?")
    go = gr.Button("Get my week plan", variant="primary", size="lg")
    spoken = gr.Audio(label="🔊 Your plan", autoplay=True, type="numpy")
    text = gr.Textbox(label="Your plan (text)", lines=6)
    heard = gr.Textbox(label="What Ugbo heard", lines=1)
    facts = gr.Textbox(label="Rain facts used", lines=2)
    gr.Markdown("Runs on open models: NaijaVox-2.0 (listening), Gemma (plan), "
                "Naija TTS and MMS-TTS (voice).")
    go.click(run, [town, crops, lang, mic, typed], [spoken, text, heard, facts])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green"))
