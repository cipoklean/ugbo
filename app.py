"""Ugbo web app. Speak or tap, hear your plan, put the phone away, go to the farm.

The plan is read aloud with Google's text-to-speech (gTTS, Nigerian English
accent; Hausa voice for Hausa). It returns a small MP3 that plays in any phone
browser and costs no GPU. The open-model voices in ugbo/voice.py remain for the CLI.
"""
from __future__ import annotations

import os

import tempfile

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

LANG_CHOICES = [("Pidgin", "pcm"), ("English", "en"), ("Hausa", "ha"), ("Yorùbá", "yo"), ("Igbo", "ig")]


def run(town, crops, lang, mic, typed):
    question = (typed or "").strip()
    heard = ""
    if mic is not None and not question:
        sr, audio = mic
        try:
            question = voice.transcribe(sr, audio, lang)
        except Exception as e:  # keep going without the question
            heard = f"Sorry, I could not understand the recording ({e}). Please type your question."
    if question:
        # echo back what Ugbo understood, so a misheard question can be corrected
        heard = f"You asked: \"{question}\". If that's wrong, type it and tap again."
    try:
        plan = make_plan(town or "Nsukka", crops or ["maize"], lang, question)
    except Exception as e:
        return None, f"Sorry, something went wrong: {e}", heard, ""
    return speak(plan.text, lang), plan.text, heard, facts_summary(plan)


def speak(text: str, lang: str):
    """MP3 of the plan in a natural voice; None if the voice service is down."""
    try:
        from gtts import gTTS
        f = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        if lang == "ha":
            gTTS(text, lang="ha").save(f.name)
        else:  # Nigerian English accent for English, Pidgin, Yoruba and Igbo text
            gTTS(text, lang="en", tld="com.ng").save(f.name)
        return f.name
    except Exception as e:
        print(f"Voice unavailable: {e}", flush=True)
        return None




with gr.Blocks(title="Ugbo - talk to your farm") as demo:
    gr.Markdown("# 🌱 Ugbo\n**Ask about your farm by voice. Hear this week's plan. Then go to the field.**")
    with gr.Row():
        town = gr.Textbox(label="Town or village", value="Nsukka")
        lang = gr.Radio(LANG_CHOICES, value="pcm", label="Language")
    crops = gr.CheckboxGroup(CROPS, value=["maize", "cassava"], label="What do you grow?")
    mic = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Tap and ask your question (optional)")
    typed = gr.Textbox(label="...or type it (optional)", placeholder="Rain don start, make I plant maize now?")
    go = gr.Button("Get my week plan", variant="primary", size="lg")
    spoken = gr.Audio(label="🔊 Listen to your plan", type="filepath", autoplay=True)
    text = gr.Textbox(label="Your plan", lines=6)
    heard = gr.Textbox(label="What Ugbo understood", lines=1)
    facts = gr.Textbox(label="Rain facts used", lines=2)
    gr.Markdown("Runs on open models: NaijaVox-2.0 (listening), Gemma (plan), "
                "and Google text-to-speech to read the plan aloud.")
    go.click(run, [town, crops, lang, mic, typed], [spoken, text, heard, facts])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green"))
