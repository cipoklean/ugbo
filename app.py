"""Ugbo web app. Speak or tap, hear your plan, put the phone away, go to the farm.

The plan is read aloud by the phone's own text-to-speech voice (the browser's
Web Speech API), which sounds natural, works on low-end Android phones and
costs no GPU. The server-side voices in ugbo/voice.py remain for the CLI.
"""
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
        return f"Sorry, something went wrong: {e}", heard, ""
    return plan.text, heard, facts_summary(plan)


# Read the plan aloud with the phone's own voice. Prefer a Nigerian voice.
READ_ALOUD = """
(text, lang) => {
  const synth = window.speechSynthesis;
  if (!synth || !text) return;
  synth.cancel();
  const want = {pcm: ["en-NG"], en: ["en-NG", "en-GB", "en"], ha: ["ha", "en-NG"],
                yo: ["yo", "en-NG"], ig: ["ig", "en-NG"]}[lang] || ["en-NG", "en"];
  const voices = synth.getVoices();
  let voice = null;
  for (const w of want) {
    voice = voices.find(v => v.lang.replace("_", "-").toLowerCase().startsWith(w.toLowerCase()));
    if (voice) break;
  }
  const u = new SpeechSynthesisUtterance(text);
  if (voice) { u.voice = voice; u.lang = voice.lang; } else { u.lang = "en-NG"; }
  u.rate = 0.95;
  synth.speak(u);
}
"""


with gr.Blocks(title="Ugbo - talk to your farm") as demo:
    gr.Markdown("# 🌱 Ugbo\n**Ask about your farm by voice. Hear this week's plan. Then go to the field.**")
    with gr.Row():
        town = gr.Textbox(label="Town or village", value="Nsukka")
        lang = gr.Radio(LANG_CHOICES, value="pcm", label="Language")
    crops = gr.CheckboxGroup(CROPS, value=["maize", "cassava"], label="What do you grow?")
    mic = gr.Audio(sources=["microphone"], type="numpy", label="🎤 Tap and ask your question (optional)")
    typed = gr.Textbox(label="...or type it (optional)", placeholder="Rain don start, make I plant maize now?")
    go = gr.Button("Get my week plan", variant="primary", size="lg")
    text = gr.Textbox(label="Your plan", lines=6)
    listen = gr.Button("🔊 Read it aloud", size="lg")
    heard = gr.Textbox(label="What Ugbo understood", lines=1)
    facts = gr.Textbox(label="Rain facts used", lines=2)
    gr.Markdown("Runs on open models: NaijaVox-2.0 (listening), Gemma (plan), "
                "and your phone's own voice to read the plan aloud.")
    go.click(run, [town, crops, lang, mic, typed], [text, heard, facts]).then(
        None, [text, lang], None, js=READ_ALOUD)
    listen.click(None, [text, lang], None, js=READ_ALOUD)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="green"))
