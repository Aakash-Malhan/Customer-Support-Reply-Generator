#!/usr/bin/env python3
"""
Customer Support Reply Generator (Styled + Templates • CPU-friendly • Model Switch)

- Default models are small, CPU-friendly (Qwen2.5-0.5B-Instruct and TinyLlama-1.1B-Chat).
- Style presets, brand voice, severity slider, safety guardrails, temperature/top-p.
- One-click templates for common intents (password reset, shipment delay, refund, account update).
- Works locally or on Hugging Face Spaces (CPU).
"""

from __future__ import annotations

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# Models (CPU-friendly)
# ---------------------------
MODELS = {
    "Qwen/Qwen2.5-0.5B-Instruct": "very fast",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "a bit slower",
}
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# ---------------------------
# Style presets & templates
# ---------------------------
STYLE_PRESETS = {
    "Banking / Fintech (formal & compliant)": {
        "tone": "formal, compliant, reassuring, concise",
        "extras": "avoid any promises; clarify next steps; never request full card details; remind about security best practices when relevant.",
    },
    "E-commerce (friendly & helpful)": {
        "tone": "friendly, concise, helpful, proactive",
        "extras": "offer clear next steps; provide links placeholders; keep sentences short.",
    },
    "Tech Support (diagnostic & structured)": {
        "tone": "diagnostic, structured, calm",
        "extras": "use step-by-step bullet points; ask one question at a time; reference a ticket id placeholder when appropriate.",
    },
    "Travel / Hospitality (empathetic & service-oriented)": {
        "tone": "empathetic, service-oriented, patient",
        "extras": "acknowledge inconvenience; provide options and ETAs where possible.",
    },
}

TEMPLATE_SNIPPETS = {
    "Password reset": (
        "Please follow these steps to reset your password:\n"
        "1) Go to the Sign-in page and select “Forgot password”.\n"
        "2) Enter your registered email and follow the link we send.\n"
        "3) Create a new strong password and sign in again.\n"
        "If you don’t receive the email, check spam or let me know so I can resend it."
    ),
    "Shipment delay": (
        "I’m sorry for the delay. I’ve requested an updated ETA from the carrier.\n"
        "Meanwhile, here’s your tracking number: <TRACKING_NUMBER>.\n"
        "You can track it at: <TRACKING_URL>.\n"
        "I’ll update you as soon as I have more details."
    ),
    "Refund request": (
        "I can help with that. I’ve opened a refund ticket: <TICKET_ID>.\n"
        "Refunds are typically processed within 5–7 business days.\n"
        "You’ll receive a confirmation email once issued."
    ),
    "Account update": (
        "For security, please confirm your account email and the specific details you want updated.\n"
        "Important: never share passwords or full card details in chat."
    ),
}

LANGUAGES = [
    "English",
    "Spanish",
    "German",
    "French",
    "Portuguese",
    "Italian",
    "Arabic",
]

# ---------------------------
# Model loader (cached)
# ---------------------------
@torch.inference_mode()
def load_pipe(model_id: str):
    """Create a CPU text-generation pipeline. Small models = fast on CPU."""
    device = "cpu"
    torch_dtype = torch.float32  # safest CPU default
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=None,
    )
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )

# Global cache (simple)
_PIPE_CACHE: dict[str, any] = {}

def get_pipe(model_id: str):
    if model_id not in _PIPE_CACHE:
        _PIPE_CACHE[model_id] = load_pipe(model_id)
    return _PIPE_CACHE[model_id]

# ---------------------------
# Prompt builder
# ---------------------------
SYSTEM_RAILS = (
    "You are a helpful, empathetic, and safe customer support agent. "
    "Never provide legal/medical/financial advice. Never ask for full card numbers, passwords, or one-time codes. "
    "Keep answers clear and concise. Use bullet points for steps. If you are not sure, say so and propose a next action."
)

def build_prompt(
    message: str,
    style_preset: str,
    brand_voice: str,
    language: str,
    severity: int,
    include_guardrails: bool,
    template_text: str | None,
) -> str:
    style_cfg = STYLE_PRESETS.get(style_preset, STYLE_PRESETS["E-commerce (friendly & helpful)"])
    tone = style_cfg["tone"]
    extras = style_cfg["extras"]

    severity_text = {
        1: "minor inconvenience; keep tone light but respectful.",
        2: "moderate issue; show empathy and solution-oriented tone.",
        3: "high-impact issue; show strong empathy; prioritize immediate steps and escalation path.",
        4: "urgent issue; maintain calm, reassure and explain immediate next steps; propose escalation.",
        5: "critical issue; highest empathy; prioritize safety and fast escalation with clear next steps.",
    }[max(1, min(5, severity))]

    rails = SYSTEM_RAILS if include_guardrails else ""
    brand = f"Brand voice guidelines: {brand_voice}." if brand_voice.strip() else ""

    template_section = f"\nAgent template to incorporate:\n{template_text}\n" if (template_text and template_text.strip()) else ""

    prompt = (
        f"{rails}\n\n"
        f"Style: {tone}\n"
        f"Additional style guidance: {extras}\n"
        f"Issue severity: {severity} ⇒ {severity_text}\n"
        f"{brand}\n"
        f"Respond in: {language}\n"
        f"{template_section}"
        f"\nCustomer: {message}\n"
        "Agent:"
    )
    return prompt

# ---------------------------
# Inference
# ---------------------------
def infer(
    model_id: str,
    message: str,
    style_preset: str,
    brand_voice: str,
    language: str,
    severity: int,
    include_guardrails: bool,
    template_key: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    if not message.strip():
        return "Please enter a customer message."

    template_text = TEMPLATE_SNIPPETS.get(template_key, "")
    prompt = build_prompt(
        message=message,
        style_preset=style_preset,
        brand_voice=brand_voice,
        language=language,
        severity=severity,
        include_guardrails=include_guardrails,
        template_text=template_text,
    )

    pipe = get_pipe(model_id)
    out = pipe(
        prompt,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
        temperature=float(temperature),
        top_p=float(top_p),
        eos_token_id=pipe.tokenizer.eos_token_id,
        pad_token_id=pipe.tokenizer.eos_token_id,
    )[0]["generated_text"]

    # Return only the agent continuation (after the last "Agent:")
    if "Agent:" in out:
        out = out.split("Agent:")[-1].strip()
    return out

# ---------------------------
# Gradio UI
# ---------------------------
def ui():
    with gr.Blocks(title="Customer Support Reply Generator") as demo:
        gr.Markdown(
            "## Customer Support Reply Generator "
            "*(Styled + Templates • CPU-friendly • Model Switch)*"
        )

        with gr.Row():
            model_id = gr.Dropdown(
                label="Model (CPU-friendly)",
                choices=list(MODELS.keys()),
                value=DEFAULT_MODEL,
                info="You can switch between small, CPU-friendly models.",
            )

        message = gr.Textbox(
            label="Customer message",
            placeholder="e.g., I can’t log in to my account. Can you help?",
            lines=6,
        )

        with gr.Accordion("One-click templates (optional)", open=False):
            template_key = gr.Dropdown(
                label="Insert a quick template outline",
                choices=["(none)"] + list(TEMPLATE_SNIPPETS.keys()),
                value="(none)",
            )

        with gr.Row():
            style_preset = gr.Dropdown(
                label="Style preset",
                choices=list(STYLE_PRESETS.keys()),
                value="Banking / Fintech (formal & compliant)",
            )
            brand_voice = gr.Textbox(
                label="Brand voice (optional)",
                placeholder="e.g., formal, friendly, concise, premium…",
            )
            language = gr.Dropdown(
                label="Response language",
                choices=LANGUAGES,
                value="English",
            )

        with gr.Row():
            severity = gr.Slider(
                label="Issue severity (adds more empathy at higher values)",
                minimum=1, maximum=5, step=1, value=2
            )
            include_guardrails = gr.Checkbox(
                label="Include policy guardrails (recommended)",
                value=True
            )

        with gr.Row():
            max_new_tokens = gr.Slider(label="Max new tokens", minimum=32, maximum=512, step=1, value=160)
            temperature = gr.Slider(label="Temperature", minimum=0.1, maximum=1.5, step=0.05, value=0.6)
            top_p = gr.Slider(label="Top-p", minimum=0.1, maximum=1.0, step=0.05, value=0.9)

        generate = gr.Button("Generate reply", variant="primary")
        output = gr.Markdown(label="Suggested agent reply")

        generate.click(
            infer,
            inputs=[
                model_id,
                message,
                style_preset,
                brand_voice,
                language,
                severity,
                include_guardrails,
                template_key,
                max_new_tokens,
                temperature,
                top_p,
            ],
            outputs=[output],
        )

        gr.Markdown(
            "—\n"
            "**Tip:** Qwen2.5-0.5B-Instruct is snappy on CPU. TinyLlama-1.1B gives a bit more headroom, but may be slower.\n"
        )
    return demo


if __name__ == "__main__":
    ui().queue().launch()
