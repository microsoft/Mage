"""Gradio app for MageFlow — text-to-image and instruction-based image editing.

    python app.py                       # serve on 0.0.0.0:7860
    python app.py --share --port 7861

Each tab has a model preset dropdown (base / rl / turbo) plus a free-form
"Custom model" box for any Hugging Face repo id or local path. Models load
lazily on first use and are cached. Notes:
  - By default the presets point at the `microsoft/Mage-Flow*` Hugging Face
    repos (downloaded + cached on first use). Set ``MAGEFLOW_HF_DIR`` to load
    local checkpoint dirs instead.
  - Turbo checkpoints are few-step: use steps=4, cfg=1.
"""
from __future__ import annotations

import argparse
import os

import gradio as gr
from PIL import Image

from mage_flow.pipeline import MageFlowPipeline

# Default to Hugging Face repo ids; if MAGEFLOW_HF_DIR is set, use local
# checkpoint dirs under it instead (local dir names match the HF repo basename).
HF_DIR = os.environ.get("MAGEFLOW_HF_DIR")


def _repo(hf_id: str, local_name: str) -> str:
    return f"{HF_DIR}/{local_name}" if HF_DIR else hf_id


T2I_MODELS = {
    "base":     _repo("microsoft/Mage-Flow-Base",     "Mage-Flow-Base"),
    "rl":       _repo("microsoft/Mage-Flow",          "Mage-Flow"),
    "turbo":    _repo("microsoft/Mage-Flow-Turbo",    "Mage-Flow-Turbo"),
    "sciforma": _repo("microsoft/Mage-Flow-SciForma", "Mage-Flow-SciForma"),
}
EDIT_MODELS = {
    "base":  _repo("microsoft/Mage-Flow-Edit-Base",  "Mage-Flow-Edit-Base"),
    "rl":    _repo("microsoft/Mage-Flow-Edit",       "Mage-Flow-Edit"),
    "turbo": _repo("microsoft/Mage-Flow-Edit-Turbo", "Mage-Flow-Edit-Turbo"),
}

DEVICE = "cuda"
TEXT_DEVICE: str | None = None
_CACHE: dict[str, MageFlowPipeline] = {}


def _get_pipe(repo: str) -> MageFlowPipeline:
    """Load (and cache) a pipeline from a local dir OR a Hugging Face repo id.

    ``MageFlowPipeline.from_pretrained`` resolves a repo id via
    ``snapshot_download`` automatically, so both are accepted here.
    """
    repo = (repo or "").strip()
    if not repo:
        raise gr.Error("No model specified.")
    if repo not in _CACHE:
        try:
            _CACHE[repo] = MageFlowPipeline.from_pretrained(
                repo, device=DEVICE, text_device=TEXT_DEVICE)
        except Exception as e:  # noqa: BLE001
            raise gr.Error(f"Failed to load model '{repo}': {type(e).__name__}: {e}")
    return _CACHE[repo]


def _resolve(preset_map, model_key, custom_model):
    """Custom repo id / path (if given) overrides the preset dropdown."""
    return (custom_model or "").strip() or preset_map[model_key]


def run_t2i(model_key, custom_model, prompt, neg_prompt, steps, cfg, height, width, seed,
            progress=gr.Progress(track_tqdm=False)):
    if not (prompt or "").strip():
        raise gr.Error("Prompt is empty.")
    repo = _resolve(T2I_MODELS, model_key, custom_model)
    progress(0.1, desc=f"loading {repo} …")
    pipe = _get_pipe(repo)
    progress(0.4, desc="generating …")
    img = pipe.generate(
        [prompt], neg_prompts=[neg_prompt or " "], seeds=[int(seed)],
        steps=int(steps), cfg=float(cfg),
        heights=[int(height)], widths=[int(width)],
    )[0]
    return img


def run_edit(model_key, custom_model, prompt, neg_prompt, ref_img, extra_files, steps, cfg, max_size, seed,
             progress=gr.Progress(track_tqdm=False)):
    if not (prompt or "").strip():
        raise gr.Error("Edit instruction is empty.")
    refs = []
    if ref_img is not None:
        refs.append(ref_img if isinstance(ref_img, Image.Image) else Image.open(ref_img))
    for f in (extra_files or []):
        refs.append(Image.open(f).convert("RGB"))
    if not refs:
        raise gr.Error("Upload at least one reference image.")
    refs = [r.convert("RGB") for r in refs]
    repo = _resolve(EDIT_MODELS, model_key, custom_model)
    progress(0.1, desc=f"loading {repo} …")
    pipe = _get_pipe(repo)
    progress(0.4, desc="editing …")
    out = pipe.edit(
        [prompt], [refs], neg_prompts=[neg_prompt or " "], seeds=[int(seed)],
        steps=int(steps), cfg=float(cfg),
        max_size=int(max_size) if max_size else None,
    )[0]
    return out


_NOTE = (
    "Pick a **preset** (base / rl / turbo) or type a **custom model** — any "
    "Hugging Face repo id (e.g. `microsoft/Mage-Flow-Turbo`) or local path; it "
    "is downloaded and cached on first use. **Turbo** models are few-step: set "
    "**steps=4, cfg=1**."
)

_CUSTOM_PH_T2I = "microsoft/Mage-Flow  (repo id or local path — overrides preset)"
_CUSTOM_PH_EDIT = "microsoft/Mage-Flow-Edit  (repo id or local path — overrides preset)"


def build_ui():
    with gr.Blocks(title="MageFlow") as demo:
        gr.Markdown("# MageFlow\nText-to-image generation and instruction-based image editing.")
        gr.Markdown(_NOTE)

        with gr.Tab("Text → Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    t_model = gr.Dropdown(list(T2I_MODELS), value="base", label="Model preset")
                    t_custom = gr.Textbox(label="Custom model (optional)", placeholder=_CUSTOM_PH_T2I, lines=1)
                    t_prompt = gr.Textbox(label="Prompt", lines=3,
                                          value="A close-up portrait of an elderly African man with deep wrinkles, wearing a traditional hat, soft natural lighting, ultra realistic.")
                    t_neg = gr.Textbox(label="Negative prompt", value=" ", lines=1)
                    with gr.Row():
                        t_steps = gr.Slider(1, 50, value=30, step=1, label="Steps")
                        t_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="CFG")
                    with gr.Row():
                        t_h = gr.Slider(256, 2048, value=1024, step=16, label="Height")
                        t_w = gr.Slider(256, 2048, value=1024, step=16, label="Width")
                    t_seed = gr.Number(value=42, precision=0, label="Seed")
                    t_btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=1):
                    t_out = gr.Image(type="pil", label="Output", height=560)
            # Clear the previous output first so the stale image isn't shown as
            # the result while the new one is still transferring (esp. over a
            # gradio share tunnel, where the image download can lag a few seconds).
            t_btn.click(lambda: None, None, t_out).then(
                        run_t2i,
                        [t_model, t_custom, t_prompt, t_neg, t_steps, t_cfg, t_h, t_w, t_seed],
                        t_out)

        with gr.Tab("Image Edit"):
            with gr.Row():
                with gr.Column(scale=1):
                    e_model = gr.Dropdown(list(EDIT_MODELS), value="base", label="Model preset")
                    e_custom = gr.Textbox(label="Custom model (optional)", placeholder=_CUSTOM_PH_EDIT, lines=1)
                    e_prompt = gr.Textbox(label="Edit instruction", lines=2,
                                          value="change the background to a city street")
                    e_neg = gr.Textbox(label="Negative prompt", value=" ", lines=1)
                    e_ref = gr.Image(type="pil", label="Reference image", height=280,
                                     value=os.path.join(os.path.dirname(__file__), "assets", "dog.jpg"))
                    e_extra = gr.File(file_count="multiple", type="filepath",
                                      label="Extra references (optional, multi-image edit)")
                    with gr.Row():
                        e_steps = gr.Slider(1, 50, value=30, step=1, label="Steps")
                        e_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="CFG")
                    e_max = gr.Slider(0, 2048, value=1024, step=16,
                                      label="Max output side (0 = keep source size)")
                    e_seed = gr.Number(value=42, precision=0, label="Seed")
                    e_btn = gr.Button("Edit", variant="primary")
                with gr.Column(scale=1):
                    e_out = gr.Image(type="pil", label="Output", height=560)
            e_btn.click(lambda: None, None, e_out).then(
                        run_edit,
                        [e_model, e_custom, e_prompt, e_neg, e_ref, e_extra, e_steps, e_cfg, e_max, e_seed],
                        e_out)
    return demo


def main():
    global DEVICE, TEXT_DEVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--text-device", default=None,
                    help="device for the text encoder (default: same as --device; "
                         "e.g. --device cuda:0 --text-device cuda:1 splits across two GPUs)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--preload", default=None,
                    help="comma-separated repo ids / paths to load at startup (else lazy)")
    args = ap.parse_args()
    DEVICE = args.device
    TEXT_DEVICE = args.text_device
    if args.preload:
        for repo in args.preload.split(","):
            _get_pipe(repo.strip())
    build_ui().queue().launch(server_name=args.host, server_port=args.port,
                              share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
