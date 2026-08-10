#!/usr/bin/env python
"""MageFlow command-line inference.

Two console scripts share this module:

  * ``mage-flow``      — text-to-image generation
  * ``mage-flow-edit`` — instruction-based image editing

Both are BATCHED: pass several prompts and they are packed into a single
transformer forward per denoise step. Sample ``i`` uses seed ``--seed + i``.

Text-to-image (multiple prompts = a batch)::

    mage-flow --prompt "a cat holding a sign that says hello" "a red ferrari" \
        --model_path microsoft/Mage-Flow-4B --steps 30 --cfg 5.0 --out ./outputs

``--model_path`` accepts a local repo dir OR a Hugging Face Hub repo id (e.g.
``microsoft/Mage-Flow-4B``), downloaded and cached automatically on first use.

Mixed resolutions — give one ``--height``/``--width`` per prompt (they are packed
into a single forward per step regardless of shape)::

    mage-flow --prompt "a tall waterfall" "a wide desert panorama" \
        --height 2048 512 --width 512 2048 \
        --model_path microsoft/Mage-Flow-4B --out ./outputs

Image editing (one ``--ref`` entry per prompt; comma-separate paths for
multi-image edit)::

    mage-flow-edit \
        --prompt "把背景改为城市街道" "把这两张图融合在一起" \
        --ref hydrant.png "scene.png,object.png" \
        --model_path /path/Mage-Flow-Edit-4B-Base --out ./outputs
"""
import argparse
import os

from mage_flow import MageFlowPipeline


def _add_common_args(p):
    p.add_argument("--model_path", required=True,
                   help="local diffusers-style repo dir OR a Hugging Face Hub repo id "
                        "(e.g. microsoft/Mage-Flow-4B); HF ids are downloaded and cached "
                        "automatically on first use")
    p.add_argument("--neg_prompt", default=None,
                   help="negative prompt applied to every sample (default: a single space)")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--cfg", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42, help="base seed; sample i uses seed + i")
    p.add_argument("--static_shift", type=float, default=None,
                   help="override scheduler shift (default: repo scheduler_config.json, 6.0)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--text-device", default=None,
                   help="device for the text encoder (default: same as --device; "
                        "e.g. --device cuda:0 --text-device cuda:1 splits across two GPUs)")
    p.add_argument("--out", default="./outputs")


def _neg_list(neg_prompt, n):
    return [neg_prompt] * n if neg_prompt is not None else None


def _size_list(vals, n, name, parser):
    """Broadcast one size value to all prompts, or use a per-prompt list."""
    if len(vals) == 1:
        return vals * n
    if len(vals) == n:
        return vals
    parser.error(f"--{name} expects 1 value (applied to all) or {n} values "
                 f"(one per prompt); got {len(vals)}")


def main():
    """``mage-flow`` — batched text-to-image generation."""
    p = argparse.ArgumentParser(
        prog="mage-flow", description="MageFlow text-to-image generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", nargs="+", required=True,
                   help="one or more prompts; multiple prompts are batched")
    p.add_argument("--height", type=int, nargs="+", default=[1024],
                   help="output height (multiple of 16); one value applied to all "
                        "prompts, or one value per prompt for mixed resolutions")
    p.add_argument("--width", type=int, nargs="+", default=[1024],
                   help="output width (multiple of 16); one value applied to all "
                        "prompts, or one value per prompt for mixed resolutions")
    p.add_argument("--prompt_template", default="mage-flow")
    _add_common_args(p)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    pipe = MageFlowPipeline.from_pretrained(args.model_path, args.device, args.text_device)
    n = len(args.prompt)
    imgs = pipe.generate(
        args.prompt,
        neg_prompts=_neg_list(args.neg_prompt, n),
        seeds=[args.seed + i for i in range(n)],
        heights=_size_list(args.height, n, "height", p),
        widths=_size_list(args.width, n, "width", p),
        steps=args.steps, cfg=args.cfg, static_shift=args.static_shift,
        prompt_template=args.prompt_template,
    )
    for i, im in enumerate(imgs):
        path = os.path.join(args.out, f"gen_{i:03d}.png")
        im.save(path)
        print(f"saved {path}")


def main_edit():
    """``mage-flow-edit`` — batched instruction-based image editing."""
    p = argparse.ArgumentParser(
        prog="mage-flow-edit", description="MageFlow instruction-based image editing.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", nargs="+", required=True,
                   help="one or more edit instructions; multiple prompts are batched")
    p.add_argument("--ref", nargs="+", required=True,
                   help="reference image(s) for each prompt, positionally aligned with "
                        "--prompt; comma-separate paths for multi-image edit "
                        "(e.g. --ref a.png 'b.png,c.png')")
    p.add_argument("--max_size", type=int, default=None,
                   help="longest side of the output (short side by aspect ratio). "
                        "Default: keep each source image's own resolution")
    p.add_argument("--height", type=int, default=None,
                   help="explicit output height (use with --width; overrides --max_size)")
    p.add_argument("--width", type=int, default=None,
                   help="explicit output width (use with --height; overrides --max_size)")
    p.add_argument("--vl_cond_long_edge", type=int, default=384,
                   help="cap the long edge of the reference image fed to the VL text "
                        "encoder (matches training preprocessing; the VAE path keeps the "
                        "full output resolution). 0 or negative disables the cap")
    p.add_argument("--prompt_template", default="mage-flow-edit")
    _add_common_args(p)
    args = p.parse_args()

    if len(args.ref) != len(args.prompt):
        p.error(f"--ref count ({len(args.ref)}) must match --prompt count ({len(args.prompt)})")

    os.makedirs(args.out, exist_ok=True)
    pipe = MageFlowPipeline.from_pretrained(args.model_path, args.device, args.text_device)
    n = len(args.prompt)
    # Each --ref token is one prompt's reference(s); commas split multi-image refs.
    ref_images = [[s.strip() for s in r.split(",") if s.strip()] for r in args.ref]

    size_kw = {}
    if args.height and args.width:
        size_kw = {"heights": [args.height] * n, "widths": [args.width] * n}

    outs = pipe.edit(
        args.prompt, ref_images,
        neg_prompts=_neg_list(args.neg_prompt, n),
        seeds=[args.seed + i for i in range(n)],
        max_size=args.max_size, steps=args.steps, cfg=args.cfg,
        static_shift=args.static_shift, prompt_template=args.prompt_template,
        vl_cond_long_edge=args.vl_cond_long_edge,
        **size_kw,
    )
    for i, im in enumerate(outs):
        path = os.path.join(args.out, f"edit_{i:03d}.png")
        im.save(path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
