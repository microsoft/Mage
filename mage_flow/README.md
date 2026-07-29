<h1 align="center">Mage-Flow<br><span style="font-size: 0.55em; font-weight: normal;">An Efficient Native-Resolution Foundation Model for Image Generation and Editing</span></h1>

<p align="center">
  <a href="https://arxiv.org/abs/2607.19064"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-Mage--Flow-b31b1b" height="22" /></a>
  <a href="https://microsoft.github.io/Mage"><img alt="Project Page" src="https://img.shields.io/badge/%F0%9F%8C%90-Project%20Page-blue" height="22" /></a>
  <a href="https://github.com/microsoft/Mage"><img src="https://img.shields.io/badge/Code-GitHub-181717?logo=github" alt="GitHub"></a>
  <a href="https://huggingface.co/microsoft/Mage-Flow-Base"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Mage--Flow--Base-yellow" height="22" /></a>
  <a href="https://huggingface.co/microsoft/Mage-Flow"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Mage--Flow-yellow" height="22" /></a>
  <a href="https://huggingface.co/microsoft/Mage-Flow-Turbo"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Mage--Flow--Turbo-yellow" height="22" /></a>
  <a href="https://huggingface.co/microsoft/Mage-Flow-Edit-Base"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Mage--Flow--Edit--Base-yellow" height="22" /></a>
  <a href="https://huggingface.co/microsoft/Mage-Flow-Edit"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Mage--Flow--Edit-yellow" height="22" /></a>
  <a href="https://huggingface.co/microsoft/Mage-Flow-Edit-Turbo"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Mage--Flow--Edit--Turbo-yellow" height="22" /></a>
  <a href="https://github.com/microsoft/Mage/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT"></a>
</p>

<div align="center">
<img src="assets/mage-flow-cover.png" width="100%" alt="gallery">
</div>

---

**Mage-Flow** is a compact **4B-scale generative stack** for efficient **text-to-image generation** and **instruction-based image editing**. Instead of scaling to tens of billions of parameters, Mage-Flow reaches state-of-the-art-competitive quality through careful **tokenizer–backbone–system co-design**, so it stays fast, memory-light, and easy to fine-tune under realistic compute budgets.

The stack is built from **two shared, co-designed components**:

- **Mage-VAE** — a lightweight, high-fidelity latent tokenizer (one-step diffusion encode/decode with anchor-latent KL regularization).
- **NR-MMDiT** — a shared 4B **Native-Resolution Multimodal Diffusion Transformer**, trained with rectified flow matching in the Mage-VAE latent space.

Together with native-resolution packing and a fused-kernel training infrastructure, this shared stack powers **two model instantiations**: **Mage-Flow** for text-to-image generation and **Mage-Flow-Edit** for instruction-based image editing. Each ships in **Base**, **RL-aligned**, and **4-step Turbo** variants.

## ✨ Highlights

- **Compact & competitive.** A single 4B family for generation *and* editing that matches or beats much larger open systems (Qwen-Image 20B, Z-Image 6B, FLUX.2 32B, FireRed-Image-Edit 20B).
- **Efficient tokenizer.** Mage-VAE matches FLUX.2-VAE reconstruction fidelity while using **~12× / ~22× fewer encode / decode MACs per pixel**, removing the VAE as the high-resolution bottleneck.
- **Native resolution.** One checkpoint generates from **512 to 2048** on any aspect ratio, including extreme **4:1** (e.g. `512×2048`, `2048×512`).
- **System-level speed.** Native-resolution packing (FlashAttention var-len + per-sample 2D RoPE) + fused CUDA kernels cut per-step training time from **~1.93 s → ~0.78 s** (**~2.5× faster training**); CFG's conditional/unconditional branches run in **one** packed forward.
- **Full family.** **Base**, **RL-aligned**, and **4-step Turbo** variants for both generation and editing.
- **Versatile editing.** Mage-Flow-Edit supports semantic content editing, appearance transformation, image restoration, and structure-aware outputs within a unified image-and-text-conditioned model. See the report's editing galleries.
- **Interactive latency.** At `1024²` on a single A100: **Mage-Flow-Turbo 0.59 s/image**, **Mage-Flow-Edit-Turbo 1.02 s/edit**, peak memory **~18–20 GB** (lowest among compared systems).

<div align="center">
<img src="assets/one_to_many_editing_diversity.jpg" width="100%" alt="One-to-many editing diversity"><br>
<em>One-to-many editing diversity — Mage-Flow-Edit can generate diverse outputs from a single reference image.</em>
</div>

## 📥 Model Zoo

Each checkpoint is a self-contained diffusers-style repo (`transformer/` + shared `vae/`, `text_encoder/`, `scheduler/`).

| Model                       | Task        | Variant            | Steps | Hugging Face                                                                              |
| :-------------------------- | :---------- | :----------------- | :---: | :---------------------------------------------------------------------------------------- |
| `Mage-Flow-4B-Base`       | text→image | Base               |  30  | [🤗 microsoft/Mage-Flow-Base](https://huggingface.co/microsoft/Mage-Flow-Base)             |
| `Mage-Flow-4B`            | text→image | RL-aligned         |  20  | [🤗 microsoft/Mage-Flow](https://huggingface.co/microsoft/Mage-Flow)                       |
| `Mage-Flow-4B-Turbo`      | text→image | Few-step distilled |   4   | [🤗 microsoft/Mage-Flow-Turbo](https://huggingface.co/microsoft/Mage-Flow-Turbo)           |
| `Mage-Flow-Edit-4B-Base`  | editing     | Base               |  30  | [🤗 microsoft/Mage-Flow-Edit-Base](https://huggingface.co/microsoft/Mage-Flow-Edit-Base)   |
| `Mage-Flow-Edit-4B`       | editing     | RL-aligned         |  30  | [🤗 microsoft/Mage-Flow-Edit](https://huggingface.co/microsoft/Mage-Flow-Edit)             |
| `Mage-Flow-Edit-4B-Turbo` | editing     | Few-step distilled |   4   | [🤗 microsoft/Mage-Flow-Edit-Turbo](https://huggingface.co/microsoft/Mage-Flow-Edit-Turbo) |

## 🖼️ Showcase

**Text-to-image** — prompt following, fine detail, and legible English/Chinese text rendering. *(The first panel is open; click a title to expand the others.)*

<details open>
<summary><b>Showcase</b></summary>
<div align="center"><img src="assets/t2i_teaser.jpg" width="100%" alt="t2i teaser"></div>
</details>

<details>
<summary><b>General scenes</b></summary>
<div align="center"><img src="assets/general.jpg" width="100%" alt="general scenes"></div>
</details>

<details>
<summary><b>Portraits</b></summary>
<div align="center"><img src="assets/portrait.jpg" width="100%" alt="portraits"></div>
</details>

<details>
<summary><b>Cuisine & still life</b></summary>
<div align="center"><img src="assets/cuisine.jpg" width="100%" alt="cuisine"></div>
</details>

<details>
<summary><b>English text rendering</b></summary>
<div align="center"><img src="assets/text_en.jpg" width="100%" alt="english text"></div>
</details>

<details>
<summary><b>Chinese text rendering</b></summary>
<div align="center"><img src="assets/text_zh.jpg" width="100%" alt="chinese text"></div>
</details>

**Instruction-based editing** — appearance, content, scene/subject, human-centered & creative, low-level, and restoration edits (source → result). *(The first panel is open; click a title to expand the others.)*

<details open>
<summary><b>Various Editing I</b></summary>
<div align="center"><img src="assets/edit_gallery_showcase_1.jpg" width="100%" alt="editing showcase 1"></div>
</details>

<details>
<summary><b>Various Editing II</b></summary>
<div align="center"><img src="assets/edit_gallery_showcase_2.jpg" width="100%" alt="editing showcase 2"></div>
</details>

<details>
<summary><b>Localized content & object editing</b></summary>
<div align="center"><img src="assets/edit_gallery_content.jpg" width="100%" alt="content editing"></div>
</details>

<details>
<summary><b>Scene, subject & camera transformations</b></summary>
<div align="center"><img src="assets/edit_gallery_scene_subject.jpg" width="100%" alt="scene and subject"></div>
</details>

<details>
<summary><b>Appearance & artistic rendering</b></summary>
<div align="center"><img src="assets/edit_gallery_appearance.jpg" width="100%" alt="appearance"></div>
</details>

<details>
<summary><b>Human-centered & creative editing</b></summary>
<div align="center"><img src="assets/edit_gallery_human_creative.jpg" width="100%" alt="human-centered and creative"></div>
</details>

<details>
<summary><b>Low-level vision & conditional reconstruction</b></summary>
<div align="center"><img src="assets/edit_gallery_lowlevel.jpg" width="100%" alt="low-level vision"></div>
</details>

<details>
<summary><b>Bidirectional degradation & restoration</b></summary>
<div align="center"><img src="assets/edit_gallery_restoration.jpg" width="100%" alt="restoration"></div>
</details>

## 📊 Performance

<details>
<summary><b>Full benchmark tables (text-to-image & image editing) — click to expand</b></summary>

**Text-to-image** — full benchmark suite: GenEval, DPG-Bench, TIIF-Bench (short/long splits), CVTG-2K, OneIG (EN/CN), LongText (EN/CN). Higher is better; GenEval / CVTG-2K / OneIG / LongText on a 0–1 scale, DPG / TIIF on 0–100. The **Type** column marks closed- vs open-source; **bold** / <ins>underline</ins> = best / second-best among **open-source** models (closed-source shown for reference, not ranked); `–` = not reported; ★ = ours.

| Model                        |  Type  | #Params | Steps |     GenEval     |       DPG       |    TIIF-Short    |    TIIF-Long    |     CVTG-2K     |     OneIG-EN     |     OneIG-CN     |   LongText-EN   |   LongText-CN   |
| :--------------------------- | :----: | :-----: | :---: | :-------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: | :--------------: |
| Seedream 3.0                 | Closed |   –   |  –  |      0.84      |      88.27      |      86.02      |      84.31      |      0.592      |      0.530      |      0.528      |      0.896      |      0.878      |
| Seedream 4.0                 | Closed |   –   |  –  |      0.84      |      88.63      |        –        |        –        |      0.892      |      0.573      |      0.554      |      0.936      |      0.946      |
| GPT-Image-1                  | Closed |   –   |  –  |      0.84      |      85.15      |      89.15      |      88.29      |      0.857      |      0.533      |      0.474      |      0.956      |      0.619      |
| Nano-Banana-Pro              | Closed |   –   |  –  |      0.83      |      87.16      |        –        |        –        |      0.779      |      0.580      |      0.570      |      0.981      |      0.949      |
| FLUX.1-dev                   |  Open  |   12B   |  50  |      0.66      |      83.84      |      71.09      |      71.78      |      0.496      |      0.434      |      0.245      |      0.607      |      0.005      |
| FLUX.1-Krea-dev              |  Open  |   12B   |  50  |      0.72      |      86.59      |      80.36      |      81.67      |      0.444      |      0.443      |      0.271      |      0.693      |      0.002      |
| FLUX.2-dev                   |  Open  |   32B   |  50  |      0.87      |      87.57      | **88.82** | **88.10** | **0.893** | **0.551** |      0.516      | **0.963** |      0.757      |
| FLUX.2-Klein-Base-4B         |  Open  |   4B   |  50  |      0.78      |      83.02      |      79.94      |      80.01      |      0.656      |      0.485      |      0.366      |      0.554      |      0.071      |
| FLUX.2-Klein-Base-9B         |  Open  |   9B   |  50  |      0.83      |      85.29      |      81.47      |      84.52      |      0.655      |      0.544      |      0.400      |      0.872      |      0.227      |
| FLUX.2-Klein-4B              |  Open  |   4B   |   4   |      0.83      |      85.53      |      78.91      |      79.04      |      0.628      |      0.500      |      0.364      |      0.649      |      0.068      |
| FLUX.2-Klein-9B              |  Open  |   9B   |   4   |      0.86      |      86.20      |      85.22      |      84.13      |      0.424      |      0.538      |      0.406      |      0.872      |      0.226      |
| Qwen-Image                   |  Open  |   20B   |  50  |      0.87      | **88.32** | <ins>86.14</ins> | <ins>86.83</ins> |      0.829      |      0.539      | **0.548** |      0.943      |      0.946      |
| JoyAI-Image                  |  Open  |   16B   |  50  |       –       |      88.05      |        –        |        –        |      0.874      |      0.542      |      0.521      | **0.963** | **0.963** |
| HunyuanImage-3.0             |  Open  |   80B   |  50  |      0.72      |      86.10      |        –        |        –        |      0.765      |        –        |        –        |        –        |        –        |
| LongCat-Image                |  Open  |   6B   |  50  |      0.87      |      86.80      |      80.93      |      81.30      |      0.866      |      0.516      |      0.518      |      0.885      | <ins>0.956</ins> |
| Z-Image-Base                 |  Open  |   6B   |  50  |      0.84      | <ins>88.14</ins> |      80.20      |      83.04      |      0.867      | <ins>0.546</ins> | <ins>0.535</ins> |      0.935      |      0.936      |
| Z-Image-Turbo                |  Open  |   6B   |   8   |      0.82      |      84.86      |      77.73      |      80.05      |      0.859      |      0.528      |      0.507      |      0.917      |      0.926      |
| **Mage-Flow-Base** ★  |  Open  |   4B   |  30  |      0.79      |      86.26      |      82.50      |      83.19      |      0.851      |      0.542      |      0.509      |      0.904      |      0.792      |
| **Mage-Flow** ★       |  Open  |   4B   |  20  | **0.90** |      86.49      |      82.19      |      84.70      | <ins>0.887</ins> |      0.536      |      0.505      | <ins>0.944</ins> |      0.823      |
| **Mage-Flow-Turbo** ★ |  Open  |   4B   |   4   | <ins>0.88</ins> |      85.48      |      83.58      |      84.16      |      0.873      |      0.523      |      0.491      |      0.911      |      0.801      |

**Image editing** — ImgEdit-Bench (0–5), GEdit-Bench EN/CN (0–10), TextEdit-Bench synthetic/real (0–25). Higher is better; the **Type** column marks closed- vs open-source; **bold** / <ins>underline</ins> = best / second-best among **open-source** models; `–` = not reported; ★ = ours.

| Model                             |  Type  | #Params | Steps |     ImgEdit     |     GEdit-EN     |     GEdit-CN     |   TextEdit-Syn   |  TextEdit-Real  |
| :-------------------------------- | :----: | :-----: | :---: | :-------------: | :--------------: | :--------------: | :--------------: | :--------------: |
| Nano-Banana                       | Closed |   –   |  –  |      4.29      |      7.291      |      7.399      |      16.54      |      18.22      |
| Seedream 4.0                      | Closed |   –   |  –  |      4.30      |      7.701      |      7.692      |      14.90      |      18.54      |
| Seedream 4.5                      | Closed |   –   |  –  |      4.32      |      7.820      |      7.800      |        –        |        –        |
| Nano-Banana-Pro                   | Closed |   –   |  –  |      4.37      |      7.738      |      7.799      |        –        |        –        |
| Step1X-Edit-v1.2                  |  Open  |   19B   |  50  |      3.95      |      7.480      |      7.467      |       9.26       |      12.02      |
| FLUX.1-Kontext-dev                |  Open  |   12B   |  28  |      3.71      |      6.462      |      1.857      |      12.14      |      14.31      |
| FLUX.2-dev                        |  Open  |   32B   |  50  |      4.35      |      7.413      |      7.278      |      11.86      |      14.71      |
| FLUX.2-Klein-Base-4B              |  Open  |   4B   |  50  |      3.80      |      7.081      |      7.102      |      11.01      |      13.79      |
| FLUX.2-Klein-4B                   |  Open  |   4B   |   4   |      4.01      |      7.717      |      7.750      |      11.84      |      14.46      |
| FLUX.2-Klein-Base-9B              |  Open  |   9B   |  50  |      4.05      |      7.740      |      7.745      |      12.76      |      15.65      |
| FLUX.2-Klein-9B                   |  Open  |   9B   |   4   |      4.18      |      8.040      |      8.055      |      12.73      |      15.75      |
| Z-Image-Edit                      |  Open  |   6B   |  50  |      4.30      |      7.570      |      7.540      |        –        |        –        |
| Qwen-Image-Edit-2509              |  Open  |   20B   |  50  |      4.31      |      7.480      |      7.467      |      13.40      |      15.81      |
| Qwen-Image-Edit-2511              |  Open  |   20B   |  50  | <ins>4.51</ins> |      7.877      |      7.819      |      13.53      | <ins>16.81</ins> |
| LongCat-Image-Edit                |  Open  |   6B   |  50  |      4.45      |      7.748      |      7.731      |      12.46      |      14.89      |
| FireRed-Image-Edit-1.0            |  Open  |   20B   |  50  | **4.56** |      7.943      |      7.887      | **15.19** | **17.23** |
| JoyAI-Image-Edit                  |  Open  |   16B   |  50  |      4.46      | **8.276** | <ins>8.125</ins> | <ins>14.80</ins> | **17.23** |
| **Mage-Flow-Edit-Base** ★  |  Open  |   4B   |  30  |      4.28      |      7.860      |      7.970      |      13.63      |      15.57      |
| **Mage-Flow-Edit** ★       |  Open  |   4B   |  30  |      4.34      |      8.127      |      8.123      |      14.14      |      16.26      |
| **Mage-Flow-Edit-Turbo** ★ |  Open  |   4B   |   4   |      4.38      | <ins>8.271</ins> | **8.264** |      12.77      |      15.41      |

</details>

## 🏗️ Architecture

**Mage-VAE** — a latent tokenizer built as a *symmetric* one-step diffusion codec: the decoder is a fully-convolutional one-step pixel-diffusion model (no global-attention blocks), and the encoder is its architectural dual (a one-step latent generator conditioned on pixels). A standard Gaussian-prior KL is replaced with an **anchor-latent KL** that regularizes the posterior toward FLUX.2-VAE latents, giving a generation-ready `128`-channel, `16×`-downsampled latent space.

<div align="center">
<img src="assets/mage_vae.jpg" width="100%" alt="Mage-VAE architecture and training"><br>
<em>Mage-VAE — anchor VAE (FLUX.2-VAE), the symmetric one-step encoder/decoder architecture, and the three-stage training pipeline.</em>
</div>

**Mage-Flow** — a 4B Multimodal DiT that encodes prompts with **Qwen3-VL** and images with Mage-VAE, then processes **packed** variable-length image+text sequences with per-sample 2D rotary embeddings and joint self-attention. **Native-resolution packing** removes bucket quantization and padding, lets one checkpoint generalize to any output size, and fuses the CFG cond/uncond branches into a single forward.

<div align="center">
<img src="assets/nr_mmdit.jpg" width="100%" alt="Mage-Flow / Native-Resolution MMDiT architecture"><br>
<em>Mage-Flow — native-resolution packing of variable-length image+text tokens through the Native-Resolution MMDiT (left), and the dual-stream MMDiT block (right).</em>
</div>

**Post-training** — from `Base`, generation is aligned with **Diffusion-NFT** (prompt following, aesthetics, text rendering, preference) to produce the RL model, and distilled with **decoupled-DMD + adversarial perceptual guidance** into the 4-step `Turbo`. Editing models reuse the recipe, trained on a mixture of generation and editing data to keep the generative prior.

## 🚀 Quick Start

### Installation

Install everything **except** `flash-attn` first, then install `flash-attn` separately with build isolation **off** — it compiles a CUDA extension against your installed torch, so torch and a matching CUDA toolkit must already be present.

```bash
cd Mage/mage_flow
uv venv && source .venv/bin/activate

# 1) Pinned, tested dependency set (torch 2.13, transformers 5.5, diffusers 0.38, pillow 12.3, …).
#    Recommended for reproducibility. `uv pip install -e .` also works, but its loose
#    bounds may resolve to a newer torch/transformers than the code was tested against.
uv pip install -r requirements.txt
uv pip install -e . --no-deps           # the mage-flow package itself

# 2) flash-attn — needs build tools present and a CUDA toolkit whose MAJOR version
#    matches your torch build (e.g. torch cu12x ↔ nvcc 12.x). A cu13/nvcc-12 mix fails.
uv pip install setuptools wheel ninja
uv pip install --no-build-isolation flash-attn==2.8.3
```

Plain `pip` is equivalent (`pip install -r requirements.txt`, `pip install -e . --no-deps`, then the two flash-attn lines). This registers three commands: `mage-flow`, `mage-flow-edit`, `mage-flow-app`.

> **torch / CUDA:** the default PyPI torch wheel targets the newest CUDA (currently cu13x). If your machine's CUDA toolkit is 12.x, install torch from the matching index first, e.g. `uv pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126`, otherwise the flash-attn build will fail with a CUDA-version mismatch. (torch 2.13.0 ships cu126/cu129/cu130 wheels — pick the one matching your `nvcc`.)

### Python API

`pipe.generate(prompts, **kw)` and `pipe.edit(prompts, ref_images, **kw)` return a `list[PIL.Image]` aligned with `prompts`. A `prompts` **list** is batched into one packed forward per denoise step (each sample can have its own resolution/seed).

**Text-to-image:**

```python
from mage_flow import MageFlowPipeline

pipe = MageFlowPipeline.from_pretrained("microsoft/Mage-Flow", device="cuda")

# 1) single image
img = pipe.generate(["A close-up portrait of an elderly African man with deep wrinkles, wearing a traditional hat, soft natural lighting, ultra realistic."],
                    steps=20, cfg=5.0, heights=[1024], widths=[1024])[0]
img.save("t2i.png")

# 2) batch: several prompts / resolutions / seeds in ONE packed forward per step
imgs = pipe.generate(
    ["the Salar de Uyuni mirror surface captured at high noon, with intimate stillness permeating the air. dew beads on every blade of grass. National Geographic editorial, cinematic depth, fine-grained natural texture.", 
     "A close-up portrait of an elderly African man with deep wrinkles, wearing a traditional hat, soft natural lighting, ultra realistic.", 
     "An immersive close-up of a steaming bowl of Sichuan mapo tofu over jasmine rice served on a hand-thrown ceramic plate, finished with a wedge of citrus. Surface oils catch a tiny specular highlight. Shot with a Hasselblad H6D-100c, ambient window light, the kind of image that makes the viewer hungry."],
    heights=[512, 1024, 1792], widths=[2048, 1024, 1024],   # per-sample; 4:1 is fine
    seeds=[1, 2, 3], steps=20, cfg=5.0,
)
```

**Image editing:**

```python
from mage_flow import MageFlowPipeline

pipe = MageFlowPipeline.from_pretrained("microsoft/Mage-Flow-Edit", device="cuda")

# single reference (path or PIL image)
img = pipe.edit(["Replace the background with a field of sunflowers"], ["assets/dog.jpg"],
                steps=30, cfg=5.0, max_size=1024)[0]
img.save("single_edit.png")

# multi-image edit — ref_images[i] is a LIST of source images
img = pipe.edit(["blend the object from image 2 into image 1"],
                [["scene.png", "object.png"]], steps=30, cfg=5.0)[0]
img.save("multi_edit.png")

# explicit output size (overrides max_size); Turbo edit = 4 steps / cfg 1
img = pipe.edit(["Replace the background with a field of sunflowers"], ["assets/dog.jpg"],
                heights=[1024], widths=[1024], steps=4, cfg=1.0)[0]
img.save("single_edit_1024x1024.png")
```

**Parameters** (shared by `generate` / `edit`):

| Parameter                        | Default                            | Description                                                                                                                                                                                      |
| :------------------------------- | :--------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompts`                      | —                                 | string or list of strings; a list is batched into one forward per step                                                                                                                           |
| `ref_images` *(edit)*        | —                                 | per prompt: one image/path, or a**list** of images for multi-image edit                                                                                                                    |
| `steps`                        | `30`                             | denoising steps — Base`30`, RL `20`, Turbo `4`                                                                                                                                            |
| `cfg`                          | `5.0`                            | classifier-free guidance scale (Turbo:`1.0`)                                                                                                                                                   |
| `heights`, `widths`          | `[1024]`                         | per-sample output size, multiple of 16; native resolution`512`–`2048`                                                                                                                       |
| `max_size` *(edit)*          | source size                        | longest output side; short side follows the reference's aspect ratio                                                                                                                             |
| `vl_cond_long_edge` *(edit)* | `384`                            | cap the long edge of the reference image fed to the**VL text encoder** (matches training preprocessing; the VAE/generation path keeps the full output resolution). `0`/`None` disables |
| `neg_prompts`                  | `" "`                            | per-sample negative prompt (applied when`cfg > 1`)                                                                                                                                             |
| `seeds`                        | `42`                             | per-sample seed;`-1` = random                                                                                                                                                                  |
| `batch_cfg`                    | `True`                           | fuse the CFG conditional + unconditional passes into one packed forward                                                                                                                          |
| `renormalization`              | `False`                          | rescale guided velocity per token (reduces over-saturation at high cfg)                                                                                                                          |
| `static_shift`                 | `6.0`                            | override the flow-matching sigma shift                                                                                                                                                           |
| `prompt_template`              | `mage-flow` / `mage-flow-edit` | text-encoder prompt template                                                                                                                                                                     |

### CLI

```bash
# text-to-image (two prompts in one batch)
mage-flow --prompt "A close-up portrait of an elderly African man with deep wrinkles, wearing a traditional hat, soft natural lighting, ultra realistic." "An immersive landscape of a Greenlandic icefjord at midnight sun, painted by early dawn light, crystal-clear skies adding drama. fine grains of sand carving sharp shadows. Peter Lik gallery print, moody atmosphere, museum-grade composition." \
          --model_path microsoft/Mage-Flow --steps 20 --cfg 5.0 \
          --height 1024 512 --width 1024 2048 --seed 42 --out ./outputs


# editing (one --ref per prompt; comma-separate sources for multi-image edit)
mage-flow-edit --prompt "Replace the background with a field of sunflowers" "blend these two images" \
               --ref assets/dog.jpg "scene.png,object.png" \
               --model_path microsoft/Mage-Flow-Edit --max_size 1024 --out ./outputs
```

| Flag                  | Scope | Meaning                                                                       |
| :-------------------- | :---: | :---------------------------------------------------------------------------- |
| `--prompt`            | both  | one or more prompts, run as a batch (sample `i` uses `--seed + i`)            |
| `--model_path`        | both  | local repo dir or HF Hub repo id (auto-downloaded + cached)                    |
| `--steps`             | both  | number of denoising steps                                                     |
| `--cfg`               | both  | classifier-free guidance scale                                                |
| `--height`            | both  | output height — one value, or one per prompt for mixed resolutions            |
| `--width`             | both  | output width — one value, or one per prompt for mixed resolutions             |
| `--seed`              | both  | base seed (sample `i` uses `--seed + i`)                                       |
| `--neg_prompt`        | both  | negative prompt                                                               |
| `--static_shift`      | both  | override the flow-matching sigma shift                                         |
| `--out`               | both  | output directory                                                              |
| `--ref`               | edit  | reference image per prompt (comma-separate paths for a multi-image edit)      |
| `--max_size`          | edit  | max size of the reference image                                               |
| `--vl_cond_long_edge` | edit  | VL-condition long edge (default `384`)                                         |

### Gradio app

```bash
mage-flow-app                     # serve on http://0.0.0.0:7860  (or: python -m mage_flow.app)
mage-flow-app --device cuda:0 --text-device cuda:1 # split to two GPUs
```

A web UI with **Text → Image** and **Image Edit** tabs; models load lazily on first use and are cached. Presets default to the `microsoft/Mage-Flow*` **Hugging Face repos** (downloaded + cached on first use); set `MAGEFLOW_HF_DIR` to load local checkpoint dirs instead.

**Launch options:**

| Flag          | Default     | Meaning                                                                     |
| :------------ | :---------- | :-------------------------------------------------------------------------- |
| `--host`    | `0.0.0.0` | bind address                                                                |
| `--port`    | `7860`    | port                                                                        |
| `--device`  | `cuda`    | inference device                                                            |
| `--text-device`  | `cuda`    | text encoding device                                                            |
| `--share`   | off         | create a public Gradio share link                                           |
| `--preload` | *(lazy)*  | comma-separated repo ids / paths to load at startup instead of on first use |

## 📝 Citation

```bibtex
@article{zhang2026mageflow,
  title={Mage-Flow: An Efficient Native-Resolution Foundation Model for Image Generation and Editing},
  author={Zhang, Xinjie and Zhang, Peng and Zheng, Shicheng and Guo, Jinghao and Jia, Zhaoyang and Shen, Yifei and Guo, Xun and Luo, Yuxuan and Li, Jiahao and Xie, Wenxuan and Pu, Fanyi and Zhang, Xiaoyi and Zhang, Kaichen and Guo, Zongyu and Bi, Tianci and Gui, Dongnan and Liu, Zhening and Wen, Zimo and Zheng, Zihan and Yang, Senqiao and Li, Xiao and Wang, Jinglu and Li, Bin and Lu, Yan},
  journal={arXiv preprint arXiv:2607.19064},
  year={2026}
}
```
