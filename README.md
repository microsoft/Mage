<div align="center">

# Mage: A Lightweight, Research-Friendly Multimodal Model Family

<p>
  <b>Microsoft Mage Team</b>
</p>

<p>
  <a href="https://github.com/microsoft/Mage"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repo-181717?logo=github&logoColor=white" height="22" /></a>
  &nbsp;
  <a href="https://huggingface.co/microsoft/Mage"><img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97-Models-yellow" height="22" /></a>
  &nbsp;
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg" height="22" /></a>
</p>

</div>

---

**Mage** is a family of lightweight, research-friendly multimodal models built at a fixed **4B-parameter** budget. It is designed to make advanced visual **understanding** and **generation** accessible for controlled experiments, post-training research, and vertical-domain applications under realistic compute budgets.

The model family is organized around two complementary model families that share the *codec-aligned efficiency* philosophy — spend representation capacity where the signal does — applied to both the understanding and the generation side:

| Model | Task | Scale | Directory |
| :--- | :--- | :---: | :--- |
| **[Mage-VL](mage_vl/)** | Image & video understanding (vision–language) | 4B | [`mage_vl/`](mage_vl/) |
| **[Mage-Flow](mage_flow/)** | Text-to-image generation & instruction-based editing | 4B | [`mage_flow/`](mage_flow/) |

---

## Responsible AI

These models are released for research purposes only and are not intended for product or service deployment. Responsible AI considerations were incorporated throughout the development process, including data selection, model training, and evaluation. The training data includes a combination of public, licensed, and internal datasets that were processed to remove clearly identifiable personal information and reduce harmful content where possible. However, as the data is largely sourced from web-scale collections, it may contain biases or uneven representation. As a result, the models may generate outputs that are inaccurate, biased, or inappropriate under certain prompts. The models should be used in controlled research settings with appropriate human oversight, and downstream users are responsible for applying additional safeguards — such as content moderation, validation, and compliance checks — before broader use.

## Privacy

This project does not collect any usage data. For more information, see the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839).

## License

This project is released under the [MIT License](LICENSE).
