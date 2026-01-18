---
title: PNA Assistant
emoji: 👨🏾‍⚕️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.12.0
app_file: app.py
pinned: false
---

<p align="center">
  <img src="https://em-content.zobj.net/source/apple/391/woman-health-worker-medium-skin-tone_1f469-1f3fd-200d-2695-fe0f.png" width="80" alt="PNA Logo">
</p>

<h1 align="center">Professional Nurse Advocate Assistant</h1>

<p align="center">
  <strong>AI-Powered Support for the A-EQUIP Model & Restorative Supervision</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/NurseCitizenDeveloper/PNA-Assistant">
    <img src="https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-yellow" alt="Hugging Face Space">
  </a>
  <a href="https://huggingface.co/NurseCitizenDeveloper/relational-intelligence-medgemma-merged">
    <img src="https://img.shields.io/badge/Model-MedGemma%20Fine--tuned-blue" alt="Model">
  </a>
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
</p>

<p align="center">
  👨🏾‍⚕️ 👩🏽‍⚕️ 👨🏿‍⚕️ 👩🏻‍⚕️ 👩‍⚕️
</p>

---

## 🏥 About

The **Professional Nurse Advocate (PNA) Assistant** is an AI-powered educational tool designed to support nursing professionals in understanding and applying the **A-EQUIP model** (Advocating and Educating for Quality Improvement).

This tool focuses on:
- **Restorative Clinical Supervision** - Supporting staff wellbeing
- **Person-Centred Communication** - Compassionate, reflective dialogue
- **Quality Improvement** - Guiding continuous professional development

---

## ⚠️ Important Disclaimers

> [!CAUTION]
> ### For Educational & Research Purposes Only
> This AI assistant is designed **exclusively for educational and research purposes**. It does NOT provide:
> - Clinical advice or diagnosis
> - Treatment recommendations
> - Mental health crisis support
> 
> **Important Note on AI Limitations:**
> While this assistant may provide general information about accessing services or resources, **we cannot guarantee the accuracy of this information**. AI models can "hallucinate" or provide outdated details. Always verify outputs with authoritative sources.

> [!WARNING]
> ### Not a Substitute for Professional Support
> - This tool is **not a replacement** for qualified Professional Nurse Advocates, clinical supervisors, or mental health professionals.
> - **If you or someone you know is experiencing a mental health crisis, please contact a qualified healthcare professional immediately or call your local emergency services.**

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **RAG-Powered Knowledge** | Retrieval-Augmented Generation using the official A-EQUIP Model Guide |
| **Fine-Tuned Model** | Built on MedGemma, fine-tuned with person-centred nursing language (FONS principles) |
| **Premium UI** | Modern, accessible interface with NHS-inspired design |
| **Diversity-First** | Inclusive design with diverse healthcare worker representation |

---

## 🧠 Technology Stack

- **Base Model**: [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it)
- **Fine-Tuned Model**: [relational-intelligence-medgemma-merged](https://huggingface.co/NurseCitizenDeveloper/relational-intelligence-medgemma-merged)
- **Framework**: Gradio + Hugging Face Spaces (ZeroGPU)
- **Knowledge Base**: RAG with Sentence Transformers + FAISS

---

## 📚 The A-EQUIP Model

The A-EQUIP model provides a framework for Professional Nurse Advocates with four key functions:

1. **Normative** - Monitoring quality and standards
2. **Formative** - Education and skill development
3. **Restorative** - Clinical supervision and staff wellbeing (primary focus)
4. **Personal Action for Quality Improvement** - Driving positive change

---

## 🛠️ Local Development

```bash
# Clone the repository
git clone https://github.com/ClinyQAi/Professional-Nurse-Advocate-Assistant.git
cd Professional-Nurse-Advocate-Assistant

# Install dependencies
pip install -r requirements.txt

# Run locally
python app.py
```

---

## 👤 Author

**Tunde Olatunji**  
*Registered Nurse | AI Researcher | Citizen Developer*

- 🐙 GitHub: [@NurseCitizenDeveloper](https://github.com/NurseCitizenDeveloper)
- 🤗 Hugging Face: [NurseCitizenDeveloper](https://huggingface.co/NurseCitizenDeveloper)
- 🏥 Organisation: [ClinyQAi](https://github.com/ClinyQAi)

---

## 🙏 Acknowledgements

This project builds upon the work of many individuals and organisations:

- **[Foundation of Nursing Studies (FONS)](https://www.fons.org/)** - For person-centred nursing principles
- **[NHS England](https://www.england.nhs.uk/)** - For the A-EQUIP Model and PNA Framework
- **[Google Health AI](https://health.google/)** - For MedGemma foundation model
- **[AI Educator Toolkit](https://github.com/NurseCitizenDeveloper/AI-Educator-Toolkit)** - Author's educational AI resources
- **[Hugging Face](https://huggingface.co/)** - For hosting and ZeroGPU infrastructure
- **[Unsloth](https://github.com/unslothai/unsloth)** - For efficient fine-tuning tools

---

## 📄 License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>Made with ❤️ for the nursing profession</sub>
</p>
