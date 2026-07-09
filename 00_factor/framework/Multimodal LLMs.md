---
aliases:
  - "多模态LLM"
  - "Multimodal Models"
tags:
  - llm
  - llm/route
---

# Multimodal LLMs

Multimodal LLMs 解释语言模型如何与图像、视频、音频或动作空间结合，成为通用接口而非纯文本系统。

## 核心直觉
语言提供统一的任务接口，多模态编码器提供感知输入，二者对齐后模型可以用语言推理视觉和其他模态内容。

## 代表论文
- [[alayrac2022FlamingoVisualLanguage]]
- [[li2023BLIP2BootstrappingLanguageImage]]
- [[openai2024GPT4TechnicalReport]]
- [[team2025GeminiFamilyHighly]]

## 边界
多模态能力需要评测真实感知、定位和跨模态推理，不能只看文本 benchmark。
