---
aliases:
  - "多模态LLM"
  - "Multimodal Models"
tags:
  - llm
  - llm/route
type: framework
---

# Multimodal LLMs

Multimodal LLMs 解释语言模型如何与图像、视频、音频或动作空间结合，成为通用接口而非纯文本系统。

## 核心直觉
语言提供统一的任务接口，多模态编码器提供感知输入，二者对齐后模型可以用语言推理视觉和其他模态内容。

## 什么时候用

- 输入或输出包含图像、音频、视频、空间位置或动作，而文本化会损失关键信息时。
- 需要判断问题主要是感知、跨模态对齐，还是语言推理时。

## 关键假设

- 模态编码器能保留任务相关细节，跨模态映射与语言空间足够对齐。
- 评测数据覆盖分辨率、视角、噪声和真实场景变化。

## 失败模式

- OCR、计数、空间关系和细粒度定位可能先于语言推理出错。
- 文本流畅性会掩盖感知错误；必须分别评估感知、对齐、推理和安全边界。

## 代表论文
- [[alayrac2022FlamingoVisualLanguage]]
- [[li2023BLIP2BootstrappingLanguageImage]]
- [[openai2024GPT4TechnicalReport]]
- [[team2025GeminiFamilyHighly]]

## 边界
多模态能力需要评测真实感知、定位和跨模态推理，不能只看文本 benchmark。
