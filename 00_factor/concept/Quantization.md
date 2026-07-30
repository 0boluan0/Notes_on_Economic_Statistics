---
aliases:
  - Quantization
  - "量化"
tags:
  - concept
  - llm
  - llm/concept
---

# Quantization

Quantization 是用更低精度表示权重或激活，以降低显存、带宽和推理成本。

## 最小定义
把 FP16/BF16 等高精度数值近似成 INT8、INT4 等低精度格式。

## 最小例子
本地部署模型常用 4-bit 量化换取更低显存占用。

## 相关卡片
[[LLM Efficiency Engineering]]、[[LLM Inference Optimization]]
