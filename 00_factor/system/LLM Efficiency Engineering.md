---
aliases:
  - "大模型效率工程"
  - "Efficiency / MoE"
tags:
  - llm
  - llm/system
---

# LLM Efficiency Engineering

LLM Efficiency Engineering 关注训练和推理成本如何被架构、内核、并行、缓存、量化和蒸馏共同压低。

## 诊断问题
- 瓶颈在训练吞吐、显存、通信还是推理延迟？
- 成本降低是否牺牲质量、稳定性或上下文能力？

## 代表论文
- [[dao2022FlashAttentionFastMemoryEfficient]]
- [[fedus2022SwitchTransformersScaling]]
- [[jiang2024MixtralExperts]]
- [[deepseek-ai2025DeepSeekV3TechnicalReport]]

## 风险点
效率优化常把复杂度转移到工程系统；MoE 路由、KV cache、量化误差和分布式通信都需要单独验证。

## 相关卡片
[[Mixture of Experts]]、[[KV Cache]]、[[Quantization]]、[[Distributed LLM Training]]、[[LLM Inference Optimization]]
