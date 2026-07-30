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

## 诊断与稳健性

- 每项优化同时测质量、吞吐、延迟、显存、成本和故障率，避免只优化单一指标。
- MoE 检查路由负载均衡；KV cache 检查上下文长度与并发；量化检查长尾输入和任务回归。

## 复现规范

记录模型/数据版本、硬件、软件栈、并行策略、精度、量化配置、基线指标和测量脚本；结果必须能与未优化基线逐项比较。
