---
aliases:
  - "大模型推理优化"
  - "Inference Optimization"
tags:
  - llm
  - llm/system
---

# LLM Inference Optimization

LLM Inference Optimization 关注模型上线后的延迟、吞吐、显存和单位 token 成本。

## 诊断问题
- 负载是短问答、长上下文、批量生成还是 agent 多轮？
- 主要瓶颈是 prefill、decode、KV cache、内存带宽还是调度？

## 代表论文
- [[dao2022FlashAttentionFastMemoryEfficient]]
- [[dao2023FlashAttention2FasterAttention]]
- [[dettmers2023QLoRAEfficientFinetuning]]
- [[deepseek-ai2024DeepSeekV2StrongEconomical]]

## 风险点
推理优化不能只看平均延迟；长尾、并发、上下文长度和质量回退都要一起看。

## 诊断与稳健性

- 分开测量 TTFT、decode tokens/s、端到端延迟、p95/p99 和峰值显存，并按上下文长度、输出长度、并发度分层。
- 量化、KV cache、continuous batching 或投机解码上线前，必须与基线比较任务质量和失败样例。
- 显存异常先排查 KV cache、序列长度、batch 策略和碎片，再调整量化或并行度。

## 关联卡片

- [[LLM Evaluation]]
- [[LLM Efficiency Engineering]]

## 复现规范

记录 GPU、驱动/框架版本、权重、精度、量化方案、请求分布、并发度和测量脚本；每项优化保留基线和回归结果。
