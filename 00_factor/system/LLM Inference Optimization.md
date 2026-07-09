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
