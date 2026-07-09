---
aliases:
  - "大模型分布式训练"
  - "Distributed Training"
tags:
  - llm
  - llm/system
---

# Distributed LLM Training

Distributed LLM Training 关注如何把超大模型、超大数据和长时间训练稳定地摊到多机多卡系统上。

## 诊断问题
- 并行策略是数据并行、张量并行、流水线并行还是专家并行？
- 通信、显存和负载均衡哪个是主瓶颈？

## 代表论文
- [[shoeybi2020MegatronLMTrainingMultiBillion]]
- [[smith2022UsingDeepSpeedMegatron]]
- [[lepikhin2020GShardScalingGiant]]
- [[deepseek-ai2025DeepSeekV3TechnicalReport]]

## 风险点
训练稳定性比单步吞吐更重要；故障恢复、数据管线和数值稳定都会影响最终模型。
