---
aliases:
  - "LLM Inference Efficiency"
  - "KV Cache"
  - "Quantization"
  - "大模型推理优化"
status: source-checked
---

# LLM 推理效率

> [!summary] 快速恢复
> **它解决什么：** 在显存、吞吐和响应时间约束下，把已经训练好的模型高效地服务给用户。
> **具体锚点：** 长提示先并行做 prefill，随后逐 token decoding；前者常受算力限制，后者频繁读取权重和 KV cache，瓶颈不同。
> **核心难点：** 参数量、首 token 延迟、每 token 延迟、并发吞吐和显存不能用一个数字概括；优化一种指标可能伤害另一种。
> **为什么重要：** 同一模型在不同 batch、上下文长度和硬件上的实际成本可相差很大。
> **继续：** 先测量 prefill/decode 和真实流量分布，再选 batching、量化、attention kernel 或 speculative decoding。

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。

## Prefill 与 decoding

prefill 对整个提示构建各层 KV，容易获得较高并行度；decoding 每步只生成一个或少数 token，却要读取大量权重和历史 KV。报告端到端延迟时至少拆 TTFT（time to first token）、TPOT（time per output token）和吞吐。

## KV cache

self-attention 生成时缓存历史 key/value，避免每步重算前缀。缓存大致随层数、序列长度、batch、KV head 和精度增长。MQA/GQA 减少 KV heads，paged attention 改善不等长请求的显存管理，但都不改变必须验证质量和调度的事实。

## Batching 与 serving

continuous batching 在请求到达/完成时动态填槽，提高设备利用率；代价是排队与尾延迟。吞吐优先的离线任务和低延迟交互服务应使用不同 batch 策略，不能只报满载峰值。

## 量化

权重或激活从 FP16/BF16 降到 INT8/INT4 可减显存与带宽。PTQ 直接校准，QAT 在训练中适应。收益依赖硬件 kernel、量化粒度和异常通道；用目标任务验证困惑度、准确率和生成行为。

## 高效 attention 与 speculative decoding

FlashAttention 通过 IO-aware 分块精确计算 attention，减少高带宽内存读写，并非近似 attention。speculative decoding 用小 draft 模型提出多个 token，再由目标模型并行验收；分布保持正确时，速度取决于接受率和两个模型的成本比。

## 测量口径

固定模型版本、精度、硬件、输入/输出长度分布和并发；同时报告 TTFT、TPOT、tokens/s、峰值显存、能耗/成本和质量变化。一次短提示的 microbenchmark 不能代表生产流量。

## 最小自检

### 为什么长上下文会显著增加 KV cache？

> [!answer]- 答案
> 每层要为每个历史位置保存 key/value，缓存随序列长度和并发近似线性增长。
### FlashAttention 是否通过近似 softmax 换速度？

> [!answer]- 答案
> 不是；它重排分块与内存访问来精确计算 attention，主要减少 HBM IO。
### 一种 INT4 方案吞吐更高，能否直接说更优？

> [!answer]- 答案
> 不能；还要在目标硬件和流量下比较延迟、显存、成本及质量退化。

## 来源与核验

- [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
- 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
