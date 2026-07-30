---
aliases:
  - "Scaling, Mixture of Experts and Distributed Training"
  - "Scaling Laws"
  - "Mixture of Experts"
  - "大模型训练"
status: source-checked
---

# 规模化、MoE 与分布式训练

> [!summary] 快速恢复
> **它解决什么：** 决定有限算力下该训练多大的模型、喂多少数据，并把一次超大训练拆到许多设备上。
> **具体锚点：** 同样的计算预算若全用来扩大参数却不给足 token，模型可能“训练不足”；Chinchilla 一类结果正是修正这种分配。
> **核心难点：** scaling law 是特定数据与训练区间内的经验关系，不是自然定律；MoE 降低每 token 激活计算，却增加路由、通信和容量管理。
> **为什么重要：** 参数量、总训练 FLOPs、推理成本和实际质量不是同一个排行。
> **继续：** 先用预算约束理解 dense scaling，再看 MoE 和并行；部署端转到 [[LLM 推理效率]]。

> [!source] 本节依据
> - [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
> - 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。

## Scaling laws

经验上，测试损失常随模型规模、数据量和计算量呈近似幂律下降。它帮助外推和分配实验，但只在观测区间、数据分布与训练配方近似不变时可信；能力指标还可能出现阈值、污染或测量噪声。

## Compute-optimal training

固定训练 FLOPs 时，要在参数 $N$ 与训练 token $D$ 之间分配。早期模型常偏大而数据不足；compute-optimal 研究表明更小模型配更多 token 可能在相同训练预算下达到更低损失。若模型会被大量调用，训练最优还需加入推理生命周期成本。

## 数据、模型与算力

有效 token 不是原始字节数：去重、质量、语言/领域混合、污染和重复 epoch 都会影响边际价值。比较模型时至少同时记录参数、激活参数、训练 token、精度、硬件和估算 FLOPs。

## MoE

Mixture of Experts 用 router 为每个 token 选择少数前馈 expert。总参数可很大而每 token 只激活一部分。主要问题是负载均衡、expert capacity、token dropping、跨设备 all-to-all 通信与训练稳定性；“总参数”尤其不能直接代表单次推理计算。

## 分布式训练

data parallel 复制模型、分数据并同步梯度；tensor parallel 拆单个算子；pipeline parallel 拆层并处理 micro-batch；参数/梯度/优化器状态还可分片。实际方案同时受显存、网络带宽、延迟、利用率和容错约束。

## 比较训练报告

先统一口径：dense 还是 MoE、总参数还是激活参数、预训练还是继续训练、tokenizer 与数据是否可比。没有数据组成、训练 token 和评测协议，单个参数数字只能说明很少。

## 最小自检

### 固定训练预算下，为什么参数越大未必越好？

> [!answer]- 答案
> 更大模型消耗更多每 token 计算，可能被迫看到更少数据而训练不足；需联合选择参数与 token。
### MoE 的“参数多、计算省”具体是什么意思？

> [!answer]- 答案
> 总 expert 参数很多，但 router 对每个 token 只激活少数 expert；代价转向路由、容量和通信。
### 比较两个模型训练规模时至少要问哪些口径？

> [!answer]- 答案
> 总/激活参数、训练 token、数据组成、数值精度、训练 FLOPs/硬件，以及 dense 或 MoE。

## 来源与核验

- [[06_paper/LLM/LLM Map Index]] 及其链接的论文笔记：用于定位主题与原论文。
- 对应 Zotero 原论文：核验架构、训练方法、实验条件与结论；论文笔记本身不替代原文。
- Kaplan et al. (2020)、Hoffmann et al. (2022)、Fedus et al. (2022)：核验 scaling、compute-optimal 与 sparse MoE。
