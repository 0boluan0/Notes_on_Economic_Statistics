---
aliases:
  - "稀疏 MoE 将总参数容量与每个 token 的激活计算部分解耦"
  - Sparse MoE partially decouples total capacity from per-token computation
  - MoE 的总参数与激活计算
student_os: knowledge-atom
atom_id: LLM-MOE-001
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 稀疏 MoE 将总参数容量与每个 token 的激活计算部分解耦

<!-- bilingual-en:start -->
*Sparse MoE partially decouples total parameter capacity from per-token computation*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 稀疏 Mixture of Experts（MoE，专家混合模型）把一组候选子网络同时放进模型，却让每个 token 只经过其中少数几个。这样，总 expert 参数可以随 expert 数量增长，而单个 token 在 expert 部分执行的计算主要随激活数量增长；两者被部分解耦，而不是变成毫无关系。
>
> <!-- bilingual-en:start -->
> A sparse Mixture of Experts (MoE) places many candidate subnetworks in a model but sends each token through only a few of them. Total expert parameters can therefore grow with the number of experts, while computation in the expert portion for one token grows mainly with the number activated. The two quantities are partially decoupled, not made unrelated.
> <!-- bilingual-en:end -->

## 自然解释

设一个典型 Transformer 保留共享的 attention 和其他 dense 层，只把若干 FFN 换成 $E$ 个 expert。若每个 token 只激活 $k\ll E$ 个 expert，那么一个粗略口径是：模型拥有约 $P_{shared}+E P_{expert}$ 的参数容量，而该 token 的前向只使用约 $P_{shared}+kP_{expert}$ 的参数路径。第二个式子近似描述 active parameters，不是 FLOPs 公式；实际算术量还取决于矩阵形状、token 数和实现。增加 $E$ 可以扩大可学习容量，而不必让每个 token 逐一运行全部 expert。

<!-- bilingual-en:start -->
Suppose a typical Transformer keeps shared attention and other dense layers, while replacing selected FFNs with $E$ experts. If each token activates only $k\ll E$ experts, a rough accounting is $P_{shared}+E P_{expert}$ total parameter capacity, while that token's forward pass uses a parameter path of about $P_{shared}+kP_{expert}$. The second expression approximates active parameters, not FLOPs; actual arithmetic also depends on matrix shapes, token count, and implementation. Increasing $E$ can expand learned capacity without running every expert on every token.
<!-- bilingual-en:end -->

这就是“参数多、激活计算少”的准确含义。它没有说整个模型的 FLOPs 与 expert 数量严格无关，也没有包含 router、容量填充、token 派发和通信成本；这些系统成本可能抵消一部分算术节省。

<!-- bilingual-en:start -->
This is the precise meaning of “many parameters, sparse active computation.” It does not imply that whole-model FLOPs are strictly independent of expert count, nor does it include routing, capacity padding, token dispatch, or communication. Those system costs can offset part of the arithmetic saving.
<!-- bilingual-en:end -->

> [!warning] 边界
> [[MoE 比较口径|总参数]]描述可学习权重的总量；它既不等于每 token 激活参数，也不等于实际延迟。共享层仍会被每个 token 使用，稀疏的通常只是指定的 expert 层。
>
> <!-- bilingual-en:start -->
> [[MoE 比较口径|Total parameters]] describe all learnable weights. They equal neither active parameters per token nor observed latency. Shared layers still run for every token; sparsity usually applies only to designated expert layers.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 一个模型从 8 个 expert 扩到 64 个，但仍让每个 token 只用 2 个。哪些量明显增长，哪些量不应直接按 8 倍估算？
>
> **答案：** 总 expert 参数明显增长；每 token 的 expert 主体计算仍由 2 个激活 expert 决定，不能按 expert 总数直接乘 8。不过存储、路由与通信是否增长，要另行测量。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§1–2：把稀疏激活表述为增加参数量而维持可控的每样本计算，并明确指出通信与稳定性代价。
- Shazeer et al. (2017), [*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538)，§2：给出每个样本仅调用少数 expert 的条件计算机制。
- Clark et al. (2022), [*Unified Scaling Laws for Routed Language Models*](https://proceedings.mlr.press/v162/clark22a.html)，§1–2：把参数容量与处理一个输入的计算要求作为近乎独立、但并非同一含义的两个缩放轴。
