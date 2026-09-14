---
aliases:
  - "稀疏路由的离散选择与数值范围会带来额外训练稳定性风险"
  - Discrete routing and router logit scale add training stability risks
  - MoE 路由稳定性
student_os: knowledge-atom
atom_id: LLM-MOE-010
atom_type: failure-mode
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 稀疏路由的离散选择与数值范围会带来额外训练稳定性风险

<!-- bilingual-en:start -->
*Discrete routing and router logit scale add training stability risks*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 稀疏 MoE 在 dense Transformer 之外增加了会切换参数路径的 router。top-$k$ 决策对分数排序敏感，softmax 又可能在低精度和过大 logits 下数值不稳；路由变化、expert 更新不均和主损失相互反馈，会形成额外的训练失败入口。
>
> <!-- bilingual-en:start -->
> Sparse MoE adds a router that switches parameter paths on top of a dense Transformer. Top-$k$ decisions are sensitive to score order, while softmax can become numerically fragile under low precision and large logits. Routing changes, unequal expert updates, and the main loss feed back into one another, creating additional failure modes.
> <!-- bilingual-en:end -->

## 自然解释

两个 expert 的 router logits 很接近时，一次小更新就可能交换排名，使一批 token 改走另一条参数路径。若某 expert 因此突然收到更多 token，它的梯度分布也随之改变；下一步 router 又在已经变化的 expert 上重新比较。这个离散切换并不意味着一定发散，却让系统比固定 dense 路径更敏感。

<!-- bilingual-en:start -->
When two router logits are close, a small update can reverse their order and send a batch of tokens down a different parameter path. The newly busy expert then receives a different gradient distribution, after which the router compares already changed experts. This discrete switching does not guarantee divergence, but it makes the system more sensitive than a fixed dense path.
<!-- bilingual-en:end -->

Switch Transformer 在其设计中把局部 router 计算转成 float32、缩小初始化尺度，以稳定 bfloat16 训练。ST-MoE 又提出 router z-loss，惩罚过大的 log-partition，限制 logits 失控。这些是有实验证据的具体措施，不是所有 MoE 都必须照抄的唯一处方。

<!-- bilingual-en:start -->
Switch Transformer casts local router computation to float32 and reduces initialisation scale to stabilise bfloat16 training in its design. ST-MoE introduces router z-loss, penalising a large log-partition to restrain runaway logits. These are empirically supported techniques for particular systems, not the only mandatory recipe for every MoE.
<!-- bilingual-en:end -->

> [!warning] 边界
> 训练 loss 抖动时不能只怪 router，也不能只降学习率。应同时检查 logit 范围、路由熵、每 expert 负载、drop、梯度范数、数值精度和 batch 构成，再判断是哪条反馈链失控。
>
> <!-- bilingual-en:start -->
> When training loss oscillates, neither blame the router by default nor merely lower the learning rate. Inspect logit scale, routing entropy, per-expert load, drops, gradient norms, numerical precision, and batch composition before locating the unstable feedback loop.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> router 的 softmax 改用 float32 后不再溢出，是否已经证明整个 MoE 稳定？
>
> **答案：** 没有。它只处理局部数值风险；负载集中、离散切换、expert 梯度、容量和主优化过程仍可能造成不稳定。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§2.4：报告 hard routing、低精度 router softmax、selective float32 与较小初始化尺度的稳定性实验。
- Zoph et al. (2022), [*ST-MoE: Designing Stable and Transferable Sparse Expert Models*](https://arxiv.org/abs/2202.08906)，§3.3–3.4：在 selective float32 仍不足的更大尺度上提出 router z-loss，并把其作用解释为限制 logit 幅度和 roundoff；它与负载均衡损失是两个独立项。
