---
aliases:
  - "总参数、激活参数、FLOPs、显存与质量是不同的 MoE 比较口径"
  - Total parameters active parameters FLOPs memory and quality are distinct MoE metrics
  - MoE 比较口径
student_os: knowledge-atom
atom_id: LLM-MOE-013
atom_type: distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 总参数、激活参数、FLOPs、显存与质量是不同的 MoE 比较口径

<!-- bilingual-en:start -->
*Total parameters, active parameters, FLOPs, memory, and quality are distinct MoE comparison metrics*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> MoE 把原本常随规模一起增长的量拆开了：total parameters 统计全部权重；active parameters 按声明的 token、样本或前向口径统计被选参数路径；FLOPs 统计算术；memory 统计权重、状态、activation 与缓存的驻留；quality 由任务评测给出。active parameters 不是 FLOPs，任意一个数字也不能替代其余数字。
>
> <!-- bilingual-en:start -->
> MoE separates quantities that often grow together in dense models. Total parameters count all weights; active parameters count the selected parameter path under a declared per-token, per-example, or forward-pass convention; FLOPs count arithmetic; memory covers resident weights, states, activations, and caches; quality comes from task evaluation. Active parameters are not FLOPs, and no single number substitutes for the others.
> <!-- bilingual-en:end -->

## 自然解释

一个“1T 参数、每 token 激活 20B”的模型，1T 说明 checkpoint 中所有 expert 与共享权重的总量，20B 说明该口径下单 token 走过的参数路径。它没有单独告诉你训练过多少 token、做了多少总 FLOPs、需要多少台设备存权重、通信多慢，也没有说明在目标任务上是否优于 70B dense 模型。

<!-- bilingual-en:start -->
A model described as “1T parameters, 20B active per token” uses 1T for all expert and shared weights in the checkpoint and 20B for the parameter path traversed by a token under that accounting convention. Those figures alone do not reveal training tokens, total training FLOPs, devices needed for weight storage, communication time, or whether it beats a 70B dense model on the target task.
<!-- bilingual-en:end -->

active parameters 也不是统一测量标准：报告可能只算 expert、也可能加共享层；embedding 是否计入、top-$k$ 各路径是否完整计数，都需查方法。显存更不能由参数量直接推断，因为训练还有梯度和优化器状态，推理还有 KV cache、activation buffer、量化与复制布局。

<!-- bilingual-en:start -->
Active parameters are not a universal convention either. Reports may count only experts or include shared layers; embeddings and all top-$k$ paths may be treated differently. Memory likewise cannot be inferred from parameter count alone because training adds gradients and optimiser state, while inference adds KV cache, activation buffers, quantisation, and replication layout.
<!-- bilingual-en:end -->

> [!warning] 边界
> “参数相同”“FLOPs 相同”和“墙钟预算相同”是三种不同实验。阅读结果时先问作者固定了哪一个，再看结论能回答什么问题。
>
> <!-- bilingual-en:start -->
> “Parameter matched,” “FLOP matched,” and “wall-clock matched” describe different experiments. First identify what the authors held fixed, then decide which question the result can answer.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么 500B-total / 30B-active 的 MoE 不能只凭 500B 这个数字与 70B dense 比推理成本？
>
> **答案：** 500B 是全部权重容量；推理计算更接近 active path，但实际成本还受共享层、$k$、显存布局、通信、batch、kernel 与延迟目标影响。

## 来源与核验

- Clark et al. (2022), [*Unified Scaling Laws for Routed Language Models*](https://proceedings.mlr.press/v162/clark22a.html)，§2–4：把 dense parameters、expert 数与计算预算作为不同缩放变量建模。
- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§1–3：分别报告总规模、每 token FLOPs、实际速度与质量。
