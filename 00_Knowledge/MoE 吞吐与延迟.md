---
aliases:
  - "激活 FLOPs 较低不保证 MoE 的端到端吞吐更高或延迟更低"
  - Lower active FLOPs do not guarantee higher end-to-end MoE throughput or lower latency
  - MoE FLOPs 与实际速度
student_os: knowledge-atom
atom_id: LLM-MOE-012
atom_type: systems-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 激活 FLOPs 较低不保证 MoE 的端到端吞吐更高或延迟更低

<!-- bilingual-en:start -->
*Lower active FLOPs do not guarantee higher end-to-end MoE throughput or lower latency*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> active FLOPs 只估算被执行算术操作的数量。MoE 的实际时间还包括 router、token 重排、collective 通信、padding 或不规则 sparse kernel、内存访问和最慢 expert 的等待。算术量较低是潜在加速条件，不是端到端加速证明。
>
> <!-- bilingual-en:start -->
> Active FLOPs estimate the arithmetic operations that execute. MoE wall-clock time also includes routing, token permutation, collectives, padding or irregular sparse kernels, memory traffic, and waiting for the slowest expert. Lower arithmetic is an opportunity for speedup, not proof of end-to-end acceleration.
> <!-- bilingual-en:end -->

## 自然解释

dense FFN 通常形成规则的大矩阵乘，硬件利用率高。MoE 把同一批 token 拆成多个大小不等的 expert batch：如果每个 batch 太小，矩阵核心吃不满；如果为固定容量 padding，又会计算空槽；若 expert 跨慢网络，[[MoE 专家并行|all-to-all]] 还可能超过省下的 FFN 时间。

<!-- bilingual-en:start -->
A dense FFN usually forms a regular large matrix multiplication with high hardware utilisation. MoE splits the same tokens into unequal expert batches. Small batches underfill matrix engines; fixed-capacity padding computes empty slots; and when experts cross a slow network, [[MoE 专家并行|all-to-all]] can cost more than the saved FFN time.
<!-- bilingual-en:end -->

训练看 tokens/s、达到同等质量所需时间和设备利用率；在线推理还要分别看首 token 延迟、每 token 延迟、并发吞吐和尾延迟。大 batch 训练中的吞吐收益，不能自动外推到 batch 很小的交互式推理。

<!-- bilingual-en:start -->
For training, measure tokens per second, time to matched quality, and device utilisation. Online inference additionally separates time to first token, time per output token, concurrent throughput, and tail latency. A throughput gain under large training batches does not automatically transfer to small-batch interactive inference.
<!-- bilingual-en:end -->

> [!warning] 边界
> FLOPs 仍然有用：它隔离了算法算术成本。错误在于把 FLOPs 当成时间单位。判断系统收益必须在同一硬件、精度、batch、序列长度和质量目标下测 wall-clock。
>
> <!-- bilingual-en:start -->
> FLOPs remain useful because they isolate arithmetic cost. The mistake is treating them as units of time. Systems benefit must be measured at matched hardware, precision, batch size, sequence length, and quality target.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 一个 MoE 每 token FLOPs 比 dense 低 30%，但 tokens/s 反而更低。最先应拆哪些时间？
>
> **答案：** router 与重排、派发/返回通信、expert kernel 利用率、padding 或不规则负载、内存访问和 straggler 等待，而不是重复引用 FLOPs 数字。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§1–3 与表 1：区分 matched FLOPs、实际 examples/s 和 time-to-quality，并指出通信成本。
- Gale et al. (2023), [*MegaBlocks*](https://proceedings.mlsys.org/paper_files/paper/2023/hash/5a54f79333768effe7e8927bcccffe40-Abstract-mlsys2023.html)，§1–4：分析 padding、dropping、稀疏 kernel 与实际训练效率的关系，并以端到端训练速度而非 FLOPs 单独验证系统收益。
