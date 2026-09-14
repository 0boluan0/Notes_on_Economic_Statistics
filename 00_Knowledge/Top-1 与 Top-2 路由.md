---
aliases:
  - "Top-1 与 Top-2 路由在质量计算和通信之间取舍"
  - Top-1 and top-2 routing trade model behaviour against compute and communication
  - MoE top-1 与 top-2
student_os: knowledge-atom
atom_id: LLM-MOE-004
atom_type: decision-rule
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# Top-1 与 Top-2 路由在质量计算和通信之间取舍

<!-- bilingual-en:start -->
*Top-1 and top-2 routing trade model behaviour against computation and communication*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> top-1 每个 token 只执行一个 expert；完整执行的 top-2 让两个 expert 同时贡献输出。增大 $k$ 会增加 expert 调用、token 派发和容量需求，却不自动提高质量。$k$ 是需要在特定模型、训练配方和硬件上验证的设计变量。
>
> <!-- bilingual-en:start -->
> Top-1 runs one expert per token; a fully executed top-2 route lets two experts contribute to the output. Increasing $k$ increases expert calls, token dispatch, and capacity demand, but it does not automatically improve quality. $k$ is a design variable to validate for the particular model, training recipe, and hardware.
> <!-- bilingual-en:end -->

## 自然解释

在其他条件近似相同、expert 大小相同且两个 assignment 都实际执行时，top-2 对 expert FFN 的主体工作约是 top-1 的两倍，并产生更多跨设备 token 副本。容量核算也必须说明该实现采用哪一种 assignment 基数和接受规则；像 GShard 这样对第二路径使用概率门槛的变体，不能机械假定每个 token 始终执行两次。

<!-- bilingual-en:start -->
With comparable expert sizes and other conditions, top-2 performs roughly twice as much expert-FFN work as top-1 when both assignments execute, and it creates more cross-device token copies. Capacity accounting must state the implementation's assignment base and acceptance rule; variants such as GShard apply a probabilistic threshold to the second route, so one cannot mechanically assume that every token always executes twice.
<!-- bilingual-en:end -->

Switch Transformer 在其 T5 型设置中发现 top-1 能保持或改善其速度—质量权衡，并简化通信；较早的 sparsely-gated MoE 和 GShard 则采用 top-2。正确结论不是“top-1 永远更好”或“top-2 更有表达力所以必胜”，而是先固定比较条件，再测质量、稳定性、drop、吞吐和显存。

<!-- bilingual-en:start -->
Switch Transformer found that top-1 preserved or improved the speed–quality trade-off in its T5-style setting while simplifying communication; earlier sparsely gated MoE and GShard systems used top-2. The right conclusion is neither “top-1 is always better” nor “top-2 must win because it is more expressive.” Hold comparison conditions fixed and measure quality, stability, drops, throughput, and memory.
<!-- bilingual-en:end -->

> [!warning] 边界
> “大约两倍 expert 计算”只描述相同 expert 形状下的主体 FFN 调用。router、共享层、padding、kernel 效率与通信不会都严格按 $k$ 线性变化。
>
> <!-- bilingual-en:start -->
> “Roughly twice the expert computation” refers to principal FFN calls with equal expert shapes. Router cost, shared layers, padding, kernel efficiency, and communication do not all scale exactly linearly with $k$.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么不能只看验证 loss 就把 top-1 改成 top-2？
>
> **答案：** 改变 $k$ 同时改变 expert 计算、派发量、容量和通信；即使 loss 改善，也要确认速度、显存和稳定性是否仍满足目标。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§2.1、表 1：在该论文设置中比较 top-1 Switch 与 top-2 MoE，并说明计算、容量与通信变化。
- Lepikhin et al. (2020), [*GShard*](https://arxiv.org/abs/2006.16668)，§2.2 与算法 1：给出带容量约束和概率性第二路径的 top-2 gating 设计。
- Zoph et al. (2022), [*ST-MoE*](https://arxiv.org/abs/2202.08906)，§5.2 与表 8：在更大训练尺度和固定 capacity factor 下观察到 top-2 相对 top-1 的小幅收益，说明早期结论会随实验尺度改变。
