---
aliases:
  - "溢出 token 的丢弃绕过重路由或无丢弃处理属于实现选择"
  - "Dropping, bypassing, rerouting, or dropless handling of overflow tokens is an implementation choice"
  - MoE token overflow policy
student_os: knowledge-atom
atom_id: LLM-MOE-008
atom_type: implementation-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 溢出 token 的丢弃绕过重路由或无丢弃处理属于实现选择

<!-- bilingual-en:start -->
*Dropping, bypassing, rerouting, or dropless handling of overflow tokens is an implementation choice*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 当某个 expert 收到的 assignment 超过固定容量，系统必须决定怎样处理超额部分：跳过该 expert 计算并沿 residual path 传递、保留另一个已选路径或显式重路由、丢弃该 expert 路径，或改用能处理不规则负载的 dropless kernel。overflow 是容量政策产生的情形；具体后果不是所有 MoE 共有的。
>
> <!-- bilingual-en:start -->
> When assignments to an expert exceed fixed capacity, the system must choose what happens to the excess: skip expert computation and continue along a residual path, retain another selected route or explicitly reroute, drop that expert path, or use a dropless kernel that handles irregular load. Overflow is created by a capacity policy; its treatment is not universal across MoE systems.
> <!-- bilingual-en:end -->

## 自然解释

Switch Transformer 的默认做法是让超容量 token 不经过该 Switch FFN，再通过 residual connection 进入下一层；其 Appendix B 另行实验了 No-Token-Left-Behind，让 overflow token 继续尝试次高分 expert，这属于较慢的备选方案。GShard 会分别给 top-1 与概率性 top-2 路径施加容量 mask；只有两个预先选定的 expert 都未贡献时，gate 向量才退化为零，而 Transformer 的 residual connection 仍把原表示传到下一层。这不是 overflow 发生后再搜索第三个 expert。MegaBlocks 则把“丢 token 还是为最坏负载 padding”视为既有实现的两难，并用 block-sparse 计算实现 dropless MoE。

<!-- bilingual-en:start -->
Switch Transformer normally skips the Switch FFN for over-capacity tokens and passes their representation onward through the residual connection. Its Appendix B separately tests No-Token-Left-Behind, which lets overflow tokens try lower-ranked experts and is a slower alternative. GShard applies capacity masks separately to its top-1 and probabilistic top-2 routes; only when neither preselected expert contributes does the gate vector become zero, while the Transformer's residual connection still carries the input onward. This is not a post-overflow search for a third expert. MegaBlocks frames earlier systems as trading token dropping against worst-case padding and implements dropless MoE with block-sparse computation.
<!-- bilingual-en:end -->

这些选择改变的不只是速度。丢弃或绕过会改变某些 token 实际经历的函数；重路由会改变负载和 expert 选择；dropless 执行保留 assignment，却要求 kernel 能有效处理大小不等的 expert batch。因此报告 drop rate 时还要说明“drop”在该实现中具体跳过了什么。

<!-- bilingual-en:start -->
These choices affect more than speed. Dropping or bypassing changes the function actually applied to some tokens; rerouting changes load and expert selection; dropless execution preserves assignments but needs kernels that efficiently handle unequal expert batches. A reported drop rate must therefore define exactly which computation is omitted.
<!-- bilingual-en:end -->

> [!warning] 边界
> 不要把“超额 token 排队”等任意策略写成默认事实。同步训练通常要求规则 shape 和集体通信协调；是否能排队、延迟到何时，以及是否改变一步语义，都取决于系统设计。
>
> <!-- bilingual-en:start -->
> Do not present an arbitrary policy such as “queue the excess tokens” as the default. Synchronous training often coordinates fixed shapes and collectives. Whether queuing is possible, how long it waits, and whether it changes step semantics all depend on the system design.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 两篇论文都报告 1% overflow，为什么质量影响仍可能不同？
>
> **答案：** 一篇可能跳过 expert 但保留 residual，另一篇可能只丢掉其中一条已选路径、显式重路由或使用 dropless 执行；相同百分比不代表相同 token、相同计算或相同函数变化。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§2.2 与 Appendix B：区分默认 residual bypass 与 No-Token-Left-Behind 的逐次重路由实验。
- Lepikhin et al. (2020), [*GShard*](https://arxiv.org/abs/2006.16668)，§2.2 与算法 1：说明两个已选路径各自受容量约束、第二路径使用随机门槛，以及两条路径都无贡献时 residual 仍传递表示。
- Gale et al. (2023), [*MegaBlocks: Efficient Sparse Training with Mixture-of-Experts*](https://arxiv.org/abs/2211.15841)，§1–3：界定 capacity-based dropping/padding 取舍并提出 dropless block-sparse 实现。
