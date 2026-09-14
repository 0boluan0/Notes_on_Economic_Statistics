---
aliases:
  - "容量因子在限容量实现中规定每个专家的 token 预算"
  - Capacity factor sets each expert token budget in capacity-limited implementations
  - MoE capacity factor
student_os: knowledge-atom
atom_id: LLM-MOE-007
atom_type: definition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 容量因子在限容量实现中规定每个专家的 token 预算

<!-- bilingual-en:start -->
*Capacity factor sets each expert's token budget in capacity-limited implementations*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 对固定 shape 的限容量实现，capacity factor $c$ 把“均匀路由时的参考容量”放大为每个 expert 的槽位预算。Switch 的 top-1 定义是 $C=cT/E$（实现时还需取整数），其中 $T$ 是该 batch 的 token 数；top-$k$、分组路由和概率性第二路径可能使用不同基数。$c>1$ 留出负载波动缓冲，却也增加 padding、显存、计算或通信开销。
>
> <!-- bilingual-en:start -->
> In a fixed-shape, capacity-limited implementation, capacity factor $c$ scales the reference capacity under balanced routing into a per-expert slot budget. Switch top-1 defines $C=cT/E$ (with integer handling left to the implementation), where $T$ is the number of tokens in the batch; top-$k$, grouped routing, and probabilistic second routes can use different bases. A value $c>1$ buffers load variation, but can add padding, memory, computation, or communication.
> <!-- bilingual-en:end -->

## 自然解释

例如 1,024 个 token、16 个 expert、top-1 routing 产生 $A=1{,}024$ 次 assignment。平均每个 expert 64 次；$c=1.25$ 时可分配约 80 个槽位。若某 expert 收到 140 次，仍有约 60 次超出其预算。capacity factor 保护的是局部波动，不会把 router 自动变均匀。

<!-- bilingual-en:start -->
For 1,024 tokens, 16 experts, and top-1 routing, $A=1{,}024$ assignments. The mean is 64 per expert; $c=1.25$ provides about 80 slots. If one expert receives 140 assignments, about 60 still exceed its budget. Capacity factor buffers local variation; it does not make the router balanced.
<!-- bilingual-en:end -->

在完整执行所有路径的 top-$k$ 中，均匀 assignment 基数常按 $kT/E$ 理解，但这不是通用定义。GShard 把 $N$ 个 token 分成 $G$ 组、每组 $S=N/G$，并为其 top-2 gate 设置组内容量 $C=2S/E$；第二路径还要通过概率门槛，所以实际执行次数可以少于 $2T$。关键是先按实现规则设定容量，再判断哪些 assignment overflow，不能用限容量后已经接受的派发数反推容量。

<!-- bilingual-en:start -->
For a full top-$k$ router that executes every selected path, balanced assignment capacity is often reasoned about from $kT/E$, but this is not a universal definition. GShard partitions $N$ tokens into $G$ groups of size $S=N/G$ and sets group capacity to $C=2S/E$ for its top-2 gate; its second route also passes a probabilistic threshold, so actual executions can be fewer than $2T$. Capacity is set from the implementation's rule before overflow is masked, not inferred afterward from accepted dispatches.
<!-- bilingual-en:end -->

> [!warning] 边界
> capacity factor 不是 MoE 数学定义的一部分。[[MoE 溢出 token 处理|dropless 或动态 shape 实现]]可以不使用相同的固定槽位规则，但会承担别的调度和 kernel 代价。
>
> <!-- bilingual-en:start -->
> Capacity factor is not part of the mathematical definition of MoE. [[MoE 溢出 token 处理|Dropless or dynamic-shape implementations]] need not use the same fixed-slot rule, but incur different scheduling and kernel costs.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 一个 top-2 模型有 1,000 个 token 和 10 个 expert。什么时候可以把均匀容量基数写成每 expert 200 次，什么时候不能直接这样写？
>
> **答案：** 若两个 assignment 对每个 token 都会执行，$2{,}000/10=200$ 是合理基数；若路由按 group 定容量、对第二路径采样或使用别的计数单位，就必须采用论文或实现给出的公式，不能只凭 “top-2” 推断。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，式 (3) 与图 3：定义 top-1 expert capacity 和 capacity factor 的缓冲—浪费取舍。
- Lepikhin et al. (2020), [*GShard*](https://arxiv.org/abs/2006.16668)，§2.2 与算法 1：展示 top-2 routing 的组内容量 $2S/E$、独立容量 mask 和概率性第二路径。
