---
aliases:
  - "Token-choice 路由可能因选择集中形成自强化的专家负载不均"
  - Token-choice routing can create self-reinforcing expert load imbalance
  - MoE 负载不均
student_os: knowledge-atom
atom_id: LLM-MOE-006
atom_type: failure-mode
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# Token-choice 路由可能因选择集中形成自强化的专家负载不均

<!-- bilingual-en:start -->
*Token-choice routing can create self-reinforcing expert load imbalance*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 当每个 token 独立选择得分最高的 expert 时，router 没有天然理由把一个 batch 平均分开。相似 token 的选择会相关，少数 expert 可能拥挤而另一些空闲；这既浪费并行设备，也会在限容量实现中造成 token overflow。
>
> <!-- bilingual-en:start -->
> When each token independently selects its highest-scoring expert, the router has no inherent reason to split a batch evenly. Similar tokens can make correlated choices, overloading a few experts while leaving others idle. This wastes parallel devices and, in capacity-limited implementations, creates token overflow.
> <!-- bilingual-en:end -->

## 自然解释

假设 8 个 expert 各放在一台设备上。一个 batch 有 800 次 top-1 assignment，平均是每个 expert 100 次；但若 300 次都去 expert 3，其他设备即使很空，也不能自动替 expert 3 计算，因为它们没有相同参数。更常被选中的 expert 还会收到更多主任务更新，可能变得更容易继续被选，形成自强化反馈。一步的完成时间则会被最拥挤的路径或固定容量政策决定。

<!-- bilingual-en:start -->
Suppose eight experts reside on eight devices. A batch has 800 top-1 assignments, averaging 100 per expert. If 300 go to expert 3, idle devices cannot automatically compute those tokens because they do not hold the same parameters. A frequently selected expert also receives more main-task updates and may become still more likely to be selected, creating feedback. Step time is governed by the busiest path or the fixed-capacity policy.
<!-- bilingual-en:end -->

负载要在正确粒度上看：逐层、逐 expert、逐 batch 统计实际 assignment、router 概率、容量利用率和 overflow。全训练期平均很均匀，仍可能掩盖某些 batch 的尖峰；某层均匀，也不代表另一层正常。

<!-- bilingual-en:start -->
Load must be measured at the right granularity: actual assignments, router probabilities, capacity utilisation, and overflow by layer, expert, and batch. A balanced training-wide average can hide batch-level spikes, and balance in one layer says nothing about another.
<!-- bilingual-en:end -->

> [!warning] 边界
> 不均衡不等于已经证明“expert collapse”，也不是所有 MoE 的固有问题。健康专门化也会产生非均匀路由，而固定哈希、平衡分配或 expert-choice 会改变这一机制；真正问题是它是否造成持续闲置、容量溢出、优化失灵或系统瓶颈。
>
> <!-- bilingual-en:start -->
> Imbalance alone does not prove “expert collapse,” nor is it inherent to every MoE. Healthy specialisation can also be nonuniform, while fixed hashing, balanced assignment, or expert-choice changes this mechanism. The relevant question is whether routing causes persistent idleness, overflow, optimisation failure, or a systems bottleneck.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么“每个 expert 整个 epoch 收到的 token 数接近”仍不能排除容量溢出？
>
> **答案：** 容量通常按单个 batch 或分组执行；不同 batch 的热点可以互相抵消，使长期平均均匀，却仍在当时超过某个 expert 的预算。

## 来源与核验

- Shazeer et al. (2017), [*Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538)，§4 与 Appendix A：说明早期路由不均会形成“富者愈富”的自强化，并定义 importance 与 load 两类缓解目标。
- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§2.2 与图 3：展示不均派发、expert overflow 与闲置容量。
- Zhou et al. (2022), [*Mixture-of-Experts with Expert Choice Routing*](https://arxiv.org/abs/2202.09368)，§2：用每个 expert 固定 bucket、由 expert 选择 token 的机制说明改变路由方向可以直接改变负载约束。
