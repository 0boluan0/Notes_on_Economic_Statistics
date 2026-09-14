---
aliases:
  - "专家并行先把 token 派发到专家设备再把输出送回"
  - Expert parallelism dispatches token activations to expert devices and returns outputs
  - MoE expert parallel
student_os: knowledge-atom
atom_id: LLM-MOE-011
atom_type: mechanism
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 专家并行先把 token 派发到专家设备再把输出送回

<!-- bilingual-en:start -->
*Expert parallelism dispatches token activations to expert devices and returns their outputs*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> expert parallelism（专家并行）把不同 expert 的权重长期放在不同设备或设备组上。router 选定目的 expert 后，系统按目的地重排并发送 token activation；各设备执行本地 expert，再把结果按原 token 位置送回。分布式 MoE 因而常在每个 expert 层包含一次派发和一次返回通信。
>
> <!-- bilingual-en:start -->
> Expert parallelism places different expert weights persistently on different devices or device groups. After routing chooses destinations, the system groups and sends token activations accordingly; devices run their local experts and return outputs to the original token positions. A distributed MoE layer therefore commonly includes a dispatch and a return communication phase.
> <!-- bilingual-en:end -->

## 自然解释

设设备 A 原本持有 token 1、2 的 hidden states，但 router 让 token 1 去设备 C 上的 expert 7，让 token 2 去设备 B 上的 expert 4。A 先发送 activation；B、C 完成本地 FFN 后再把输出送回 A，A 才能恢复原序列布局并继续下一共享层。多设备实现常用 all-to-all 完成这种多源多目的交换。

<!-- bilingual-en:start -->
Suppose device A initially holds hidden states for tokens 1 and 2, while the router sends token 1 to expert 7 on device C and token 2 to expert 4 on device B. A first sends the activations; B and C run their local FFNs and return the outputs to A, which restores sequence order before the next shared layer. Multi-device implementations commonly use all-to-all collectives for this many-source, many-destination exchange.
<!-- bilingual-en:end -->

expert parallel 可以与 data、tensor 和 pipeline parallel 组合。布局的核心问题是：哪些设备保存哪些 expert、哪些通信留在高速节点内、每次交换多大，以及路由不均会不会让某一设备成为 straggler（拖慢整步的尾部设备）。

<!-- bilingual-en:start -->
Expert parallelism can be combined with data, tensor, and pipeline parallelism. The key layout questions are which devices hold which experts, which exchanges stay within fast intra-node links, how large messages are, and whether routing imbalance creates a straggler that delays the whole step.
<!-- bilingual-en:end -->

> [!warning] 边界
> all-to-all 是常见实现，不是 MoE 定义。expert 都在同一设备时不需要跨设备派发；分层、复制或分组布局也可能改用其他 collective。通信的是 token activation 与结果，不能误写成每个 token 都搬运全部 expert 权重。
>
> <!-- bilingual-en:start -->
> All-to-all is common, not definitional. Collocated experts need no cross-device dispatch, and hierarchical, replicated, or grouped layouts may use other collectives. Token activations and outputs are communicated; the system does not move every expert's weights for every token.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么把 64 个 expert 分到 64 张卡后，路由不均仍会拖慢所有卡？
>
> **答案：** 一步通常要等派发、expert 计算和返回共同完成；最拥挤设备处理或通信最久，会成为同步路径上的尾部瓶颈。

## 来源与核验

- Lepikhin et al. (2020), [*GShard*](https://arxiv.org/abs/2006.16668)，§3.1–3.3 与算法 2：描述 expert 权重按 expert 维分片、token activation 的 dispatch/combine，以及 resharding 所用的 all-to-all。
- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§5.4–5.5：说明 expert 权重跨设备划分及 data/model/expert parallel 的组合。
