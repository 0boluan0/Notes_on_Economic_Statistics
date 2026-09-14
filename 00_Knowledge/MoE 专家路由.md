---
aliases:
  - "常见稀疏 MoE 层由路由器为每个 token 选择少数前馈专家"
  - An MoE layer routes each token to a small subset of feed-forward experts
  - MoE 路由器
student_os: knowledge-atom
atom_id: LLM-MOE-002
atom_type: mechanism
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# 常见稀疏 MoE 层由路由器为每个 token 选择少数前馈专家

<!-- bilingual-en:start -->
*An MoE layer uses a router to select a small subset of feed-forward experts for each token*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 在常见的稀疏 Transformer MoE 层中，router（路由器）读取当前 token 的 hidden state，为所有可用 expert 产生分数，再只把该 token 派发给得分最高的少数 expert。被替换的通常是逐 token 的前馈网络，而 attention 等共享部分仍按原路径执行。
>
> <!-- bilingual-en:start -->
> In a common sparse Transformer MoE layer, a router reads the current token hidden state, scores the available experts, and dispatches that token only to a few top-scoring experts. The replaced component is usually the token-wise feed-forward network, while shared components such as attention continue along the ordinary path.
> <!-- bilingual-en:end -->

## 自然解释

若 token 表示为 $x$，最简单的 router 可用线性映射得到 logits：$h(x)=W_r x$，再经 softmax 得到对 $E$ 个 expert 的权重 $p_i(x)$。路由决策发生在每一个 MoE 层；同一个 token 在不同层看到的 hidden state 已经变化，因此不必始终去同一 expert。

<!-- bilingual-en:start -->
For token representation $x$, a simple router computes logits $h(x)=W_r x$ and applies softmax to obtain weights $p_i(x)$ over $E$ experts. Routing occurs separately at every MoE layer. Since the token hidden state changes across layers, the same token need not visit the same expert throughout the network.
<!-- bilingual-en:end -->

“expert”在这里通常不是一个完整模型，而是结构相同、参数不同的 FFN 子网络。router 做的是条件选择：让不同 token 使用不同参数路径；随后还需要[[MoE 专家并行|派发与回收]]，才能在多设备上真正执行这些路径。

<!-- bilingual-en:start -->
An “expert” here is usually not a complete model but an FFN subnetwork with the same architecture and different parameters. The router performs conditional selection so different tokens use different parameter paths. Distributed execution still requires [[MoE 专家并行|dispatching tokens to expert devices and returning their outputs]].
<!-- bilingual-en:end -->

> [!warning] 边界
> token-choice learned routing 只是 MoE 的一种常见形式。也存在固定哈希、跨 token 的平衡分配和 expert-choice 等路由设计；因此“学习一个线性 router，让每个 token 独立选 top-$k$”不能当作所有 MoE 的定义。
>
> <!-- bilingual-en:start -->
> Learned token-choice routing is a common MoE design, not the only one. Fixed hashing, balanced assignment across tokens, and expert-choice routing also exist, so learning a linear router that independently selects top-$k$ experts per token is not a universal definition of MoE.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 为什么一个词在第 4 层和第 12 层可能被送到不同 expert？
>
> **答案：** 每层 router 读取的是该层当前的 hidden state；经过前面层后表示和语境已经变化，路由分数也会随之变化。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，§2.1 与图 2：给出 token-level router、softmax 概率和 FFN expert 的结构。
- Shazeer et al. (2017), [*Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538)，§2：给出稀疏门控层及 top-$k$ 选择。
- Roller et al. (2021), [*Hash Layers for Large Sparse Models*](https://papers.nips.cc/paper_files/paper/2021/hash/92bf5e6240737e0326ea59846a83e076-Abstract.html)，§2–3；Lewis et al. (2021), [*BASE Layers*](https://arxiv.org/abs/2103.16716)，§2；Zhou et al. (2022), [*Mixture-of-Experts with Expert Choice Routing*](https://arxiv.org/abs/2202.09368)，§2：分别核验固定哈希、平衡 assignment 与由 expert 选择 token 的非标准路由边界。
