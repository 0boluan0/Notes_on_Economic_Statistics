---
aliases:
  - "Top-k 路由按门控权重组合被选专家的输出"
  - Top-k routing combines selected expert outputs with gate weights
  - MoE top-k 输出
student_os: knowledge-atom
atom_id: LLM-MOE-003
atom_type: mechanism
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Mixture of Experts（MoE）.canvas]]"
---

# Top-k 路由按门控权重组合被选专家的输出

<!-- bilingual-en:start -->
*Top-k routing combines the selected expert outputs using gate weights*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> router 不只决定“去哪几个 expert”，还给出这些路径的门控权重。若 $T(x)$ 是 token $x$ 被选中的 expert 集合，典型输出写成 $y=\sum_{i\in T(x)}p_i(x)E_i(x)$：未被选中的 expert 不为该 token 计算，选中的输出按门控值组合。
>
> <!-- bilingual-en:start -->
> A router decides not only which experts receive a token but also the gate weights on those paths. If $T(x)$ is the selected expert set for token $x$, a typical output is $y=\sum_{i\in T(x)}p_i(x)E_i(x)$. Unselected experts do no work for that token; selected outputs are combined using their gate values.
> <!-- bilingual-en:end -->

## 自然解释

对 top-1 routing，$T(x)$ 只有一个元素，输出就是被选 expert 的结果乘其 gate value。对 top-2，两个 expert 都执行，随后加权相加。这样，expert 参数和 router 参数可以围绕主任务损失共同学习；实际训练还会加入[[MoE 负载均衡损失|负载均衡目标]]等额外信号。

<!-- bilingual-en:start -->
With top-1 routing, $T(x)$ has one member, so the selected expert output is multiplied by its gate value. With top-2, both experts run and their weighted outputs are summed. Expert and router parameters can therefore learn jointly from the main task loss, often with additional signals such as a [[MoE 负载均衡损失|load-balancing objective]].
<!-- bilingual-en:end -->

top-$k$ 包含两个不同步骤：先按分数选择离散集合，再在集合内使用连续权重。论文和实现可能对选中权重重新归一化、加入噪声、设置第二 expert 的阈值，或采用不同梯度处理；写公式时应说明所指变体。

<!-- bilingual-en:start -->
Top-$k$ contains two distinct steps: select a discrete set by score, then apply continuous weights within that set. Papers and implementations may renormalise selected weights, add noise, threshold the second expert, or use different gradient treatments. A formula should identify the intended variant.
<!-- bilingual-en:end -->

> [!warning] 边界
> softmax 给出连续概率，并不使 top-$k$ 集合选择本身处处可微。训练可行依赖被选 gate 值的梯度、辅助目标和具体估计方式，不能仅凭“用了 softmax”推断所有 expert 都得到主任务梯度。
>
> <!-- bilingual-en:start -->
> Softmax produces continuous probabilities, but the top-$k$ set selection itself is not differentiable everywhere. Training relies on gradients through selected gate values, auxiliary objectives, and implementation-specific estimators; softmax alone does not give every expert a main-loss gradient for every token.
> <!-- bilingual-en:end -->

> [!question]- 回忆与应用
> top-2 router 选中 A、B，权重分别为 0.7、0.2。为什么不能说其余 expert 也因 softmax 有非零概率而执行了前向计算？
>
> **答案：** softmax 分数可覆盖全部 expert，但稀疏派发只执行选中集合；其余 expert 对该 token 没有前向输出进入这次加权和。

## 来源与核验

- Fedus, Zoph, and Shazeer (2022), [*Switch Transformers*](https://www.jmlr.org/papers/v23/21-0998.html)，式 (1)–(2)：给出 router softmax、top-$k$ 集合及加权 expert 输出。
- Shazeer et al. (2017), [*Sparsely-Gated Mixture-of-Experts Layer*](https://arxiv.org/abs/1701.06538)，§2：给出 noisy top-$k$ gating 的原始机制。
