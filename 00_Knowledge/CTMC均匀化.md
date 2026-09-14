---
aliases:
  - "均匀化用支配 Poisson 时钟精确表示有界率 CTMC"
  - CTMC uniformization
  - Jensen method
  - Randomization method
  - 均匀化
student_os: knowledge-atom
atom_id: PROB-CTMC-020
atom_set: continuous-time-markov-chains
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[生成矩阵约束]]"
  - "[[CTMC转移半群]]"
  - "[[有界率非爆炸]]"
related:
  - "[[生成矩阵分解]]"
  - "[[嵌入跳链]]"
  - "[[CTMC矩阵指数]]"
leads_to:
  - "[[CTMC平稳分布]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 均匀化用支配 Poisson 时钟精确表示有界率 CTMC
<!-- bilingual-en:start -->
*Uniformization represents a bounded-rate CTMC exactly with a dominating Poisson clock*
<!-- bilingual-en:end -->

> [!summary] 精确随机化而非时间离散近似
> 若出口率有统一上界，选
> $$
> \nu\ge\sup_iq_i>0,
> \qquad R=I+\frac Q\nu.
> $$
> 则 $R$ 是行随机矩阵，而且
> $$
> P(t)=e^{-\nu t}\sum_{n=0}^{\infty}\frac{(\nu t)^n}{n!}R^n.
> $$
> 即先取 $N(t)\sim\operatorname{Poisson}(\nu t)$，再运行 $N(t)$ 步离散链 $R$。这与原 CTMC 的有限维分布完全一致。
> <!-- bilingual-en:start -->
> Uniformization is an exact Poisson mixture of a discrete transition matrix, not a fixed-grid approximation.
> <!-- bilingual-en:end -->

$R_{ii}=1-q_i/\nu$ 允许 self-loop：Poisson 时钟响了，但此次候选事件被“稀释”，状态不变。这是 virtual jump；真正的 [[嵌入跳链|embedded jump chain]] 只记录状态改变，通常令非吸收态的对角元为零。

有限状态自动存在有限的最大出口率。可数状态只有在 $\sup_iq_i<\infty$ 时才能用一个全局 $\nu$；若速率无界，不能把不存在的最大值塞入公式。若所有状态都吸收，直接有 $P(t)=I$，也可任取 $\nu>0$ 得 $R=I$。

> [!example] 选择更大的 ν 不改变模型
> 把 $\nu$ 增大，会产生更多 Poisson 事件，同时提高 $R_{ii}$、增加 virtual jumps；两者恰好抵消，真实状态路径分布不变，只改变计算量。

> [!question]- 自检
> uniformized chain 的一次 self-loop 是否等于 CTMC 发生了一次真实跳跃又立即跳回？
>
> **答案：** 不是。它是支配 Poisson 时钟中的虚事件，原 CTMC 状态没有改变。

## 来源与核验

- [Rao and Teh, 2013, Proposition 1](https://jmlr.csail.mit.edu/papers/volume14/rao13a/rao13a.pdf#page=5)：核对支配率、virtual jumps、Poisson mixture 及有限维分布完全相同。
- [Ward Whitt, Continuous-Time Markov Chains](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf)：核对 uniformization/randomization 的矩阵表示。
