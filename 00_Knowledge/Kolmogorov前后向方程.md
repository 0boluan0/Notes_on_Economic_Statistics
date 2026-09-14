---
aliases:
  - "Kolmogorov 后向与前向方程分别分解第一跳与最后一跳"
  - Kolmogorov forward equation
  - Kolmogorov backward equation
  - CTMC master equation
  - 前向方程与后向方程
student_os: knowledge-atom
atom_id: PROB-CTMC-009
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[生成矩阵约束]]"
  - "[[CTMC转移半群]]"
  - "[[Markov矩阵左右约定]]"
related:
  - "[[CTMC短时概率展开]]"
leads_to:
  - "[[CTMC矩阵指数]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# Kolmogorov 后向与前向方程分别分解第一跳与最后一跳
<!-- bilingual-en:start -->
*The Kolmogorov backward and forward equations decompose the first and last infinitesimal transitions*
<!-- bilingual-en:end -->

> [!summary] 先固定矩阵方向
> 在本库的行概率约定下，有限状态齐次 CTMC 满足
> $$
> \underbrace{P'(t)=QP(t)}_{\text{后向：先看起点后的第一小步}},
> \qquad
> \underbrace{P'(t)=P(t)Q}_{\text{前向：看终点前的最后一小步}},
> \qquad P(0)=I.
> $$
> 分量形式分别为
> $$
> p'_{ij}(t)=\sum_kq_{ik}p_{kj}(t),\qquad
> p'_{ij}(t)=\sum_kp_{ik}(t)q_{kj}.
> $$
> <!-- bilingual-en:start -->
> The backward equation acts on the starting-state index through the first short interval; the forward equation acts on the destination-state index through the last short interval.
> <!-- bilingual-en:end -->

若初始分布是行向量 $\alpha(0)$，时刻 $t$ 的分布为 $\alpha(t)=\alpha(0)P(t)$，于是
$$
\alpha'(t)=\alpha(t)Q,
$$
这常被称为 forward master equation。改用列概率向量时所有乘法方向转置；不能混用公式再靠直觉补救。

有限状态时两式都成立且由同一个 $e^{tQ}$ 解出。可数状态若可能爆炸或 $Q$ 无界，前向式、后向式的存在与唯一性条件并不自动相同，必须说明 regularity。

> [!example] 流入减流出
> 对两状态链，$p_1'(t)=\lambda p_0(t)-\mu p_1(t)$：第一项是流入故障态的概率率，第二项是从故障态修复离开的概率率。

> [!question]- 自检
> 行向量分布 $\alpha(t)$ 应满足 $Q\alpha(t)$ 还是 $\alpha(t)Q$？
>
> **答案：** $\alpha'(t)=\alpha(t)Q$。若写成列向量，才是 $\alpha'(t)=Q^T\alpha(t)$。

## 来源与核验

- [Ward Whitt, Continuous-Time Markov Chains, Theorem 3.1](https://www.columbia.edu/~ww2040/4106S11/CTMCchapter121906.pdf#page=8)：核对前向、后向方程的分量式、矩阵方向和有限状态条件。
- [[01_Math/05_随机过程/05_连续时间的马尔可夫链.md]]：核对课程要求理解两式但不要求手算一般解；方程按标准矩阵乘法记号书写。
