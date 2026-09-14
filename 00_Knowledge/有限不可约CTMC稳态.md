---
aliases:
  - "有限不可约 CTMC 存在唯一且对每个状态均为正的平稳分布"
  - Finite irreducible CTMC stationary distribution
  - Unique stationary law of a finite irreducible CTMC
student_os: knowledge-atom
atom_id: PROB-CTMC-032
atom_set: continuous-time-markov-chains
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC平稳分布]]"
  - "[[跳链决定CTMC沟通类]]"
related:
  - "[[有限不可约链稳态]]"
  - "[[有限CTMC平稳生成矩阵判据]]"
leads_to:
  - "[[有限CTMC平稳分解]]"
  - "[[有限CTMC全局收敛]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 有限不可约 CTMC 存在唯一且对每个状态均为正的平稳分布
<!-- bilingual-en:start -->
*A finite irreducible CTMC has a unique stationary distribution with positive mass on every state*
<!-- bilingual-en:end -->

> [!summary] 类内存在、唯一且全支持
> 若 CTMC 的状态空间有限且链不可约，则存在唯一平稳分布 $\pi$，并且
> $$
> \pi_i>0\qquad\text{对每个状态 }i.
> $$
> 有限状态使过程自动非爆炸；不可约使所有状态属于同一个闭沟通类。这里不需要额外的非周期条件。
> <!-- bilingual-en:start -->
> Finite irreducibility gives one full-support stationary distribution. Aperiodicity is not needed for existence, uniqueness, or positivity in continuous time.
> <!-- bilingual-en:end -->

有限不可约情形的唯一性是类内结论。有限可约链需把多个闭类的平稳分布组合成全链平稳分布，见 [[有限CTMC平稳分解]]；从任意初态的收敛条件见 [[有限CTMC全局收敛]]。

> [!example] 两状态链
> 对
> $$Q=\begin{pmatrix}-\lambda&\lambda\\ \mu&-\mu\end{pmatrix},
> \qquad \lambda,\mu>0,$$
> 链有限且不可约，唯一平稳分布为
> $$
> \pi=\left(\frac{\mu}{\lambda+\mu},\frac{\lambda}{\lambda+\mu}\right),
> $$
> 两个分量都严格为正。

> [!question]- 自检
> 一条有限不可约 CTMC 的 embedded jump chain 有周期 2。能否据此否定平稳分布的存在或唯一性？
>
> **答案：** 不能。有限不可约已经保证平稳分布存在、唯一且全支持；跳链周期不参与这个结论。

## 来源与核验

- [MIT OCW 6.436J, Lecture 24](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/087af3cedbc9def5b156c5e1665ac79c_MIT6_436JF18_lec24.pdf#page=7)：核对有限不可约 CTMC 的平稳存在、唯一性与正性。
- [[有限不可约链稳态]]：交叉核对有限不可约 Markov 链的类内稳态结构，并区分连续时间的收敛条件。
