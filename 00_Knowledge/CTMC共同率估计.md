---
aliases:
  - "完整路径下，共同参数约束的 CTMC 生成率应汇总计数与暴露时间后估计"
  - CTMC shared-rate estimation
  - Pooled CTMC rate MLE
  - 共同生成率估计
student_os: knowledge-atom
atom_id: PROB-CTMC-031
atom_set: continuous-time-markov-chains
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[CTMC完整路径估计]]"
  - "[[生成矩阵约束]]"
related:
  - "[[Poisson是常速纯生链]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 完整路径下，共同参数约束的 CTMC 生成率应汇总计数与暴露时间后估计
<!-- bilingual-en:start -->
*CTMC rates constrained by a shared parameter must be estimated from pooled counts and exposure times*
<!-- bilingual-en:end -->

> [!summary] 先施加约束，再求 MLE
> 在完整路径模型中，设一组有向边 $E$ 满足
> $$
> q_{ij}=c_{ij}\theta,\qquad (i,j)\in E,
> $$
> 其中 $c_{ij}>0$ 已知，$\theta\ge0$ 是共同未知参数。记
> $$
> N_E:=\sum_{(i,j)\in E}N_{ij},
> \qquad
> D_E:=\sum_{(i,j)\in E}c_{ij}S_i.
> $$
> 当 $D_E>0$ 时，共同参数的最大似然估计为
> $$
> \widehat\theta=\frac{N_E}{D_E}.
> $$
> <!-- bilingual-en:start -->
> Under a shared-rate constraint, the MLE is the total constrained-edge count divided by the corresponding weighted exposure.
> <!-- bilingual-en:end -->

这个公式来自先把 $q_{ij}=c_{ij}\theta$ 代入完整路径 likelihood：与 $\theta$ 有关的部分正比于
$$
\theta^{N_E}\exp(-\theta D_E).
$$
因此不能先逐边算 $N_{ij}/S_i$，再对这些比率做简单平均。若 $D_E=0$，路径没有为这个参数提供暴露，$\theta$ 无法由该记录识别；若 $D_E>0$ 而 $N_E=0$，允许参数边界 $\theta=0$ 时 MLE 为 0。

系数为 0 的边是模型规定的结构零，不应放入 $E$。如果完整路径却在结构零边上观察到正的跳转计数，问题不是“怎样估计 $\theta$”，而是当前参数化已与数据矛盾。

> [!example] Poisson 的 $N(T)/T$
> 对[[Poisson是常速纯生链]]，所有 $n\to n+1$ 边共享 $q_{n,n+1}=\lambda$，所以 $c_{n,n+1}=1$。完整观察 $[0,T]$ 时，总边计数是 $N(T)$，各已访问状态的暴露时间之和是 $T$，故
> $$
> \widehat\lambda=\frac{N(T)}{T}.
> $$

> [!question]- 自检
> 三条边共享同一参数 $\theta$。能否分别估计三次后取算术平均？
>
> **答案：** 一般不能。不同边的暴露时间不同；应先汇总为 $N_E/D_E$，让每条边按其加权暴露贡献信息。

## 来源与核验

- [Oxford Statistical Lifetime-Models notes, Lecture 10, pp. 63–64](https://www.stats.ox.ac.uk/~winkel/bs3b10_14.pdf)：核对完整路径计数与暴露时间的 pooled likelihood 和共享率估计。
- [Hobolth et al., complete CTMC sufficient statistics](https://pmc.ncbi.nlm.nih.gov/articles/PMC3329461/)：核对完整 CTMC 路径的充分统计量为跳转计数与状态暴露时间。
