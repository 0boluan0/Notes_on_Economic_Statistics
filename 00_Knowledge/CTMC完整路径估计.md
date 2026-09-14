---
aliases:
  - "完整路径下生成率由跳转计数与状态暴露时间共同估计"
  - CTMC complete-path MLE
  - Complete-path generator rate estimation
  - Complete-path transition counts and exposure times
  - CTMC 完整路径生成率估计
student_os: knowledge-atom
atom_id: PROB-CTMC-024
atom_set: continuous-time-markov-chains
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[生成矩阵约束]]"
  - "[[生成矩阵分解]]"
  - "[[出口率]]"
related:
  - "[[嵌入跳链]]"
  - "[[Markov充分状态]]"
leads_to:
  - "[[CTMC共同率估计]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

# 完整路径下生成率由跳转计数与状态暴露时间共同估计
<!-- bilingual-en:start -->
*With a fully observed path, generator rates are estimated from jump counts and state exposure times together*
<!-- bilingual-en:end -->

> [!summary] 充分统计量有两部分
> 对生成矩阵 $Q$ 在窗内固定的时间齐次 CTMC，设完整观察窗为 $[0,T]$。模型须非爆炸，或这段实际观察满足 $T<\zeta$；观察记录必须包含窗内每次真实跳时和所处状态。令 $N_{ij}$ 为 $i\to j$ 的跳跃次数，$S_i$ 为过程在状态 $i$ 的总暴露时间，则 likelihood 中与第 $i$ 行相关的因子为
> $$
> \exp(-q_iS_i)\prod_{j\ne i}q_{ij}^{N_{ij}},
> $$
> 因而当 $S_i>0$ 时
> $$
> \widehat q_{ij}=\frac{N_{ij}}{S_i},
> \qquad \widehat q_{ii}=-\sum_{j\ne i}\widehat q_{ij}.
> $$
> <!-- bilingual-en:start -->
> Complete-path estimation needs both how many transitions occurred and how long the process was at risk of leaving each state.
> <!-- bilingual-en:end -->

只用 $N_{ij}/\sum_{k\ne i}N_{ik}$ 至多估计 embedded jump probability $r_{ij}$，会丢掉出口率 $q_i$。若 $S_i=0$，数据没有在状态 $i$ 暴露，不能从该路径估计这一行；零次观察跳跃也不等于真实速率严格为零。

这里的 $S_i$ 必须包含观察终点前最后一段尚未以跳跃结束的停留时间；它是右删失的暴露时间，不能因为“没有看到离开”而删掉。上式还把各个非对角率 $q_{ij}$ 当作可分别估计的自由参数；若多条边共享同一参数，应转到[[CTMC共同率估计]]，先施加约束再估计。

若只在离散时点看到状态，中间可能漏掉一次或多次跳跃，此时并没有 $N_{ij}$ 与 $S_i$ 的完整记录。likelihood 应使用相邻观测间的转移概率 $P(\Delta)=e^{\Delta Q}$，不能直接套“计数除以暴露时间”。

> [!example] 相同跳数，不同速率
> 两条记录都看到 10 次 $i\to j$；第一条在 $i$ 暴露 2 小时，第二条暴露 20 小时。估计率分别为 5/小时与 0.5/小时，说明跳数本身没有时间尺度。

> [!question]- 自检
> 每小时抽样一次，共看到五次相邻样本从 $i$ 变成 $j$。能否把 5 除以在 $i$ 的样本小时数作为完整路径 MLE？
>
> **答案：** 不能。抽样间隔内可能发生未观察的多次跳跃；这属于离散观察 likelihood，而不是完整路径计数。

## 来源与核验

- [Hobolth et al., complete CTMC sufficient statistics](https://pmc.ncbi.nlm.nih.gov/articles/PMC3329461/)：核对完整路径 likelihood、状态暴露时间 $S_i$ 与跳转计数 $N_{ij}$。
- [Oxford Statistical Lifetime-Models notes, Lecture 10, pp. 63–64](https://www.stats.ox.ac.uk/~winkel/bs3b10_14.pdf)：核对 $\widehat q_{ij}=N_{ij}/S_i$ 与完整路径的计数、暴露时间 likelihood。
- [Efficient maximum likelihood parameterization of continuous-time Markov processes](https://pmc.ncbi.nlm.nih.gov/articles/PMC4514821/)：核对离散时点观察应通过 $e^{\Delta Q}$ 建 likelihood，而非把相邻样本转移当完整跳数。
