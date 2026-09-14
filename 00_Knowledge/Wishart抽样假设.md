---
aliases:
  - "iid 多元正态保证样本协方差的精确 Wishart 律而相同协方差本身不够"
  - IID multivariate normal sampling guarantees the exact Wishart law while a common covariance alone is insufficient
  - Boundary of the Wishart sample-covariance law
  - Wishart 精确抽样律的假设边界
student_os: knowledge-atom
atom_id: STAT-WISH-006
atom_set: wishart-sample-covariance
atom_type: assumption-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[样本协方差Wishart律]]"
part_of:
  - "[[Wishart 分布与样本协方差推断.canvas]]"
related:
  - "[[正态均值协方差独立]]"
---

# iid 多元正态保证样本协方差的精确 Wishart 律而相同协方差本身不够
<!-- bilingual-en:start -->
*IID multivariate-normal sampling guarantees the exact Wishart law for sample covariance; a common covariance alone is insufficient*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 若 $X_i\overset{iid}{\sim}N_p(\mu,\Sigma)$，则
> $$(n-1)S\sim W_p(\Sigma,n-1)$$
> 是精确有限样本结论。这是标准定理的一组充分条件；只有“各观察具有相同均值与协方差”并不能推出 Wishart 分布，也不能推出 $\bar X$ 与 $S$ 独立。
> <!-- bilingual-en:start -->
> IID multivariate-normal sampling gives the standard exact finite-sample Wishart theorem. Equal first and second moments alone do not determine that law or mean-covariance independence.
> <!-- bilingual-en:end -->

原因在于协方差只固定二阶矩，而 $S$ 的完整分布还会受到更高阶矩与观察之间联合结构的影响：

- **重尾或偏态：** 极端向量出现得更频繁或方向不对称，外积和的尾部与 Wishart 不同；
- **相关观察：** 即使 iid 正态样本的原始中心化残差也彼此受“和为零”约束；Wishart 证明依靠的是正交变换后得到 $n-1$ 个独立 Gaussian 对比。时间或群组依赖一般会破坏这组独立同尺度对比，因此不能继续沿用自由度 $n-1$；
- **异方差或结构变化：** 各观察的协方差不相同，外积和不再由同一个尺度矩阵生成；
- **仅有不相关：** 即便均值部分与残差部分协方差为零，非 Gaussian 结构也通常不能把它升级为独立。

这不意味着非正态数据不能估计协方差。$S$ 在适当条件下仍可能无偏或一致，也可以使用渐近近似、bootstrap、稳健协方差估计或针对依赖结构的模型；但这些是新的推断依据，不能倒过来证明有限样本分布仍是 Wishart。

一个一维反例已经足够。令 $X_i$ iid 等概率取 $-1$ 或 $1$，则 $E[X_i]=0$、$\operatorname{Var}(X_i)=1$，与标准正态有相同前两阶矩。取 $n=2$ 时，无偏样本方差 $S^2$ 以各 $1/2$ 的概率等于 $0$ 或 $2$；而正态样本应满足 $(n-1)S^2\sim\chi^2_1$，后者是连续分布。相同均值与方差显然没有决定精确抽样律。

实务上要分开问两件事：

1. **估计层面：** $S$ 是否仍是目标协方差的合理估计？
2. **校准层面：** 依赖 Wishart、$S^{-1}$ 或精确尾概率的检验和区间是否仍有依据？

前者成立不保证后者成立。

> [!question]- 自检
> 若一组 iid 数据均值为 $\mu$、协方差为 $\Sigma$，且样本协方差无偏，为什么仍不能直接套用 Wishart 临界值？
>
> **答案：** 无偏只规定 $E[S]=\Sigma$；Wishart 临界值依赖 $S$ 的完整有限样本分布，而后者需要多元正态等更强条件。

## 来源与核验

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.6. Wishart 分布|多元统计课程 §§1.5.2–1.6.2]]：核对 Wishart 抽样律与均值—协方差独立性所使用的 iid 多元正态前提。
- [Penn State STAT 505, Lesson 4](https://online.stat.psu.edu/stat505/Lesson04)：核对多元正态抽样框架；偏离该框架时需另行建立校准方法。
- [Purdue Statistics, *Graduate Probability*, Theorem 1.91](https://www.stat.purdue.edu/~dasgupta/gradprob.pdf#page=171)：核对 iid 多元正态样本给出的标准精确 Wishart 抽样律与均值—协方差独立性。
