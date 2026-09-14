---
aliases:
  - "同成功概率且相互独立的 Bernoulli 变量之和服从二项分布"
  - Sum of Bernoulli variables and the Binomial law
  - Bernoulli 和 Binomial
student_os: knowledge-atom
atom_id: PROB-RV-008
atom_set: random-variables-distributions-moments
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Bernoulli分布]]"
  - "[[二项分布]]"
  - "[[相互独立]]"
related:
  - "[[期望线性性]]"
  - "[[和的方差协方差项]]"
  - "[[指示变量与随机计数.canvas|指示变量与随机计数]]"
part_of:
  - "[[随机变量、分布与矩.canvas]]"
---

# 同成功概率且相互独立的 Bernoulli 变量之和服从二项分布
<!-- bilingual-en:start -->
*A sum of mutually independent Bernoulli variables with a common success probability is Binomial*
<!-- bilingual-en:end -->

> [!summary] 原子定理与边界
> 若 $X_1,\ldots,X_n$ 相互独立且每个 $X_i\sim\operatorname{Bernoulli}(p)$，则
> $$S=\sum_{i=1}^nX_i\sim\operatorname{Binomial}(n,p).$$
> 共同的 $p$ 与相互独立构成这条标准推导的充分条件；若缺少任一条件，不能再仅凭各变量的边际分布推出二项结论。
> <!-- bilingual-en:start -->
> A sum of mutually independent Bernoulli$(p)$ variables is Binomial$(n,p)$. Without the common-$p$ or independence condition, the marginal laws alone no longer imply a Binomial sum.
> <!-- bilingual-en:end -->

若只知道每个边际成功率都是 $p$，[[期望线性性]]仍给出 $E[S]=np$；但方差要加入协方差项，整个分布也未必二项。有限总体不放回抽样通常使 indicators 负相关，计数常服从超几何分布。

若 $X_i$ 相互独立但成功概率分别为 $p_i$，其和服从 Poisson-binomial 分布而非单一 $p$ 的二项分布。此时 $E[S]=\sum_ip_i$，$\operatorname{Var}(S)=\sum_ip_i(1-p_i)$。

> [!question]- 自检
> 只知道 $n$ 个 0/1 变量的边际成功率都为 $p$，能确定什么？
>
> **答案：** 能确定和的期望是 $np$；不能据此确定方差或二项分布。

## 来源与核验

- [MIT 18.05 Probability Reading](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf) Class 4–7：核对 Bernoulli 和、二项 PMF、期望线性性与方差条件。
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf|MIT Mathematics for Computer Science]] 第 19.3 与 20.3 节：核对独立计数和方差相加条件。
<!-- bilingual-en:start -->
- MIT 18.05 and MIT Mathematics for Computer Science were checked for the common-$p$, independence, mean, and variance boundaries.
<!-- bilingual-en:end -->
