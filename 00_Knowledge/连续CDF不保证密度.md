---
aliases:
  - "连续的累积分布函数不一定来自概率密度函数"
  - A continuous CDF need not have a density
student_os: knowledge-atom
atom_id: PROB-RV-018
atom_set: random-variables-distributions-moments
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[累积分布函数]]"
  - "[[概率密度函数]]"
related:
  - "[[PMF不等于密度]]"
part_of:
  - "[[随机变量、分布与矩.canvas]]"
---

# 连续的累积分布函数不一定来自概率密度函数
<!-- bilingual-en:start -->
*A continuous cumulative distribution function need not arise from a probability density function*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> CDF 连续只说明它没有跳跃，因此每个单点的概率为 0。存在密度要求更强的绝对连续性：必须有可积 $f$ 使
> $$F(x)=\int_{-\infty}^{x}f(t)\,dt.$$
> <!-- bilingual-en:start -->
> Continuity of a CDF rules out jumps and point masses. Having a density requires the stronger property of absolute continuity.
> <!-- bilingual-en:end -->

Cantor 分布给出标准反例：它的 CDF 连续，却把全部概率集中在 Lebesgue 测度为 0 的 Cantor 集上，因而不存在通常意义下对 Lebesgue 测度的密度。

所以“离散或连续”还不够精确。一个分布可以是离散的、绝对连续的、混合的，也可以含奇异连续成分；[[累积分布函数|CDF]] 能统一表示这些情形。

> [!question]- 自检
> 已知 $F$ 处处连续，能否直接令 $f=F'$ 并称它为密度？
>
> **答案：** 不能。还要确认 $F$ 绝对连续，并能由 $F'$ 的积分恢复。

## 来源与核验

- [MIT 6.436J, Lecture 4](https://live.ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/9797310bed4c7f5b5d40d007783eec8d_MIT6_436JF18_lec04.pdf#page=14)：用 Cantor 分布核对连续 CDF 不保证密度。
- [MIT 18.175, Lecture 3](https://www.ocw.mit.edu/courses/18-175-theory-of-probability-spring-2014/827332a721e47b09cb0cbcfb4b60ecf6_MIT18_175S14_Lecture3.pdf)：核对 CDF、概率测度与绝对连续性的区分。
<!-- bilingual-en:start -->
- MIT 6.436J and MIT 18.175 were checked for the singular-continuous boundary and the Cantor example.
<!-- bilingual-en:end -->
