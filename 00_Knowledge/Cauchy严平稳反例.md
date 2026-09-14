---
aliases:
  - "独立同分布 Cauchy 序列严平稳但不是宽平稳"
  - IID Cauchy stationary counterexample
student_os: knowledge-atom
atom_id: TS-STAT-004
atom_set: stationarity-ergodicity-spectrum
atom_type: counterexample
status: source-checked
mastery_state: unassessed
requires:
  - "[[严平稳定义]]"
  - "[[宽平稳定义]]"
related:
  - "[[严平稳与宽平稳]]"
part_of:
  - "[[平稳性、遍历性与谱.canvas]]"
---

# 独立同分布 Cauchy 序列严平稳但不是宽平稳
<!-- bilingual-en:start -->
*An IID Cauchy sequence is strictly stationary but not wide-sense stationary*
<!-- bilingual-en:end -->

> [!summary] 原子反例
> 令 $X_t$ 独立同分布且都服从标准 Cauchy 分布。任意有限向量的联合分布都是同一边际分布的乘积，所以共同平移时间指标不会改变它：过程严平稳。但 Cauchy 变量没有有限均值或方差，宽平稳所需的二阶矩不存在，因此它不是宽平稳过程。
> <!-- bilingual-en:start -->
> An IID standard-Cauchy sequence is strictly stationary because every finite joint law is the same product law after a time shift. It is not WSS because the Cauchy distribution has no finite mean or variance.
> <!-- bilingual-en:end -->

这个例子排除一种常见偷步：不能先用分布平移不变得到“均值相同”，再默认这个均值存在。分布相同只保证各时点具有同样的矩存在性；它不会创造有限矩。
<!-- bilingual-en:start -->
The example blocks a common shortcut: identical shifted distributions do not create moments that fail to exist. They preserve moment existence, not finiteness that was never present.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一列 IID Cauchy 随机变量为什么能通过严平稳检验，却连宽平稳的候选都不是？
>
> **答案：** IID 保证任意有限维联合分布在平移后不变；但 Cauchy 没有有限二阶矩，宽平稳的方差和协方差因而无法定义。

## 来源与核验

- [MIT OCW 18.05, All Probability Reading](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf)：核对 Cauchy 分布是均值和方差不收敛的标准例子。
- [MIT OCW 6.450, Chapter 7, Definition 7.5.1](https://ocw.mit.edu/courses/6-450-principles-of-digital-communications-i-fall-2006/49163236e20779bae41639ff9dec1ac4_book_7.pdf#page=18)：核对严平稳的有限维分布定义。
<!-- bilingual-en:start -->
- MIT 18.05 was checked for the failure of finite Cauchy moments.
- MIT 6.450 was checked for the strict-stationarity definition used in the construction.
<!-- bilingual-en:end -->
