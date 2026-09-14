---
aliases:
  - "PMF 在单点给概率而密度只有积分后才给概率"
  - PMF is not a density
  - 概率质量与概率密度
  - PMF 与密度的概率边界
student_os: knowledge-atom
atom_id: PROB-RV-003
atom_set: random-variables-distributions-moments
atom_type: distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[概率质量函数]]"
  - "[[概率密度函数]]"
related:
  - "[[累积分布函数]]"
part_of:
  - "[[随机变量、分布与矩.canvas]]"
---

# PMF 在单点给概率而密度只有积分后才给概率
<!-- bilingual-en:start -->
*A PMF gives point probabilities, whereas a density gives probabilities only after integration*
<!-- bilingual-en:end -->

> [!summary] 区分什么
> [[概率质量函数|PMF]] 的值就是 $P(X=x)$，所以每个值都在 $[0,1]$ 内。[[概率密度函数|密度]]的高度是单位长度上的概率强度，可以大于 1；只有区间下的面积才是概率。
> <!-- bilingual-en:start -->
> A PMF value is the point probability $P(X=x)$ and lies in $[0,1]$. A density is probability intensity per unit of $X$; it may exceed one, and only integrated area is probability.
> <!-- bilingual-en:end -->

两者都必须归一化，但使用不同运算：离散质量求和，连续密度积分。对绝对连续变量，$P(X=x)=0$；这不表示该数值逻辑上不可能出现，而是任意预先指定的单点没有正质量。

<!-- bilingual-en:start -->
Both representations are normalised, but discrete masses are summed while continuous densities are integrated. A specified point has probability zero under an absolutely continuous law; this is not a claim of logical impossibility.
<!-- bilingual-en:end -->

混合分布可同时含点质量与连续部分；还有分布既没有 PMF，也没有密度。遇到这些情形应回到 [[累积分布函数|CDF]]，不要强迫所有分布二选一。

> [!question]- 自检
> 为什么“密度为 0.2”和“该点概率为 0.2”不是同一句话？
>
> **答案：** 密度必须乘上一段宽度并积分才能产生概率；PMF 才直接给单点质量。

## 来源与核验

- [MIT 18.05 Probability Reading](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf) Class 4–5：核对 PMF 的点质量与密度的面积解释。
- [MIT 6.436J, Lecture 4](https://live.ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/9797310bed4c7f5b5d40d007783eec8d_MIT6_436JF18_lec04.pdf)：核对 PMF、密度并不穷尽所有分布。
<!-- bilingual-en:start -->
- MIT 18.05 was checked for the point-mass versus area distinction.
- MIT 6.436J was checked for distributions outside the PMF-versus-density dichotomy.
<!-- bilingual-en:end -->
