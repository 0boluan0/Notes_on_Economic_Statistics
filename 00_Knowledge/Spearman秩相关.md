---
aliases:
  - "Spearman 秩相关是概率尺度秩的 Pearson 相关，度量单调一致性"
  - "Spearman rank correlation"
  - "Spearman相关"
student_os: knowledge-atom
atom_id: RM-DEP-005
atom_set: dependence-and-copulas
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[相关系数]]"
  - "[[累积分布函数]]"
leads_to:
  - "[[相关度量比较]]"
related:
  - "[[Kendall秩相关]]"
  - "[[Copula]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# Spearman 秩相关是概率尺度秩的 Pearson 相关，度量单调一致性
<!-- bilingual-en:start -->
*Spearman rank correlation is Pearson correlation on probability-scale ranks and measures monotone concordance*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 若 $X,Y$ 的边际分布连续，令 $U=F_X(X)$、$V=F_Y(Y)$，则 Spearman 秩相关定义为
> $$\rho_S=\operatorname{Corr}(U,V)=12\operatorname E[UV]-3.$$
> 它比较两个变量的相对排序，而不是原数值之间是否近似一条直线。
> <!-- bilingual-en:start -->
> With continuous margins, Spearman's rho is the Pearson correlation of $U=F_X(X)$ and $V=F_Y(Y)$. It measures agreement in relative ordering rather than linearity in the original values.
> <!-- bilingual-en:end -->

## 样本版本与直觉

没有 ties 时，样本 Spearman 相关就是两个秩序列的 Pearson 相关。若 $d_i$ 是第 $i$ 个观测在两列中的秩之差，则

$$
\widehat\rho_S=1-\frac{6\sum_{i=1}^n d_i^2}{n(n^2-1)}.
$$

严格单调递增变换不会改变秩，因此不会改变 $\rho_S$。例如 $Y=e^X$ 且 $X$ 连续非退化时，$\rho_S=1$，即使原数值关系不是直线。

## 边界

存在 ties 时，必须说明平均秩等处理规则；上面的秩差简式也不再原样适用。Spearman 相关能识别单调非线性关系，但仍只是一个数，非单调依赖可能被它漏掉，见 [[相关度量比较]]。

> [!question]- 自检
> 把 $X$ 与 $Y$ 分别做严格递增但高度非线性的变换，会改变总体 Spearman 相关吗？
>
> **答案：** 连续边际下不会，因为两列的相对次序没有改变。

## 来源与核验

- Charles Spearman (1904), [“The Proof and Measurement of Association between Two Things”](https://doi.org/10.2307/1412159)：核对样本秩相关构造。
- Roger B. Nelsen, *An Introduction to Copulas*, 2nd ed., §5.1：[出版社页面](https://link.springer.com/book/10.1007/0-387-28678-0)；核对连续边际下的概率尺度表示。
