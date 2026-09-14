---
aliases:
  - "Kendall 秩相关是同序对概率减去逆序对概率"
  - "Kendall rank correlation"
  - "Kendall tau"
student_os: knowledge-atom
atom_id: RM-DEP-006
atom_set: dependence-and-copulas
atom_type: definition
status: source-checked
mastery_state: unassessed
leads_to:
  - "[[相关度量比较]]"
related:
  - "[[Spearman秩相关]]"
  - "[[Copula]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# Kendall 秩相关是同序对概率减去逆序对概率
<!-- bilingual-en:start -->
*Kendall rank correlation is the probability of concordance minus the probability of discordance*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 取 $(X,Y)$ 的独立同分布副本 $(X',Y')$。连续情形下，Kendall 的 $\tau$ 定义为
> $$\tau=P((X-X')(Y-Y')>0)-P((X-X')(Y-Y')<0).$$
> 第一项是两个变量排序一致的概率，第二项是排序相反的概率。
> <!-- bilingual-en:start -->
> With an independent copy $(X',Y')$, Kendall's tau is the probability of a concordant pair minus the probability of a discordant pair.
> <!-- bilingual-en:end -->

## 样本版本与直觉

没有 ties 时，在全部 $\binom n2$ 对观测中记同序对数为 $C$、逆序对数为 $D$，则

$$
\widehat\tau=\frac{C-D}{\binom n2}.
$$

严格单调递增变换保留每一对观测的顺序，因此不改变 $\tau$。$\tau=1$ 表示任意两条记录都同序；$\tau=-1$ 表示任意两条记录都逆序。

## ties 的边界

若某一对在 $X$ 或 $Y$ 上同值，就既不是严格同序也不是严格逆序。此时 $\tau_a$、$\tau_b$、$\tau_c$ 的分母修正不同，报告结果必须注明版本。不能把无 ties 的简式直接当作所有软件输出的定义。

> [!question]- 自检
> 若 $Y=e^X$ 且 $X$ 连续非退化，Kendall 的 $\tau$ 是多少？
>
> **答案：** $1$。严格递增函数保留每一对观测的顺序。

## 来源与核验

- M. G. Kendall (1938), [“A New Measure of Rank Correlation”](https://doi.org/10.1093/biomet/30.1-2.81)：核对同序对、逆序对与样本统计量。
- Roger B. Nelsen, *An Introduction to Copulas*, 2nd ed., §5.1：[出版社页面](https://link.springer.com/book/10.1007/0-387-28678-0)；核对总体概率表示与单调不变性。
