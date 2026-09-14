---
aliases:
  - "Bernoulli 分布描述一次成功概率为 p 的 0 或 1 试验"
  - Bernoulli distribution
  - 伯努利分布
student_os: knowledge-atom
atom_id: PROB-RV-016
atom_set: random-variables-distributions-moments
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[概率质量函数]]"
related:
  - "[[事件指示变量]]"
leads_to:
  - "[[二项分布]]"
  - "[[Bernoulli和的二项分布]]"
part_of:
  - "[[随机变量、分布与矩.canvas]]"
---

# Bernoulli 分布描述一次成功概率为 p 的 0 或 1 试验
<!-- bilingual-en:start -->
*A Bernoulli distribution describes one zero-one trial with success probability $p$*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 若 $0\le p\le1$ 且
> $$P(X=1)=p,\qquad P(X=0)=1-p,$$
> 就写作 $X\sim\operatorname{Bernoulli}(p)$。数值 1 通常编码“成功”，0 编码“失败”。
> <!-- bilingual-en:start -->
> A Bernoulli$(p)$ variable takes value one with probability $p$ and zero with probability $1-p$.
> <!-- bilingual-en:end -->

因为 $X^2=X$，
$$E[X]=p,\qquad \operatorname{Var}(X)=p(1-p).$$
任意事件 $A$ 的指示变量 $I_A$ 都服从 $\operatorname{Bernoulli}(P(A))$；这里的“成功”只是编码，不带价值判断。

Bernoulli 分布只描述一次 0/1 结果。多次试验的联合关系还需另给：每次边际上都是 Bernoulli，并不自动表示它们独立或成功率相同。

> [!question]- 自检
> $X\sim\operatorname{Bernoulli}(0.2)$ 时，$X=0$ 的概率是多少？
>
> **答案：** $0.8$。

## 来源与核验

- [MIT 18.05 Probability Reading](https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_probability.pdf) Class 4–5：核对 Bernoulli PMF、均值与方差。
<!-- bilingual-en:start -->
- MIT 18.05 was checked for the Bernoulli PMF, mean, and variance.
<!-- bilingual-en:end -->
