---
aliases:
  - "Sklar 定理把联合分布分成边际与 copula，连续边际时 copula 唯一"
  - "Sklar theorem separates marginals from the copula"
  - "Sklar定理"
student_os: knowledge-atom
atom_id: RM-DEP-002
atom_set: dependence-and-copulas
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[Copula]]"
  - "[[累积分布函数]]"
leads_to:
  - "[[Gaussian Copula]]"
  - "[[t Copula]]"
related:
  - "[[边际分布不定联合分布]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# Sklar 定理把联合分布分成边际与 copula，连续边际时 copula 唯一
<!-- bilingual-en:start -->
*Sklar's theorem separates a joint distribution into its margins and a copula; the copula is unique when the margins are continuous*
<!-- bilingual-en:end -->

> [!summary] 定理说了什么
> 设联合累积分布函数 $H$ 的边际为 $F_1,\ldots,F_d$。Sklar 定理保证存在 [[Copula]] $C$，使
> $$H(x_1,\ldots,x_d)=C\!\left(F_1(x_1),\ldots,F_d(x_d)\right).$$
> 它把“每个变量怎样分布”与“这些变量怎样一起变化”分开。
> <!-- bilingual-en:start -->
> If $H$ has margins $F_1,\ldots,F_d$, Sklar's theorem gives a copula $C$ such that $H(x)=C(F_1(x_1),\ldots,F_d(x_d))$. This separates marginal behaviour from dependence.
> <!-- bilingual-en:end -->

## 唯一性与逆命题

若所有 $F_i$ 连续，$C$ 在整个 $[0,1]^d$ 上唯一。若有边际不连续，$C$ 只在

$$
\operatorname{Ran}(F_1)\times\cdots\times\operatorname{Ran}(F_d)
$$

上唯一；向单位立方体其余部分的扩张一般不唯一。

逆向也成立：任取 copula $C$ 与一维分布函数 $F_i$，上面的复合式都会给出一个以 $F_i$ 为边际的联合分布。
<!-- bilingual-en:start -->
With continuous margins the copula is unique on the whole unit cube. With discontinuous margins it is unique only on the product of the marginal ranges. Conversely, combining any copula with univariate distribution functions produces a joint distribution with those margins.
<!-- bilingual-en:end -->

## 定理没有说什么

“能分解”不等于“相互独立”。独立只是使用乘积 copula

$$
\Pi(u_1,\ldots,u_d)=\prod_{i=1}^d u_i
$$

的特殊情形。一般 copula 会偏离 $\Pi$。连续边际下 $U_i=F_i(X_i)$ 恰为 Uniform$(0,1)$；离散边际下，未经随机化的 $F_i(X_i)$ 通常不均匀，且上述不唯一性必须保留。

若 $F_X(x)=0.8$、$F_Y(y)=0.25$，并选择独立 copula，则

$$P(X\le x,Y\le y)=\Pi(0.8,0.25)=0.20.$$

> [!question]- 自检
> 两个 Bernoulli 边际给定后，能否声称它们的 copula 在整个 $[0,1]^2$ 上唯一？
>
> **答案：** 不能。表示仍存在，但只在两个边际 CDF 的值域乘积上唯一。

## 来源与核验

- Abe Sklar (1959), “Fonctions de répartition à $n$ dimensions et leurs marges”；[原文重排与英译本](https://doi.org/10.2139/ssrn.4198458)：核对表示、连续边际下的唯一性与逆命题。
- 作者逐式复核日：2026-09-01；定理、离散边际边界与数值例子已按原文逐项核对。
