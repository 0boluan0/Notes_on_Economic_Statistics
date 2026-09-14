---
aliases:
  - "Copula 是边际均为 Uniform(0,1) 的联合分布函数"
  - "Copula definition"
  - "Copula函数"
student_os: knowledge-atom
atom_id: RM-DEP-007
atom_set: dependence-and-copulas
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[累积分布函数]]"
leads_to:
  - "[[Copula分解]]"
  - "[[Gaussian Copula]]"
  - "[[t Copula]]"
  - "[[尾部依赖]]"
related:
  - "[[相关度量比较]]"
part_of:
  - "[[依赖建模.canvas]]"
---

# Copula 是边际均为 Uniform(0,1) 的联合分布函数
<!-- bilingual-en:start -->
*A copula is a joint distribution function whose margins are Uniform$(0,1)$*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> $d$ 维 copula 是定义在 $[0,1]^d$ 上的联合累积分布函数 $C$，并且每一个一维边际都为 Uniform$(0,1)$。等价地，固定第 $i$ 个坐标为 $u_i$、其余坐标为 1 时，
> $$C(1,\ldots,1,u_i,1,\ldots,1)=u_i.$$
> <!-- bilingual-en:start -->
> A $d$-dimensional copula is a joint CDF on $[0,1]^d$ whose univariate margins are uniform.
> <!-- bilingual-en:end -->

## 直觉

把每个变量都转换到 $0$ 到 $1$ 的边际概率尺度以后，原来的量纲和单体分布形状被拿掉，剩下的是这些概率位置怎样共同出现。Copula 正是这部分联合结构。

最简单的例子是乘积 copula

$$
\Pi(u_1,\ldots,u_d)=\prod_{i=1}^d u_i,
$$

它表示各个均匀坐标相互独立。一般 copula 不等于 $\Pi$。

## 与联合分布的关系

Copula 本身不替代边际分布。[[Copula分解|Sklar 定理]]说明怎样把一个 copula 与各边际组合成原变量的联合分布，也说明何时能从联合分布唯一找回 copula。

> [!question]- 自检
> 若一个二维函数满足 $C(u,1)=u$、$C(1,v)=v$，是否已经足以保证它是 copula？
>
> **答案：** 还不够。它还必须是一个合法的二维联合分布函数，例如对每个矩形给出非负概率。

## 来源与核验

- Roger B. Nelsen, *An Introduction to Copulas*, 2nd ed., Definition 2.2.2：[出版社页面](https://link.springer.com/book/10.1007/0-387-28678-0)；核对 copula 定义与边际条件。
- Abe Sklar (1959), [原文重排与英译本](https://doi.org/10.2139/ssrn.4198458)：核对该对象在联合分布分解中的角色。
