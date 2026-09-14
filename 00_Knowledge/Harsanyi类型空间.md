---
aliases:
  - "Harsanyi 类型空间用类型及其条件信念编码收益不确定性、私人信息与信念层级"
  - "Harsanyi type space"
  - "Harsanyi type-space representation"
student_os: knowledge-atom
atom_id: GT-HAR-001
atom_set: bayesian-games
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[贝叶斯博弈]]"
related:
  - "[[博弈类型]]"
  - "[[事中信念]]"
  - "[[共同先验]]"
  - "[[Harsanyi转换]]"
  - "[[实际与假想类型剖面]]"
  - "[[类型代理表示]]"
part_of:
  - "[[贝叶斯博弈.canvas]]"
---

# Harsanyi 类型空间用类型及其条件信念编码收益不确定性、私人信息与信念层级
<!-- bilingual-en:start -->
*A Harsanyi type space uses types and their conditional beliefs to encode payoff uncertainty, private information, and hierarchies of belief*
<!-- bilingual-en:end -->

**Harsanyi 类型空间**把玩家不知道的收益参数、各自观察到的私人信息以及“我怎样相信你、你又怎样相信我”的高阶信念，统一编码进类型及其条件信念。
<!-- bilingual-en:start -->
A **Harsanyi type space** encodes unknown payoff parameters, players' private information, and higher-order beliefs about one another through types and their conditional beliefs.
<!-- bilingual-en:end -->

## 从未知收益到类型空间

<!-- bilingual-en:start -->
*From Unknown Payoffs to a Type Space*
<!-- bilingual-en:end -->

设 $\theta\in\Theta$ 表示收益相关的物理状态，$t_i\in T_i$ 表示玩家 $i$ 的[[博弈类型|类型]]。类型 $t_i$ 不只对应一条私人信号；它还带有 belief

<!-- bilingual-en:start -->
Let $\theta\in\Theta$ denote the payoff-relevant physical state and $t_i\in T_i$ player $i$'s [[博弈类型|type]]. Type $t_i$ does not merely correspond to a private signal; it also carries a belief
<!-- bilingual-en:end -->

$$
p_i(\cdot\mid t_i)\in\Delta(\Theta\times T_{-i}),
$$

所以它同时给出玩家对 $\theta$ 和他人类型的判断。由他人类型又可推出他人的一阶 beliefs，再推出玩家对他人 beliefs 的二阶 belief，如此形成 belief hierarchy。类型空间把这条无限层级压进有限或可处理的数学对象中。
<!-- bilingual-en:start -->
Thus a type determines beliefs about both $\theta$ and other players' types. Those other types encode their own first-order beliefs, which in turn determine beliefs about beliefs and higher levels of the hierarchy. The type space packages this hierarchy into a tractable mathematical object.
<!-- bilingual-en:end -->

类型空间列出的剖面不都要在现实中实际发生；[[实际与假想类型剖面]]说明它们怎样共同表达一条信念层级。
<!-- bilingual-en:start -->
Not every profile in a type space has to occur in the actual situation. [[实际与假想类型剖面|Actual and hypothetical type profiles]] explains how the remaining profiles support the hierarchy of beliefs.
<!-- bilingual-en:end -->

## 与两种求解表示的关系

<!-- bilingual-en:start -->
*Relation to Two Solution Representations*
<!-- bilingual-en:end -->

有[[共同先验]]时，类型空间可以进一步写成[[Harsanyi转换|事前 Nature 抽取]]；有限类型时，也可构造[[类型代理表示|事中类型代理博弈]]。前者把共享概率画成树，后者把逐类型最优化写成普通 Nash 问题；两者都不是 Harsanyi 类型空间定义本身的一部分。
<!-- bilingual-en:start -->
With a [[共同先验|common prior]], the type space can be drawn as an [[Harsanyi转换|ex-ante Nature draw]]. With finite type sets, it can also be solved through an [[类型代理表示|interim-agent game]]. The first draws a shared distribution as a tree and the second turns type-by-type optimization into an ordinary Nash problem; neither representation is part of the definition of a Harsanyi type space.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一个类型只记录玩家自己的私人信号，却没有说明他怎样相信未知参数和其他玩家的类型。为什么这还不是完整的 Harsanyi 类型？
>
> **答案：** 因为类型还必须确定相应的条件信念。对他人类型的信念会递归带出对他人信念的判断；漏掉这一部分就没有编码完整的信念层级。
> <!-- bilingual-en:start -->
> **Self-check.** A type records only the player's own private signal but does not specify their beliefs about the unknown parameters or the other players' types. Why is this not yet a complete Harsanyi type?
>
> **Answer:** A type must also determine the corresponding conditional belief. Beliefs about the other players' types recursively induce beliefs about their beliefs; omitting this component leaves the hierarchy of beliefs incompletely encoded.
> <!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and Verification*
<!-- bilingual-en:end -->

- [MIT OCW 14.126, *Game Theory, Lecture Notes*, Chapter 3 §3.1 “Meaning of a Type Space”，正文 pp.69–71](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_yildiz-lecture-notes.pdf)：核对 Harsanyi 类型空间对一阶、二阶直至更高阶 beliefs 的表示，以及实际类型组合与假想组合的区别。
