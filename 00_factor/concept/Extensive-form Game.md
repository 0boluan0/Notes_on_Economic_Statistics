---
aliases:
- Extensive-form Game
- Extensive Game
- 扩展式博弈
- 扩展性博弈
tags:
- concept
- game-theory
---
# Extensive-form Game

## 一句话记忆

扩展式博弈是用博弈树记录行动顺序、历史路径和信息结构的博弈。

## 它是什么

扩展式博弈不仅说明谁有哪些策略和收益，还说明：

- 谁先行动、谁后行动。
- 玩家在行动时知道什么。
- 哪些历史是终端历史。
- 自然节点以什么概率发生。

常用形式：

$$
G=\langle N,H,P,f,\{\mathcal I_i\}_{i\in N},\{u_i\}_{i\in N}\rangle.
$$

## 解决什么判断

- 题目是否需要画博弈树。
- 是否能用逆向归纳。
- 某个威胁是否可信。
- 信息集是否导致不完美信息问题。

## 最小例子

进入威慑中，进入者先选择 Enter 或 Stay Out；若 Enter，在位者再选择 Fight 或 Truce。这需要扩展式表达。

## 易混点

- 扩展式博弈中的 strategy 是完整计划，不是单个节点的 action。
- payoff 只定义在终端历史上。
- 信息集决定玩家是否知道自己处于哪个节点。

## 来自课程位置

- [[02_初步认识博弈#1.3. 扩展式表达]]
- [[06_扩展性博弈#7. 扩展式博弈的一般形式]]

## 关联卡片

- [[Extensive Form to Strategic Form]]
- [[Backward Induction]]
- [[Backward Induction Procedure]]
- [[Information Set]]
- [[Subgame]]
- [[Subgame Perfect Nash Equilibrium]]
