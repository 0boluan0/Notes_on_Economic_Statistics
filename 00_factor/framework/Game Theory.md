---
aliases:
- Game Theory
- Game Theory Basics
- 博弈论
- 博弈论基础
tags:
- framework
- game-theory
type: framework
---
# Game Theory

## 什么时候用

当一个人的最优选择依赖于别人怎么选，且别人也会反过来根据你的选择调整行为时，用博弈论框架。

## 为什么这样看

博弈论关心的不是单边最优，而是相互最优反应。结果取决于玩家如何预期对方、对方又如何预期自己。

## 题型识别

- 多个玩家同时选择策略：先写 [[strategic-form Games]]。
- 有行动顺序或威胁：先写 [[Extensive-form Game]]。
- 有私人信息或类型：先写 [[Bayesian Game]]。
- 有信号和路径外 belief：转向 [[Perfect Bayesian Equilibrium]]。

## 边界条件

- 参与者的偏好或收益需要能被建模。
- 玩家至少知道规则和信息结构。
- 多重均衡时，博弈论本身可能不能唯一预测结果。

## 失败模式

- 把行为习惯或道德规范直接当作 payoff，但没有解释。
- 忽略信息结构，导致错用均衡概念。
- 只找社会最优，不检查个体是否愿意单独偏离。

## 来自课程位置

- [[01_导论#1.1. 博弈论研究什么]]

## 关联卡片

- [[Game Theory-hub]]
- [[Game Theory Problem Solving Map]]
- [[Nash Equilibrium]]
- [[Mixed Strategy Nash Equilibrium]]
- [[Subgame Perfect Nash Equilibrium]]
- [[Bayesian Game]]
- [[Perfect Bayesian Equilibrium]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
