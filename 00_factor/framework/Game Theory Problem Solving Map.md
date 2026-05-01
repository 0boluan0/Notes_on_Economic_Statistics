---
aliases:
- Game Theory Problem Solving Map
- Game Theory Problem Map
- 博弈论解题框架
tags:
- framework
- game-theory
---
# Game Theory Problem Solving Map

## 什么时候用

当一道博弈题不知道先找均衡、先删策略，还是先画博弈树时，用这张框架卡。

## 为什么这样看

博弈题的第一步不是计算，而是判断表示方式和信息结构。不同结构对应不同求解工具。

## 题型识别

- payoff matrix：先看 [[Strictly Dominated Strategy]]，再用 [[Finding Pure Strategy Nash Equilibria]]。
- 没有纯策略 NE：转向 [[Finding Mixed Strategy Nash Equilibrium]]。
- continuous strategy：写 [[Best-Reply Function]]，解交点。
- game tree：用 [[Backward Induction Procedure]]。
- dynamic game with threats：检查 [[Subgame Perfect Nash Equilibrium]]。
- incomplete information：需要 [[Bayesian Game]] 或 [[Perfect Bayesian Equilibrium]]。

## 边界条件

- 策略式表达适合一次性比较策略组合。
- 扩展式表达适合时间顺序和信息结构。
- 混合策略求解依赖支持集假设。
- SPNE 只能排除子博弈中的不可信威胁；没有真子博弈时筛选力有限。

## 失败模式

- 没分清玩家和策略。
- 跨玩家比较 payoff 数字。
- 用纯策略方法处理本来需要混合策略的问题。
- 动态博弈只求 NE，不检查可信性。
- 不完美信息博弈乱用逆向归纳。

## 来自课程位置

- [[03_策略式博弈]]
- [[06_扩展性博弈]]
- [[07_子博弈不完全信息]]

## 关联卡片

- [[strategic-form Games]]
- [[Nash Equilibrium]]
- [[Finding Pure Strategy Nash Equilibria]]
- [[Finding Mixed Strategy Nash Equilibrium]]
- [[Backward Induction Procedure]]
- [[Subgame Perfect Nash Equilibrium]]
- [[Game Theory-hub]]

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
