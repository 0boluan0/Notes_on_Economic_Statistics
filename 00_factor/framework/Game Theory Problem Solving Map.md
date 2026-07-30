---
aliases:
- Game Theory Problem Solving Map
- Game Theory Problem Map
- 博弈论解题框架
tags:
- framework
- game-theory
type: framework
---
# Game Theory Problem Solving Map

## 什么时候用

当一道博弈题不知道先删策略、先找均衡、先画博弈树，还是先写 belief 时，用这张框架卡。

## 为什么这样看

博弈题的第一步不是计算，而是判断表示方式、行动顺序和信息结构。不同结构对应不同工具。

## 题型识别

| 题目形态 | 第一工具 |
|---|---|
| payoff matrix | [[Iterated Deletion of Strictly Dominated Strategies]] 或 [[Finding Pure Strategy Nash Equilibria]] |
| 没有纯策略 NE | [[Finding Mixed Strategy Nash Equilibrium]] |
| 连续策略、可求导利润 | [[Best-Reply Function]] |
| 厂商价格/产量/差异化竞争 | [[Oligopoly Competition Map]] |
| game tree、顺序行动 | [[Backward Induction Procedure]] |
| 动态威胁 | [[Subgame Perfect Nash Equilibrium]] |
| 类型、私人信息、静态不完全信息 | [[Bayesian Nash Equilibrium]] |
| 信号、belief、路径外信息集 | [[Checking Perfect Bayesian Equilibrium]] |

## 边界条件

- 策略式表达适合一次性比较策略组合。
- 扩展式表达适合时间顺序和信息结构。
- 混合策略求解依赖支持集假设。
- SPNE 只能检查合法子博弈；没有真子博弈时筛选力有限。
- PBE 需要同时写策略和 belief。

## 失败模式

- 没分清玩家、行动和策略。
- 跨玩家比较 payoff 数字。
- 用纯策略方法处理本来需要混合策略的问题。
- 在标准 Bertrand 模型里硬套一阶条件。
- 动态博弈只求 NE，不检查可信性。
- 不完美信息博弈只写策略，不写 belief。

## 来自课程位置

- [[02_初步认识博弈]]
- [[03_策略式博弈]]
- [[04_案例（囚徒困境与纳什均衡）]]
- [[05_贝叶斯博弈]]
- [[06_扩展性博弈]]
- [[07_子博弈不完全信息]]

## 关联卡片

- [[strategic-form Games]]
- [[Extensive-form Game]]
- [[Nash Equilibrium]]
- [[Finding Pure Strategy Nash Equilibria]]
- [[Finding Mixed Strategy Nash Equilibrium]]
- [[Backward Induction Procedure]]
- [[Subgame Perfect Nash Equilibrium]]
- [[Checking Perfect Bayesian Equilibrium]]
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
