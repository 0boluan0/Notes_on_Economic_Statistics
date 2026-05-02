---
aliases:
- Strategy
- Strategies
- strategy
- strategies
- 策略
tags:
- concept
- game-theory
---
# strategies

## 一句话记忆

strategy 是玩家在博弈中可提交的完整选择方案，不只是最后实际发生的一个动作。

## 它是什么

在策略式博弈中，玩家 $i$ 的策略集合记作 $S_i$。一个策略组合记作：

$$
s=(s_1,\dots,s_n)
$$

其中 $s_i\in S_i$。

## 解决什么判断

- 玩家有哪些可选方案。
- 一个结果是由哪些玩家的策略组合出来的。
- 动态博弈转成策略式博弈时，每个玩家在所有可能决策点的完整计划是什么。

## 最小例子

在囚徒困境中，每个玩家的策略集合是：

$$
S_i=\{\text{Cooperate},\text{Defect}\}
$$

## 易混点

- strategy 不是“计策”或“想法”，而是模型里允许选择的方案。
- 动态博弈中 strategy 是完整 contingent plan，不只是某一个节点上的 action。
- mixed strategy 是对纯策略的概率分布。

## 来自课程位置

- [[03_策略式博弈#2.1. 策略式博弈]]

## 关联卡片

- [[player]]
- [[payoffs]]
- [[strategic-form Games]]
- [[Mixed Strategy Nash Equilibrium]]
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
