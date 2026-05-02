---
aliases:
- Payoff
- Payoffs
- payoff
- payoffs
- 收益
- 效用
tags:
- concept
- game-theory
---
# payoffs

## 一句话记忆

payoff 是某个策略组合发生后，每个玩家各自得到的收益。

## 它是什么

在策略式博弈中，玩家 $i$ 的收益函数常写作：

$$
u_i:S_1\times\cdots\times S_n\to \mathbb{R}
$$

它把一个策略组合映射成玩家 $i$ 的收益。

## 解决什么判断

- 每个格子里的数字分别属于谁。
- 玩家是否有动力单独偏离。
- 最优反应、占优策略和纳什均衡如何比较。

## 最小例子

若某格子写作 $(3,1)$，通常表示行玩家 payoff 为 3，列玩家 payoff 为 1。

## 易混点

- payoff 是玩家自己的排序或效用，不一定是金钱。
- 同一策略组合下，不同玩家的 payoff 可以完全不同。
- 求纳什均衡时，比较的是同一玩家在给定对手策略下的 payoff。

## 来自课程位置

- [[03_策略式博弈#2.1. 策略式博弈]]

## 关联卡片

- [[player]]
- [[strategies]]
- [[strategic-form Games]]
- [[Nash Equilibrium]]
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
