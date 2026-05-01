---
aliases:
- Pure Strategy Nash Equilibrium
- Pure Strategy NE
- 纯策略纳什均衡
tags:
- concept
- game-theory
---
# Pure Strategy Nash Equilibrium

## 一句话记忆

纯策略纳什均衡是每个玩家都选择一个确定策略，且互为最优反应的策略组合。

## 它是什么

Pure Strategy Nash Equilibrium 是 [[Nash Equilibrium]] 在纯策略集合中的情形。每个玩家不是随机化，而是确定选择某个 $s_i\in S_i$。

## 解决什么判断

- 一个 payoff matrix 中哪些格子稳定。
- 是否存在确定策略均衡。
- 是否需要转向 [[Mixed Strategy Nash Equilibrium]]。

## 最小例子

猎鹿博弈中 $(Stag, Stag)$ 和 $(Hare, Hare)$ 都可能是纯策略纳什均衡，因为给定对方选择时，自己不愿单独偏离。

## 易混点

- 纯策略均衡可以不唯一。
- 没有纯策略均衡不代表没有纳什均衡，有限博弈通常仍存在混合策略均衡。
- 找纯策略均衡时，必须按玩家分别比较 payoff，不能跨玩家比较数字。

## 来自课程位置

- [[03_策略式博弈#纯策略的纳什均衡]]

## 关联卡片

- [[Nash Equilibrium]]
- [[Finding Pure Strategy Nash Equilibria]]
- [[Best-Reply Function]]
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
