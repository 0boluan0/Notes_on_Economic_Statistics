---
aliases:
- Player
- player
- 玩家
- 局中人
tags:
- concept
- game-theory
---
# player

## 一句话记忆

player 是博弈中拥有选择权、收益目标和策略集合的决策者。

## 它是什么

在策略式博弈中，玩家集合通常记作 $N$，其中 $i\in N$ 表示一个具体玩家。

## 解决什么判断

- 谁的选择会影响博弈结果。
- 每个收益数字属于哪个玩家。
- 后续写策略集合 $S_i$ 和收益函数 $u_i$ 时，下标对应谁。

## 最小例子

囚徒困境中有两个 player：囚徒 1 和囚徒 2。两人都要在合作和背叛之间选择。

## 易混点

- player 不一定是自然人，也可以是企业、政府、国家或算法。
- player 是决策者，不是策略本身。
- 同一个玩家在动态博弈中可能有多个决策点，但仍是同一个 player。

## 来自课程位置

- [[03_策略式博弈#2.1. 策略式博弈]]

## 关联卡片

- [[strategies]]
- [[payoffs]]
- [[strategic-form Games]]
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
