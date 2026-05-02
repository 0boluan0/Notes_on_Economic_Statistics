---
aliases:
- Information Set
- 信息集
tags:
- concept
- game-theory
---
# Information Set

## 一句话记忆

信息集是玩家行动时无法区分的一组可能决策节点。

## 它是什么

在扩展式博弈中，如果玩家处于同一信息集中的若干节点之一，但不知道具体是哪一个节点，那么他必须在这些节点上采取同一行动。

## 解决什么判断

- 博弈是否是完美信息博弈。
- 某个节点能否作为子博弈起点。
- 是否需要为玩家写 belief。

## 最小例子

若玩家 1 先选 $C/D$，玩家 2 后行动但不知道玩家 1 选了什么，则玩家 2 的两个决策节点属于同一个信息集。

## 易混点

- 单点信息集表示玩家知道自己在哪个节点。
- 子博弈必须从单点信息集开始，且不能切割任何信息集。
- 非单点信息集才需要讨论 belief。

## 来自课程位置

- [[06_扩展性博弈#6. 信息集]]
- [[07_子博弈不完全信息#2. belief 的引入]]

## 关联卡片

- [[Extensive-form Game]]
- [[Subgame]]
- [[Subgame Perfect Nash Equilibrium]]
- [[Belief (Game Theory)]]
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
