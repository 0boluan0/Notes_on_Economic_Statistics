---
aliases:
- Strategic-form Game
- Strategic-form Games
- strategic-form Games
- Normal-form Game
- 策略式博弈
- 策略式表达
tags:
- concept
- game-theory
---
# strategic-form Games

## 一句话记忆

策略式博弈用玩家、策略集合和收益函数一次性描述同时选择的博弈。

## 它是什么

Strategic-form game 通常写作：

$$
G=(N,(S_i)_{i\in N},(u_i)_{i\in N})
$$

其中 $N$ 是玩家集合，$S_i$ 是玩家 $i$ 的策略集合，$u_i$ 是玩家 $i$ 的收益函数。

## 解决什么判断

- 一道题能否用矩阵或策略组合来表示。
- 给定策略组合时，每个玩家收益是多少。
- 能否用占优策略、最优反应或纳什均衡来求解。

## 最小例子

囚徒困境可以写成两个玩家、两个策略集合和四个收益组合的 $2\times 2$ 策略式博弈。

## 易混点

- 策略式博弈不等于一定“同时行动”；扩展式博弈也可以转成策略式表达。
- 动态博弈转策略式时，策略必须写成完整计划。
- 策略式表达会压平时间顺序，所以可信威胁要进一步用 [[Subgame Perfect Nash Equilibrium]] 检查。

## 来自课程位置

- [[03_策略式博弈#2.1. 策略式博弈]]
- [[06_扩展性博弈#2. 扩展式转策略式]]

## 关联卡片

- [[player]]
- [[strategies]]
- [[payoffs]]
- [[Nash Equilibrium]]
- [[Extensive-form Game]]
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
