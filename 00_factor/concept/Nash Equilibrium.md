---
aliases:
- Nash Equilibrium
- NE
- 纳什均衡
tags:
- concept
- game-theory
---
# Nash Equilibrium

## 一句话记忆

纳什均衡是没有任何一个玩家愿意单独偏离的策略组合。

## 它是什么

策略组合 $s^*=(s_i^*,s_{-i}^*)$ 是 Nash Equilibrium，如果对每个玩家 $i$ 和每个可选策略 $s_i\in S_i$ 都有：

$$
u_i(s_i^*,s_{-i}^*)\ge u_i(s_i,s_{-i}^*)
$$

## 解决什么判断

- 一个策略组合是否稳定。
- 玩家给定他人选择后是否还有更优偏离。
- 多个玩家的最优反应是否同时成立。

## 最小例子

囚徒困境中 $(Defect, Defect)$ 是纳什均衡，因为任一囚徒在对方背叛时单独改成合作都会更差。

## 易混点

- 纳什均衡不一定是社会最优。
- 纳什均衡可以有多个，也可以没有纯策略均衡。
- 动态博弈中的纳什均衡可能包含不可信威胁，需要 [[Subgame Perfect Nash Equilibrium]] 进一步筛选。

## 来自课程位置

- [[03_策略式博弈#2.7. 纳什均衡]]
- [[06_扩展性博弈#3. 纳什均衡与可信性]]

## 关联卡片

- [[Best-Reply Function]]
- [[Pure Strategy Nash Equilibrium]]
- [[Mixed Strategy Nash Equilibrium]]
- [[Finding Pure Strategy Nash Equilibria]]
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
