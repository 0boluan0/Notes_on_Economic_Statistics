---
aliases:
- Subgame Perfect Nash Equilibrium
- SPNE
- 子博弈精炼纳什均衡
- 子博弈完美纳什均衡
tags:
- concept
- game-theory
---
# Subgame Perfect Nash Equilibrium

## 一句话记忆

SPNE 是在每一个子博弈里都仍然是纳什均衡的策略组合。

## 它是什么

Subgame Perfect Nash Equilibrium 是比 [[Nash Equilibrium]] 更强的动态博弈均衡概念。它要求策略组合在原博弈和每一个 [[Subgame]] 中都是纳什均衡。

## 解决什么判断

- 一个动态博弈均衡是否包含不可信威胁。
- 给定某个子博弈后，玩家是否仍会按原策略行动。
- 完美信息有限博弈的逆向归纳结果是否可信。

## 最小例子

进入威慑中，“不进入，进入后在位者打价格战”可能是纳什均衡，但如果进入已经发生，在位者打价格战不是最优反应，那么它不是 SPNE。

## 易混点

- SPNE 一定是 NE，但 NE 不一定是 SPNE。
- 子博弈必须从单点信息集开始，且不能切割信息集。
- 在完美信息有限博弈中，逆向归纳通常给出 SPNE。

## 来自课程位置

- [[06_扩展性博弈#8. 子博弈精炼纳什均衡]]

## 关联卡片

- [[Nash Equilibrium]]
- [[Backward Induction]]
- [[Backward Induction Procedure]]
- [[Subgame]]
- [[Information Set]]
- [[Perfect Bayesian Equilibrium]]
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
