---
aliases:
- Rationalisability
- Rationalizability
- Rationalizable Strategies
- 可理性化
tags:
- concept
- game-theory
---
# Rationalisability

## 一句话记忆

可理性化策略是能作为某种合理信念下最优反应的策略。

## 它是什么

若一个策略可以被解释为玩家在某个关于他人策略的信念下的最优反应，则该策略是 rationalisable。

## 解决什么判断

- 哪些策略在理性玩家的共同理性假设下还能保留。
- 连续删除严格劣势策略后剩下哪些策略。
- 为什么纳什均衡策略一定可理性化，但可理性化策略不一定组成纳什均衡。

## 最小例子

如果某策略虽然不是均衡策略，但它是玩家对某个可能对手选择的最优反应，那么它可以是可理性化的。

## 易混点

- Nash Equilibrium 要求所有玩家的策略同时匹配；rationalisability 只要求每个策略能被某种信念支持。
- 严格劣势策略不可理性化。
- 可理性化集合通常比纳什均衡集合更大。

## 来自课程位置

- [[03_策略式博弈#定义 8（最优反应函数, Best-reply Function）]]
- [[03_策略式博弈#混合策略扩展中的定义]]

## 关联卡片

- [[Strictly Dominated Strategy]]
- [[Best-Reply Function]]
- [[Nash Equilibrium]]
- [[Game Theory Problem Solving Map]]
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
