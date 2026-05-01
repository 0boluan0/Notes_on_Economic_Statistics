---
aliases:
- Strictly Dominated Strategy
- Strict Dominance
- 严格劣势策略
- 严格占优策略
tags:
- concept
- game-theory
---
# Strictly Dominated Strategy

## 一句话记忆

严格劣势策略是无论别人怎么选，都比另一个策略更差的策略。

## 它是什么

若存在玩家 $i$ 的另一个策略 $s_i'$，使得对所有对手策略 $s_{-i}$ 都有：

$$
u_i(s_i',s_{-i})>u_i(s_i,s_{-i})
$$

则 $s_i$ 被 $s_i'$ 严格支配，$s_i$ 是 strictly dominated strategy。

## 解决什么判断

- 哪些策略理性玩家绝不会选择。
- 能否通过连续删除严格劣势策略简化博弈。
- 哪些策略可以进入可理性化集合。

## 最小例子

囚徒困境中，对每个囚徒来说，背叛严格支配合作，因为无论对方合作还是背叛，背叛收益都更高。

## 易混点

- “严格占优策略”和“严格劣势策略”是一对关系，不能混着说。
- 严格劣势策略可以安全删除；弱劣势策略删除顺序可能影响结果。
- 一个混合策略也可能严格支配某个纯策略。

## 来自课程位置

- [[03_策略式博弈#定义 3（严格劣势策略, Strictly Dominated Strategy）]]
- [[03_策略式博弈#混合策略中的严格优势与严格劣势]]

## 关联卡片

- [[Weakly Dominated Strategy]]
- [[Rationalisability]]
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
