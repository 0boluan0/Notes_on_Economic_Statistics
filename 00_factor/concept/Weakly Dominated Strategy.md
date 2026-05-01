---
aliases:
- Weakly Dominated Strategy
- Weak Dominance
- 弱劣势策略
- 弱占优策略
tags:
- concept
- game-theory
---
# Weakly Dominated Strategy

## 一句话记忆

弱劣势策略是在所有情况下不更好，并且至少有一种情况下更差的策略。

## 它是什么

若存在策略 $s_i'$，使得对所有 $s_{-i}$：

$$
u_i(s_i',s_{-i})\ge u_i(s_i,s_{-i})
$$

并且至少对某些 $s_{-i}$ 严格大于，则 $s_i$ 是 weakly dominated strategy。

## 解决什么判断

- 某个策略是否“永远不更好”。
- 为什么弱劣势删除比严格劣势删除更危险。
- 动态博弈剪枝时某些选择为什么可以被排除。

## 最小例子

如果策略 A 在所有情况下至少和策略 B 一样好，且在一种情况下更好，那么 B 被 A 弱支配。

## 易混点

- 弱支配不是严格支配；删除弱劣势策略可能改变均衡集合。
- 弱支配中的相等情况很重要。
- 看到“至少一样好”时，不要直接当作严格占优。

## 来自课程位置

- [[03_策略式博弈#定义 2（严格占优, Strict Dominance）]]
- [[03_策略式博弈#纯策略的纳什均衡]]

## 关联卡片

- [[Strictly Dominated Strategy]]
- [[Backward Induction]]
- [[Rationalisability]]
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
