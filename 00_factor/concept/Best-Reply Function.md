---
aliases:
- Best-Reply Function
- Best Response Function
- Best Reply
- 最优反应函数
tags:
- concept
- game-theory
---
# Best-Reply Function

## 一句话记忆

最优反应函数给出：对手这么选时，我哪些策略最优。

## 它是什么

给定对手策略 $s_{-i}$，玩家 $i$ 的最优反应集合为：

$$
BR_i(s_{-i})=\arg\max_{s_i\in S_i}u_i(s_i,s_{-i})
$$

## 解决什么判断

- 某个策略是否是对对手策略的最优反应。
- 两个玩家的最优反应是否相交。
- 如何从反应函数图或 payoff matrix 找纳什均衡。

## 最小例子

若列玩家选 $L$ 时，行玩家选 $T$ 的收益高于选 $B$，则 $T\in BR_{\text{row}}(L)$。

## 易混点

- 最优反应可以是集合，不一定只有一个策略。
- 纳什均衡要求所有玩家同时处在彼此的最优反应上。
- 连续策略下 best reply 可能是一条函数或对应关系。

## 来自课程位置

- [[03_策略式博弈#2.9. 最优反应函数]]
- [[04_案例（囚徒困境与纳什均衡）#3.2. 最优反应函数]]

## 关联卡片

- [[Nash Equilibrium]]
- [[Finding Pure Strategy Nash Equilibria]]
- [[Finding Mixed Strategy Nash Equilibrium]]
- [[Cournot Competition]]
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
