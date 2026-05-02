---
aliases:
- Mixed Strategy Nash Equilibrium
- Mixed Strategy NE
- 混合策略纳什均衡
tags:
- concept
- game-theory
---
# Mixed Strategy Nash Equilibrium

## 一句话记忆

混合策略纳什均衡是玩家用概率随机选择纯策略，并让对方在被混合的纯策略之间无差异。

## 它是什么

Mixed Strategy 是纯策略集合上的概率分布。若玩家 $i$ 的纯策略集合为 $S_i$，混合策略 $\sigma_i$ 给每个纯策略分配概率，且概率非负、总和为 1。

## 解决什么判断

- 没有纯策略纳什均衡时，稳定的随机化结果是什么。
- 对手以什么概率混合，才能让自己愿意混合。
- 哪些纯策略在均衡中被使用，哪些概率为 0。

## 最小例子

Matching Pennies 没有纯策略纳什均衡。双方各以 $1/2$ 概率选择正反面时，对方无法通过改变概率获得更高期望收益。

## 易混点

- 混合策略不是“随便随机”，而是让对手无差异。
- 被赋予正概率的纯策略必须带来相同期望收益。
- 概率为 0 的纯策略不需要和支持集内策略收益相等。

## 来自课程位置

- [[03_策略式博弈#9. 混合策略纳什均衡]]
- [[04_案例（囚徒困境与纳什均衡）]]

## 关联卡片

- [[Finding Mixed Strategy Nash Equilibrium]]
- [[Pure Strategy Nash Equilibrium]]
- [[Best-Reply Function]]
- [[strategies]]
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
