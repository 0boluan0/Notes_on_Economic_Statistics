---
aliases:
- Finding Mixed Strategy Nash Equilibrium
- Mixed Strategy Nash Equilibrium Procedure
- 混合策略纳什均衡求解步骤
tags:
- procedure
- game-theory
---
# Finding Mixed Strategy Nash Equilibrium

## 输入

- 策略式博弈矩阵。
- 候选支持集，也就是每个玩家可能赋予正概率的纯策略。

## 输出

- 每个玩家的混合概率。
- 支持集是否有效的检查结果。

## Step 1：设概率

在 $2\times2$ 博弈中，设行玩家以概率 $p$ 选择上方策略，列玩家以概率 $q$ 选择左侧策略。

## Step 2：让对方无差异

求行玩家的混合概率 $p$ 时，让列玩家在自己的两个纯策略之间期望收益相等。

求列玩家的混合概率 $q$ 时，让行玩家在自己的两个纯策略之间期望收益相等。

## Step 3：解概率

解出 $p$ 和 $q$，并检查：

$$
0\le p\le 1,\quad 0\le q\le 1
$$

若概率不在区间内，当前支持集假设无效。

## Step 4：检查未使用策略

若某些纯策略概率为 0，要确认这些未使用策略的期望收益不高于支持集内策略。

## Step 5：写出均衡

用概率分布写出：

$$
(\sigma_1^*,\sigma_2^*)
$$

并说明每个玩家为什么没有偏离激励。

## 检查点

- 让“对方”无差异，不是让自己无差异来求自己的概率。
- 只对支持集内的纯策略要求收益相等。
- 先找纯策略 NE，再补混合策略 NE。

## 常见错误

- 把 $p$ 和 $q$ 对应玩家写反。
- 没检查概率是否在 $[0,1]$。
- 忽略纯策略均衡。
- 把所有纯策略都强行设为正概率。

## 来自课程位置

- [[03_策略式博弈#9. 混合策略纳什均衡]]
- [[04_案例（囚徒困境与纳什均衡）]]

## 关联卡片

- [[Mixed Strategy Nash Equilibrium]]
- [[Pure Strategy Nash Equilibrium]]
- [[Best-Reply Function]]
- [[Nash Equilibrium]]
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
