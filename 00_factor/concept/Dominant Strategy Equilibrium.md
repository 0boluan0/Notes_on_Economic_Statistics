---
aliases:
- Dominant Strategy Equilibrium
- Dominant-strategy Equilibrium
- 优势策略均衡
tags:
- concept
- game-theory
---
# Dominant Strategy Equilibrium

## 一句话记忆

优势策略均衡是每个玩家都选择“不管别人怎么选都最优”的策略组合。

## 它是什么

若每个玩家 $i$ 都有一个策略 $s_i^*$，使得对任意对手策略 $s_{-i}$ 和任意其他策略 $s_i$ 都有：

$$
u_i(s_i^*,s_{-i})\ge u_i(s_i,s_{-i}),
$$

则 $s^*=(s_i^*)_{i\in N}$ 是优势策略均衡。

## 解决什么判断

- 玩家是否存在不依赖对手选择的最优策略。
- 囚徒困境为什么会落在 $(D,D)$。
- 某个均衡是否比普通 [[Nash Equilibrium]] 更强。

## 最小例子

囚徒困境中，每个囚徒无论对方合作还是背叛，背叛都更好，因此 $(D,D)$ 是优势策略均衡。

## 易混点

- 优势策略均衡一定是纳什均衡，但纳什均衡不一定是优势策略均衡。
- “给定对方当前策略最优”只是纳什均衡；“不管对方怎么选都最优”才是优势策略。

## 来自课程位置

- [[03_策略式博弈#2.5. 优势策略均衡]]

## 关联卡片

- [[Nash Equilibrium]]
- [[Strictly Dominated Strategy]]
- [[Prisoner's Dilemma]]
- [[Game Theory-hub]]
