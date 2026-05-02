---
aliases:
- Bayesian Nash Equilibrium
- BNE
- 贝叶斯纳什均衡
tags:
- concept
- game-theory
---
# Bayesian Nash Equilibrium

## 一句话记忆

BNE 是每个类型在给定信念和其他类型策略时都不愿偏离的策略组合。

## 它是什么

在贝叶斯博弈中，策略是类型到行动的函数。策略组合 $s^*$ 是 Bayesian Nash Equilibrium，如果对每个玩家 $i$ 的每个类型 $t_i$，$s_i^*(t_i)$ 都最大化其条件期望收益。

形式上：

$$
s_i^*(t_i)\in\arg\max_{a_i}
\mathbb E_{t_{-i}\mid t_i}
\left[
u_i(a_i,s_{-i}^*(t_{-i});t_i,t_{-i})
\right].
$$

## 解决什么判断

- 不完全信息下某个类型是否愿意偏离。
- 玩家不知道对方类型时如何计算期望收益。
- 贝叶斯版性别博弈、拍卖和招标问题的均衡。

## 最小例子

若玩家 2 有两种类型，则玩家 1 的收益要对这两种类型取期望；玩家 2 的每个类型分别检查自己的最优反应。

## 易混点

- BNE 是静态不完全信息博弈的均衡；动态不完全信息通常需要 [[Perfect Bayesian Equilibrium]]。
- 检查 BNE 时要逐类型检查，不是只检查自然人平均收益。

## 来自课程位置

- [[05_贝叶斯博弈#1.4. 纯策略均衡]]
- [[05_贝叶斯博弈#1.5. 混合策略均衡验证]]

## 关联卡片

- [[Bayesian Game]]
- [[Nash Equilibrium]]
- [[Type (Game Theory)]]
- [[Belief (Game Theory)]]
- [[Perfect Bayesian Equilibrium]]
