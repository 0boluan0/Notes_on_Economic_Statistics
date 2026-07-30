---
aliases:
- Mixed Strategy Dominance
- Dominance by Mixed Strategies
- 混合策略占优
- 被混合策略严格占优
tags:
  - concept
  - game-theory
---
# Mixed Strategy Dominance

## 一句话记忆

混合策略占优说明：一个纯策略即使不被任何单个纯策略打败，也可能被若干纯策略的概率组合打败。

## 它是什么

若存在玩家 $i$ 的某个混合策略 $\sigma_i$，使得对所有 $s_{-i}$ 都有：

$$
U_i(\sigma_i,s_{-i})>u_i(s_i',s_{-i}),
$$

则纯策略 $s_i'$ 被混合策略严格占优。

## 解决什么判断

- 为什么有些策略不能靠纯策略占优删除，但仍然不合理。
- 混合策略扩展后如何重新理解严格劣势。
- 为什么删除策略时要明确“谁占优谁”。

## 最小例子

某列玩家的 $R$ 可能不被 $L$ 或 $M$ 单独严格占优，但被 $\frac12L+\frac12M$ 严格占优。

## 易混点

- “某策略占优一个混合策略”不能推出它能删除组成该混合策略的纯策略。
- 要删除 $s_i'$，必须证明 $s_i'$ 被别的纯策略或混合策略占优。

## 来自课程位置

- [[03_策略式博弈#10. 混合策略中的占优]]

## 关联卡片

- [[Mixed Strategy Nash Equilibrium]]
- [[Strictly Dominated Strategy]]
- [[Iterated Deletion of Strictly Dominated Strategies]]
