---
aliases:
- Investment Competition Game
- All-pay Investment Game
- 投资竞争博弈
tags:
  - concept
  - game-theory
---
# Investment Competition Game

## 一句话记忆

投资竞争博弈是“谁投入稍微更高谁赢，但投入成本无论输赢都要付”的竞争。

## 它是什么

两名玩家选择投资 $x,y\in[0,1]$。投资更高者获得收益，投资较低者仍承担成本。

玩家 1 的收益可写为：

$$
\pi_1(x,y)=
\begin{cases}
1-x, & x>y,\\
\frac12-x, & x=y,\\
-x, & x<y.
\end{cases}
$$

## 解决什么判断

- 为什么“比对方高一点点”会破坏纯策略均衡。
- 为什么某些连续策略博弈需要混合策略。
- 为什么投资、竞赛、拍卖类问题容易出现过度投入。

## 最小例子

若两人投资相等，任一方都可能略微提高投资并抢走全部收益；若一方刚好赢，对方也可能再加一点反超。

## 易混点

- 没有纯策略均衡不代表没有均衡。
- 这里的成本是 sunk cost，输了也要付。

## 来自课程位置

- [[04_案例（囚徒困境与纳什均衡）#8. 投资竞争博弈]]

## 关联卡片

- [[Oligopoly Competition Map]]
- [[Mixed Strategy Nash Equilibrium]]
- [[Finding Mixed Strategy Nash Equilibrium]]
