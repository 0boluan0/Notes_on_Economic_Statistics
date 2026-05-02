---
aliases:
- Advertising Game
- 广告竞争博弈
tags:
- concept
- game-theory
---
# Advertising Game

## 一句话记忆

广告竞争博弈中，广告投入提高市场份额，但投入本身也是成本。

## 它是什么

两家厂商选择广告投入 $x,y$，市场份额由投入占比决定：

$$
\pi_1(x,y)=\frac{x}{x+y}-x.
$$

## 解决什么判断

- 为什么投入的最优反应可能先上升后下降。
- 为什么该模型不一定是纯粹亏模或超模。
- 广告投入为什么会出现边际收益递减。

## 最小例子

当 $y>0$ 时：

$$
\frac{\partial\pi_1}{\partial x}
=\frac{y}{(x+y)^2}-1=0,
$$

因此：

$$
B_1(y)=\max\{0,\sqrt y-y\}.
$$

## 易混点

- 广告投入不是越多越好，因为投入本身减少利润。
- 反应函数可能非单调，不能简单归类为 Cournot 或 Hotelling。

## 来自课程位置

- [[04_案例（囚徒困境与纳什均衡）#7. 广告竞争博弈]]

## 关联卡片

- [[Oligopoly Competition Map]]
- [[Best-Reply Function]]
- [[Submodular Game]]
- [[Supermodular Game]]
