---
aliases:
- Submodular Game
- Strategic Substitutes
- 亏模博弈
- 策略替代
tags:
- concept
- game-theory
---
# Submodular Game

## 一句话记忆

亏模博弈中，对手策略变量上升时，我的最优策略变量下降。

## 它是什么

Submodular game 常用于描述策略替代（strategic substitutes）：别人多做，我反而少做。

## 解决什么判断

- 最优反应函数为什么向下倾斜。
- Cournot 中对手增产后我为什么减产。
- 某个连续策略博弈是否表现为策略替代。

## 最小例子

Cournot 竞争中：

$$
B_1(y)=\frac{a-c-by}{2b},
$$

因此 $B_1'(y)=-\frac12<0$。

## 易混点

- 亏模不等于收益低；它说的是策略之间的方向关系。
- 要看最优反应怎么随对手变量变化，而不是只看均衡点。

## 来自课程位置

- [[04_案例（囚徒困境与纳什均衡）#6. 亏模与超模博弈]]

## 关联卡片

- [[Supermodular Game]]
- [[Cournot Competition]]
- [[Best-Reply Function]]
