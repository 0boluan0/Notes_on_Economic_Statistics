---
aliases:
- Theta
- 期权Theta
- 时间损耗
- θ
tags:
- concept
- derivatives
---
# Theta

## 先记一句话

Theta 衡量时间流逝对衍生品价值的影响，常被叫作 time decay。

## 它是什么

Theta 常写作：

$$
\Theta=\frac{\partial V}{\partial t}
$$

实际使用时要确认 $t$ 是日历时间还是剩余到期时间；不同教材符号可能相反。

## 解决什么判断

它回答：“其他条件不变，过一天我的期权组合会损耗或增加多少价值？”

## 最小例子

期权多头 Theta = -500，表示其他条件不变，每过一天价值约下降 500。

## 易混点

- Theta 的符号定义容易因 $t$ 的方向不同而相反，做题要看题目约定。
- Theta 不能像 Delta 那样直接用交易标的完全对冲，因为时间一定流逝。
- 卖期权常有正 Theta，但会承担 Gamma/Vega 风险。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Delta]]
- [[Gamma]]
- [[Vega]]
- [[Greeks Hedging Map]]
