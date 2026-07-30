---
aliases:
- Macaulay Duration
- 马考利久期
- 麦考利久期
tags:
  - concept
  - fixed-income
---
# Macaulay Duration

## 先记一句话

Macaulay Duration 是按现金流现值加权的平均收款时间。

## 它是什么

若每期现金流为 $CF_t$，到期收益率为 $y$，则：

$$
D_M=
\frac{\sum_{t=1}^{n}t\frac{CF_t}{(1+y)^t}}
{\sum_{t=1}^{n}\frac{CF_t}{(1+y)^t}}
$$

若一年付息 $m$ 次，周期收益率应使用 $y/m$，最后再把周期数除以 $m$ 转成年。

## 解决什么判断

它回答：“这只债券的现金流平均要多久收回来？”

## 最小例子

3 年期零息债券没有中间现金流，$D_M=3$ 年；同期限附息债券有票息提前收回，所以 $D_M<3$ 年。

## 易混点

- Macaulay Duration 的单位是时间，不是百分比价格变化。
- 估计价格变化时要先转成 [[Modified Duration]]。
- 收益率周期、现金流周期和时间单位必须统一。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[duration|Duration]]
- [[Macaulay Duration Calculation]]
- [[Modified Duration]]
