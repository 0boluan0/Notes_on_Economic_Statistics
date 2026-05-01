---
aliases:
- Key Rate Duration
- 局部久期
- 关键期限久期
tags:
- concept
- fixed-income
- risk-management
---
# Key Rate Duration

## 先记一句话

Key Rate Duration 衡量收益率曲线上某个关键期限利率单独变化时，组合价值怎么变。

## 它是什么

对期限点 $T_i$ 的零息收益率 $y(T_i)$，局部久期可写作：

$$
D_i=-\frac{1}{P}\frac{\partial P}{\partial y(T_i)}
$$

它把总利率风险拆到 1 年、2 年、5 年、10 年等关键期限上。

## 解决什么判断

它回答：“我的利率风险主要暴露在曲线短端、中端还是长端？”

## 最小例子

若组合 10 年 key rate duration 很高，而 1 年 key rate duration 很低，则长端利率上升对组合价格更危险。

## 易混点

- 普通 [[Modified Duration]] 通常默认平行移动；Key Rate Duration 处理非平行移动。
- 它是管理 [[Yield Curve Risk]] 的工具，不是现金流平均回收期。
- 多个 key rate duration 的加总近似总久期，但取决于期限点设置和曲线插值方法。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Yield Curve Risk]]
- [[duration|Duration]]
- [[Basis Point Value (BPV)]]
