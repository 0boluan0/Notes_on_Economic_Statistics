---
aliases:
- Convexity
- 凸性
- bond convexity
tags:
- concept
- fixed-income
---

# Convexity

## 先记一句话

Convexity 是债券价格对收益率变化的二阶敏感度，用来修正久期的线性近似误差。

## 它是什么

标准化凸性可写为：

$$
C=\frac{1}{P}\frac{d^2P}{dy^2}
$$

结合久期的一阶项，价格变化近似为：

$$
\frac{\Delta P}{P}\approx -D\Delta y+\frac{1}{2}C(\Delta y)^2
$$

## 解决什么判断

它回答：“利率变化不小的时候，只用久期会错多少，二阶修正应该往哪里加？”

## 最小例子

两只债券久期相同，凸性更高的那只在利率大幅上升时价格跌得较少，在利率大幅下降时涨得更多。

## 易混点

- 课程里有时把 convexity 叫 [[Curvature|曲率]]；在固定收益风险里二者指同一个二阶敏感度口径。
- 凸性不是一阶敏感度，不能替代 [[Modified Duration]]。
- 含权债券可能出现负凸性，需要结合 [[Effective Duration]]。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[duration|Duration]]
- [[Modified Duration]]
- [[Effective Duration]]
- [[Curvature]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Curvature]]、[[Modified Duration]]、[[Effective Duration]]、[[09_利率风险]]、[[duration]]。
