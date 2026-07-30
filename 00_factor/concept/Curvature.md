---
aliases:
- Curvature
- 曲率
- fixed-income curvature
tags:
- concept
- fixed-income
---

# Curvature

## 先记一句话

在这门利率风险课里，Curvature 基本就是 [[Convexity|债券凸性]]：价格-收益率曲线的二阶弯曲程度。

## 它是什么

Curvature 强调图形直觉：债券价格和收益率不是一条直线，所以当收益率变化较大时，久期给出的线性切线近似会产生误差。

## 解决什么判断

它回答：“为什么同样久期的组合，在大幅利率变化下价格变动仍然不同？”

## 最小例子

组合 A 和组合 B 久期都为 5.95，但 A 的 curvature 更高。若收益率从 10% 跳到 15%，A 的价格跌幅会小于只用久期估计的线性跌幅。

## 易混点

- 严格做公式和价格近似时，优先看 [[Convexity]]。
- Curvature 不是 [[Key Rate Duration]]；前者是二阶敏感度，后者是曲线不同期限的一阶敏感度。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Convexity]]
- [[Modified Duration]]
- [[duration|Duration]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[Convexity]]、[[Key Rate Duration]]、[[09_利率风险]]、[[Modified Duration]]、[[duration]]。
