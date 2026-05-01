---
aliases:
- Effective Duration
- 有效久期
tags:
- concept
- fixed-income
- risk-management
---
# Effective Duration

## 先记一句话

Effective Duration 用“利率上下小幅移动后重新定价”的方式测敏感度，适合现金流会变的债券。

## 它是什么

有效久期常用数值差分：

$$
D_{eff}=\frac{P_- - P_+}{2\Delta y P_0}
$$

其中 $P_-$ 是利率下降后价格，$P_+$ 是利率上升后价格，$P_0$ 是当前价格。

## 解决什么判断

它回答：“如果利率变化会改变现金流本身，价格真实敏感度是多少？”

## 最小例子

可赎回债券在利率下降时更可能被发行人赎回，现金流缩短；此时普通 Macaulay/Modified Duration 不够，需要 Effective Duration。

## 易混点

- [[Modified Duration]] 假设现金流固定；Effective Duration 允许现金流随利率变化。
- 有效久期依赖定价模型，模型错会直接影响结果。
- 它通常用于含权债券、MBS、可提前还款贷款等。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[Implied Option Risk]]
- [[duration|Duration]]
- [[Convexity]]
