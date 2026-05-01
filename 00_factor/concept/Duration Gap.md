---
aliases:
- Duration Gap
- 久期缺口
- 持续期缺口
tags:
- concept
- risk-management
- banking
---
# Duration Gap

## 先记一句话

Duration Gap 用资产久期和负债久期的差衡量利率变化对银行经济价值的影响。

## 它是什么

常见口径为：

$$
DGAP=D_A-\frac{L}{A}D_L
$$

其中 $D_A$ 是资产久期，$D_L$ 是负债久期，$A$ 和 $L$ 是资产与负债市值。

## 解决什么判断

它回答：“利率变动会让银行净值上升还是下降？”

## 最小例子

若 $DGAP>0$，资产久期相对负债更长。利率上升时资产现值下跌更多，银行经济价值通常下降。

## 易混点

- [[Interest Rate Sensitivity Gap]] 看净利息收入，Duration Gap 看经济价值。
- DGAP 为 0 是免疫目标，但现实中现金流、期权和曲线形状会变。
- 利率非平行移动时，要结合 [[Key Rate Duration]]。

## 来自课程位置

- [[09_利率风险]]

## 关联卡片

- [[duration|Duration]]
- [[Modified Duration]]
- [[Effective Duration]]
- [[Interest Rate Sensitivity Gap]]
