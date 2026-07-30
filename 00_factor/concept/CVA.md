---
aliases:
- CVA
- Credit Valuation Adjustment
- 信用估值调整
tags:
- concept
- credit-risk
- derivatives
---

# CVA

## 先记一句话

CVA 是因交易对手可能违约而从无违约衍生品价值中扣掉的信用风险成本。

## 它是什么

离散近似可写为：

$$
CVA=\sum_i PD(t_{i-1},t_i)\cdot EAD(t_i)\cdot LGD\cdot DF(t_i)
$$

风险调整价值：

$$
V_{credit}=V_{risk-free}-CVA
$$

## 解决什么判断

它回答：“如果交易对手可能违约，这笔衍生品对我还值多少钱？”

## 最小例子

一笔互换对银行为正价值。交易对手信用恶化后，银行预期能收到的钱变得不确定，CVA 增大，衍生品价值下降。

## 易混点

- CVA 是交易对手违约风险；[[DVA]] 是自身违约风险对估值的调整。
- CVA 依赖未来暴露路径，不只是当前市值。
- [[Netting]] 和抵押品会降低 EAD，从而降低 CVA。

## 来自课程位置

- [[17_OTC衍生产品市场的监管]]
- [[19_违约风险]]

## 关联卡片

- [[Credit Risk]]
- [[DVA]]
- [[EAD]]
- [[Netting]]
- [[Basel Accords]]

## 边界与相关概念

本卡片只保留该概念的定义、记号与最小直觉；具体估计步骤、证明或完整应用应放在对应的 `procedure`、`proof` 或 `framework` 卡片中。

相关卡片：[[DVA]]、[[Netting]]、[[17_OTC衍生产品市场的监管]]、[[19_违约风险]]、[[Credit Risk]]、[[EAD]]、[[Basel Accords]]。
