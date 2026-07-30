---
aliases:
- Greeks Hedging Map
- Greeks对冲框架
- 希腊值对冲地图
tags:
- framework
- derivatives
- risk-management
type: framework
---
# Greeks Hedging Map

## 先记一句话

Greeks 对冲的顺序是先控制一阶方向风险，再处理二阶、波动率和时间损耗；不同 Greek 需要不同工具。

## 什么时候用

当题目问“该怎么对冲一个期权组合”或“为什么 Delta 中性仍会亏损”时，用这张卡组织判断。

## 题型识别

| 问题 | 关注 Greek | 常见工具 |
| --- | --- | --- |
| 标的价格小幅变化 | [[Delta]] | 标的资产、期货、远期 |
| Delta 会快速变化 | [[Gamma]] | 其他期权 |
| 隐含波动率变化 | [[Vega]] | 其他期权、波动率产品 |
| 时间流逝损耗 | [[Theta]] | 组合结构调整 |
| 利率变化 | [[Rho]] | 利率产品、期限结构工具 |

## 为什么这样看

标的资产只能改变 Delta，不能有效改变 Gamma 或 Vega。要管理非线性风险，通常必须引入期权类工具。

## 边界条件

- Delta 中性只在当前点附近有效，市场动了要再平衡。
- Gamma/Vega 中性通常需要至少两个不同期权工具。
- Theta 很难“对冲掉”，更多是持仓成本和策略收益来源。

## 失败模式

- 只做 Delta hedge，却忽略短期期权的高 Gamma。
- 用同一标的现货试图对冲 Vega。
- 不区分“风险敞口为 0”和“未来不会亏钱”。

## 来自课程位置

- [[08_操作员如何管理风险暴露]]

## 关联卡片

- [[Delta Hedging]]
- [[Greeks Calculation]]
- [[Delta-Gamma Approximation]]
- [[Linear Products]]
- [[Nonlinear Products]]
