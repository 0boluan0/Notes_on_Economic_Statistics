---
aliases:
- Basel Capital Adequacy Ratio
- 资本充足率
- 巴塞尔资本充足率
tags:
- system
- banking
- regulation
---
# Basel Capital Adequacy Ratio

## 诊断目标

判断银行资本是否足以覆盖风险加权资产，以及资本质量是否符合监管要求。

## 输入

- CET1、AT1、Tier 2 资本。
- [[Risk-Weighted Assets|RWA]]。
- 适用监管口径和资本缓冲要求。

## 核心公式

总资本充足率：

$$
CAR=\frac{Tier\ 1+Tier\ 2}{RWA}
$$

Tier 1 比率：

$$
Tier\ 1\ Ratio=\frac{Tier\ 1}{RWA}
$$

CET1 比率：

$$
CET1\ Ratio=\frac{CET1}{RWA}
$$

## Basel III 快速阈值

| 指标 | 最低要求 | 加资本留存缓冲后 |
| --- | --- | --- |
| CET1 | 4.5% | 7.0% |
| Tier 1 | 6.0% | 8.5% |
| Total Capital | 8.0% | 10.5% |

## 诊断流程

1. 先确认 RWA 是否算对，见 [[Risk-Weighted Assets]]。
2. 再看总资本是否超过 8%。
3. 再拆资本质量：CET1、AT1、Tier 2。
4. 最后检查资本缓冲、杠杆率和流动性约束。

## 常见风险点

- RWA 被低估，导致资本充足率看起来过高。
- Tier 2 占比过高，资本质量不足。
- 资本充足率达标但 [[Leverage Ratio]] 不达标。
- 压力情景下资本被快速吃掉，见 [[Stress Testing]]。

## 来自课程位置

- [[16_巴塞尔协议]]
- [[15_《巴塞尔协议I II》和 偿付能力法案II]]

## 关联卡片

- [[Cooke Ratio]]
- [[Risk-Weighted Assets]]
- [[Tier 1 Capital Ratio]]
- [[Tier 2 Capital Ratio]]
- [[Leverage Ratio]]
