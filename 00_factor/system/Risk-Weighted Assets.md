---
aliases:
- Risk-Weighted Assets
- RWA
- 风险加权资产
tags:
- system
- banking
- regulation
---
# Risk-Weighted Assets

## 诊断目标

RWA 把不同风险程度的资产折算成资本监管分母，避免只看账面总资产。

## 标准法公式

$$
RWA=\sum_i Exposure_i\times Risk\ Weight_i
$$

Basel I 常见权重包括 0%、20%、50%、100%。

## IRB 口径

内部评级法中常见折算关系：

$$
RWA=12.5\times K\times EAD
$$

其中 $12.5=1/8\%$，$K$ 是基于 [[PD]]、[[LGD]]、相关性和期限调整得到的资本系数。

## 诊断流程

1. 判断资产或表外项目属于哪类暴露。
2. 确定风险权重或 IRB 参数。
3. 对衍生品先算信用等价额或 [[EAD]]，再乘权重。
4. 加总得到 RWA。
5. 用 RWA 计算 [[Basel Capital Adequacy Ratio]]。

## 常见风险点

- 表外承诺没有用 CCF 转换。
- 衍生品没有考虑潜在未来暴露和 [[Netting]]。
- 风险权重套错资产类别。
- 监管 RWA 和内部经济资本差异过大。

## 来自课程位置

- [[15_《巴塞尔协议I II》和 偿付能力法案II]]
- [[16_巴塞尔协议]]

## 关联卡片

- [[Cooke Ratio]]
- [[PD]]
- [[LGD]]
- [[EAD]]
- [[Basel Capital Adequacy Ratio]]
