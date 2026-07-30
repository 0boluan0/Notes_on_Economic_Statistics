---
aliases:
- Cook's Distance
- Cook distance
- Cook距离
- 库克距离
tags:
- system
- econometrics
---
# Cook's Distance

## 诊断目标

Cook's Distance 用来识别对回归系数有强影响的观测点。

## 核心公式

常见形式：

$$
D_i=\frac{\hat u_i^2}{k\hat\sigma^2}\cdot\frac{h_{ii}}{(1-h_{ii})^2}
$$

其中 $\hat u_i$ 是残差，$h_{ii}$ 是杠杆值，$k$ 是参数个数。

## 判断

- $D_i>0.5$：值得检查。
- $D_i>1$：强影响点信号。

阈值只是经验规则，必须结合数据背景判断。

## 诊断流程

1. 计算所有观测的 Cook's Distance。
2. 找到高影响点。
3. 检查是否为录入错误、极端但真实观测、模型设定错误。
4. 做包含/排除该点的敏感性分析。

## 易混点

- Cook's Distance 是计量里的影响点诊断；Basel 资本监管中的库克比率是 [[Cooke Ratio]]。
- 高影响点不一定该删除，可能是有信息的真实样本。
- 只报告删除后的结果不透明，应报告敏感性。

## 来自课程位置

- [[04_模型设定]]

## 关联卡片

## 复现规范

记录输入数据与样本区间、模型/检验设定、阈值或显著性水平、软件版本和处理决策；保留诊断图表与原始输出，使“发现—判断—修正”链条可复核。

- [[Outlier Detection]]
- [[Residual]]
- [[Robust Regression]]
