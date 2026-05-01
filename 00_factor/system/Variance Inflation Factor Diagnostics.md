---
aliases:
- VIF Diagnosis
- Variance Inflation Factor diagnostics
- 方差膨胀因子诊断
tags:
- system
- econometrics
---
# Variance Inflation Factor Diagnostics

## 诊断目标

用 VIF 找出哪些解释变量的系数方差因多重共线性而被显著放大，并决定是否处理。

## 诊断流程

1. 对每个解释变量 $x_j$，用其他解释变量回归，得到 $R_j^2$。
2. 计算：

$$
VIF_j=\frac{1}{1-R_j^2}
$$

3. 按阈值初筛：

| VIF | 解释 |
| --- | --- |
| 1-5 | 通常可接受 |
| 5-10 | 需要关注 |
| 10+ | 严重共线性信号 |

## 处理判断

- 若研究目标是预测，且预测稳定，未必需要删变量。
- 若目标是解释单个系数，VIF 高会削弱结论。
- 若变量理论上必须保留，不要只因 VIF 高就删除。

## 常见处理

- 合并高度相关变量。
- 删除理论次要变量。
- 增加样本或扩大变量取值范围。
- 使用 [[Ridge Regression]] 或 [[PCA]] 作预测/降维。

## 来自课程位置

- [[06_多重共线性]]

## 关联卡片

- [[Variance Inflation Factor]]
- [[Multicollinearity]]
- [[Condition Index]]
