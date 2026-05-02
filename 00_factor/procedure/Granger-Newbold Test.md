---
aliases:
- Granger-Newbold Test
- Granger Newbold Test
- GN Test
- Granger-Newbold检验
- 格兰杰-纽博尔德检验
tags:
- procedure
- 时间序列
---
# Granger-Newbold Test

## 这张卡什么时候用

当你比较两个模型的预测误差方差，并且允许两个模型的同期预测误差相关时，用 Granger-Newbold 检验。

## 输入

- 模型 1 的预测误差 $e_{1i}$；
- 模型 2 的预测误差 $e_{2i}$；
- 同一段样本外预测期 $i=1,\ldots,H$。

## 输出

- 两个模型 MSPE 是否有显著差异；
- 若有差异，选择误差方差更小的模型。

## Step 1. 构造和差序列

$$
x_i=e_{1i}+e_{2i}.
$$

## Step 2. 构造差分序列

$$
z_i=e_{1i}-e_{2i}.
$$

若两个模型预测误差方差相同，则 $x_i$ 和 $z_i$ 的相关性应接近 0。

## Step 3. 计算相关系数

计算样本相关系数：
$$
r_{xz}.
$$

## Step 4. 构造 t 统计量

$$
T=\frac{r_{xz}}{\sqrt{(1-r_{xz}^2)/(H-1)}}.
$$

在课程假设下，它近似服从自由度 $H-1$ 的 t 分布。

## Step 5. 写结论

- 显著：两个模型预测误差方差不同，选 MSPE 更小的模型。
- 不显著：MSPE 更小的一方未必稳定优于另一方。

## 常见错误

- 忘记两个模型必须在同一预测样本上比较。
- 只比较 MSPE 大小，不判断差异是否显著。
- 把 Granger-Newbold 和 [[Granger Causality Test]] 混在一起。前者是预测误差比较，不是因果检验。

## 来自课程位置

- [[03_平稳时间序列模型#5.5. 预测效果评估|时间序列 03：预测误差比较]]

## 关联卡片

- [[Forecast Evaluation]]
- [[Diebold-Mariano Test]]
- [[ARMA]]

