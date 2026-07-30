---
aliases:
- Weighted Least Squares Estimation
- WLS Estimation
- FGLS Estimation
- 加权最小二乘估计
tags:
- procedure
- econometrics
type: procedure
---
# Weighted Least Squares Estimation

## 这张卡什么时候用

确认存在异方差，并且有可信权重或可估计方差函数时，用 WLS/FGLS 做估计。

## 输入

- 原模型 $y=X\beta+u$。
- 方差形式或方差代理变量。
- 权重 $w_i$，或可用于估计权重的数据。

## 输出

- WLS/FGLS 系数。
- 对应标准误和诊断结果。

## Step 1：先跑 OLS

用 [[OLS Estimation Steps]] 估计原模型，保存残差 $\hat u_i$。

## Step 2：判断权重来源

- 若已知 $Var(u_i)=\sigma_i^2$，直接取 $w_i=1/\sigma_i^2$。
- 若未知，用残差平方对方差决定变量建模，得到 $\hat\sigma_i^2$。

## Step 3：构造权重

$$
w_i=\frac{1}{\hat\sigma_i^2}
$$

权重必须为正。

## Step 4：做加权回归

最小化：

$$
\sum_i w_i(y_i-x_i'\beta)^2
$$

或等价地对 $\sqrt{w_i}y_i$ 和 $\sqrt{w_i}x_i$ 做 OLS。

## Step 5：复查诊断

重新检查残差图、White Test 和结果稳健性。若权重模型不可靠，优先报告 [[White Robust Standard Errors]]。

## 常见错误

- 把权重设为方差本身，而不是方差倒数。
- 使用负权重或零权重。
- 没有说明权重从哪里来。
- 权重拟合过度，导致小样本不稳。

## 来自课程位置

- [[07_异方差]]

## 关联卡片

- [[Weighted Least Squares]]
- [[FGLS]]
- [[Heteroscedasticity Diagnosis]]
- [[White Robust Standard Errors]]
