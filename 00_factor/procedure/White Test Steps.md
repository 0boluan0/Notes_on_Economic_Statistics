---
aliases:
- White Test Steps
- White检验步骤
- White异方差检验步骤
tags:
- procedure
- econometrics
---
# White Test Steps

## 这张卡什么时候用

需要按手算或软件输出复现 White 异方差检验时使用。

## 输入

- 原回归模型。
- 原回归 OLS 残差 $\hat u_i$。
- 样本量 $n$。
- 辅助回归形式：完整形式或简化形式。

## 输出

- $LM=nR^2_{aux}$。
- 自由度 $q$。
- 是否拒绝同方差原假设。

## Step 1：估计原模型

用 [[OLS Estimation Steps|OLS]] 估计原回归，保存残差 $\hat u_i$。

## Step 2：构造残差平方

$$
\hat u_i^2
$$

这是辅助回归的因变量。

## Step 3：构造辅助回归

完整形式包括原解释变量、平方项和交叉项：

$$
\hat u_i^2=\alpha_0+\sum_j\alpha_j x_{ji}+\sum_j\gamma_jx_{ji}^2+\sum_{j<l}\delta_{jl}x_{ji}x_{li}+v_i
$$

变量太多时可使用简化形式，只保留解释变量和平方项。

## Step 4：计算统计量

得到辅助回归 $R^2_{aux}$：

$$
LM=nR^2_{aux}
$$

## Step 5：确定自由度并判断

自由度为辅助回归中除常数外的解释变量个数 $q$。

- $p\le 0.05$：拒绝同方差。
- $p>0.05$：没有足够证据拒绝同方差。

## 常见错误

- 把原回归的 $R^2$ 当成辅助回归的 $R^2$。
- 自由度把常数项也算进去。
- 变量很多还用完整形式，导致检验功效很低。

## 来自课程位置

- [[07_异方差]]

## 关联卡片

- [[White Test]]
- [[Heteroskedasticity]]
- [[White Robust Standard Errors]]
- [[Weighted Least Squares Estimation]]
