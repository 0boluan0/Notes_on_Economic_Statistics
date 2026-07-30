---
aliases:
- VaR Parametric Method
- VaR Parameter Calculation
- 方差-协方差法VaR
- VaR参数法计算
tags:
- procedure
- 金融风险
- VaR
type: procedure
---
# VaR Parametric Method

## 这张卡什么时候用

当组合主要是线性产品，且可以接受正态/参数分布假设时，用参数法快速计算 VaR。

## 输入

- 组合市值或头寸；
- 波动率；
- 协方差矩阵或相关矩阵；
- 置信水平 $\alpha$；
- 持有期 $T$。

## 输出

- 给定置信水平和持有期下的 VaR 金额。

## Step 1. 计算组合波动率

单资产：
$$
\sigma_P=V\sigma.
$$

多资产：
$$
\sigma_P^2=w^T\Sigma w.
$$

如果 $w$ 是金额头寸，结果直接是金额波动。

## Step 2. 查分位数

常用：

- 95%：$z_\alpha\approx1.645$；
- 99%：$z_\alpha\approx2.33$；
- 97.5%：$z_\alpha\approx1.96$。

## Step 3. 计算单期 VaR

短期均值忽略时：
$$
VaR_\alpha=z_\alpha\sigma_P.
$$

若保留均值，要明确损失定义和符号，不要把收益分位数和损失分位数混在一起。

## Step 4. 调整持有期

若每日收益独立同分布：
$$
VaR_{\alpha,T}=VaR_{\alpha,1}\sqrt{T}.
$$

如果存在自相关或波动聚集，平方根规则可能不可靠。

## Step 5. 处理非线性产品

期权等非线性产品不能只用线性 delta。

可选：

- [[Delta Approximation]]；
- [[Delta-Gamma Approximation]]；
- [[Cornish-Fisher Expansion]]；
- 直接转 [[Monte Carlo Simulation VaR]]。

## 常见错误

- 把收益标准差和金额标准差混用。
- 忘记相关性，直接把单资产 VaR 相加。
- 对期权组合只用线性近似。
- 在厚尾和危机相关性变化下仍机械用正态 VaR。

## 来自课程位置

- [[14_VaR参数法和模拟法|金融风险管理 14：VaR 参数法]]

## 关联卡片

- [[Variance-Covariance Method]]
- [[VaR]]
- [[Historical Simulation VaR]]
- [[Monte Carlo Simulation VaR]]
- [[Delta-Gamma Approximation]]

## 课程笔记反链

<!-- course-backlinks-panel -->
```dataview
LIST FROM ""
WHERE (
  contains(file.path, "01_Math/") OR
  contains(file.path, "02_Economy/") OR
  contains(file.path, "03_Computer_Science/")
) AND contains(file.outlinks, this.file.link)
SORT file.mtime DESC
```
