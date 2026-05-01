---
aliases:
- Variance-Covariance Method
- Parametric VaR Method
- 方差-协方差法
- 参数法
tags:
- concept
- 金融风险
- VaR
---
# Variance-Covariance Method

## 先记一句话

方差-协方差法就是：**假设收益分布形状已知，通常是正态，然后用均值、波动率和相关性直接算 VaR**。

它是最快的 VaR 方法。

## 它是什么

若组合损失近似正态，短期均值可忽略，则：
$$
VaR_\alpha\approx z_\alpha\sigma_P.
$$

多资产组合的方差：
$$
\sigma_P^2=w^T\Sigma w.
$$

其中 $\Sigma$ 是协方差矩阵。

## 它解决什么判断

适合：

- 线性产品组合；
- 日常快速风险监控；
- 波动率和相关性比较稳定；
- 不需要完整重估组合的场景。

具体计算步骤见 [[VaR Parametric Method]]。

## 主要边界

- 正态假设会低估厚尾风险。
- 含期权等非线性产品时，线性近似可能不够。
- 危机时相关性跳变，历史协方差矩阵可能失效。

## 常见误区

- 参数法不是“精确法”；它的速度来自强假设。
- 多日 VaR 的 $\sqrt{T}$ 缩放依赖独立同分布或弱相关假设。
- 非线性产品至少要考虑 [[Delta-Gamma Approximation]] 或模拟法。

## 来自课程位置

- [[12_VAR风险#2.1 方差-协方差法（正态分布假设法）|金融风险管理 12：方差-协方差法]]
- [[14_VaR参数法和模拟法|金融风险管理 14：VaR 参数法]]

## 关联卡片

- [[VaR]]
- [[VaR Parametric Method]]
- [[Historical Simulation Method]]
- [[Monte Carlo Simulation Method]]
- [[Delta-Gamma Approximation]]
- [[Cornish-Fisher Expansion]]

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
