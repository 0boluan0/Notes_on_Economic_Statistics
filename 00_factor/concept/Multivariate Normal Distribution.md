---
aliases:
- Multivariate Normal Distribution
- MVN
- 多元正态分布
tags:
  - concept
  - multivariate statistics
---
# Multivariate Normal Distribution

>[!note] 一句话记忆
> 多元正态分布用均值向量和协方差矩阵描述多个连续变量的联合正态结构。

## 它是什么

若 $X$ 是 $p$ 维随机向量，
$$
X\sim N_p(\mu,\Sigma),
$$
则密度为
$$
f(x)=
\frac{1}{(2\pi)^{p/2}|\Sigma|^{1/2}}
\exp\left[-\frac12(x-\mu)'\Sigma^{-1}(x-\mu)\right].
$$

其中：

- $\mu$ 是 [[Mean Vector]]；
- $\Sigma$ 是 [[Covariance Matrix]]；
- $(x-\mu)'\Sigma^{-1}(x-\mu)$ 是平方 [[Mahalanobis Distance]]。

## 解决什么判断

- 多个连续变量是否可用联合正态近似。
- 线性组合、仿射变换和条件分布是否仍然正态。
- Hotelling $T^2$、Wishart 分布和多元正态性检查的理论来源是什么。

## 最小例子

二维情形：
$$
\begin{bmatrix}X\\Y\end{bmatrix}
\sim N_2\left(
\begin{bmatrix}\mu_X\\\mu_Y\end{bmatrix},
\begin{bmatrix}
\sigma_X^2&\rho\sigma_X\sigma_Y\\
\rho\sigma_X\sigma_Y&\sigma_Y^2
\end{bmatrix}
\right).
$$

## 核心性质

1. 线性组合：
   $$
   a'X\sim N(a'\mu,a'\Sigma a).
   $$
2. 仿射变换：
   $$
   AX+b\sim N(A\mu+b,A\Sigma A').
   $$
3. 二次型：
   $$
   (X-\mu)'\Sigma^{-1}(X-\mu)\sim\chi_p^2.
   $$
4. 零协方差与独立：
   在多元正态下，零协方差等价于独立。

## 易混点

- 每个边际变量正态，不保证联合分布是多元正态。
- 密度公式要求 $\Sigma$ 正定。
- 多元正态的等密度曲线是椭圆或椭球，不是普通圆，除非协方差矩阵是标量倍的单位矩阵。

## 来自课程位置

- [[04_多元正态分布The Multivariate Normal Distribution#1. 第4章：多元正态分布（The Multivariate Normal Distribution）|第4章 多元正态分布]]

## 关联卡片

- [[Bivariate Normal Distribution]]
- [[Conditional Multivariate Normal Distribution]]
- [[Wishart Distribution]]
- [[Mahalanobis Distance]]
- [[Hotelling T2 Test]]
- [[Multivariate Normality Check]]

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
