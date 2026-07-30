---
aliases:
- Yule-Walker equations
- Yule-Walker 方程
- Yule-Walker方程
- Yule Walker equations
tags:
- proof
- 时间序列
---
# Yule-Walker equations

## 假设

考虑零均值平稳 AR(p)：
$$
y_t=\phi_1y_{t-1}+\cdots+\phi_py_{t-p}+\varepsilon_t,
$$
其中 $\varepsilon_t$ 是白噪声，且与过去的 $y$ 不相关。

令
$$
\gamma_k=\operatorname{Cov}(y_t,y_{t-k}).
$$

## 推导链

对 AR(p) 方程两边乘以 $y_{t-k}$，再取期望：
$$
E(y_ty_{t-k})
=\sum_{i=1}^{p}\phi_iE(y_{t-i}y_{t-k})
+E(\varepsilon_ty_{t-k}).
$$

当 $k\geq1$ 时，$y_{t-k}$ 属于过去信息，和当前创新 $\varepsilon_t$ 不相关，所以
$$
E(\varepsilon_ty_{t-k})=0.
$$

利用平稳性：
$$
E(y_ty_{t-k})=\gamma_k,
\qquad
E(y_{t-i}y_{t-k})=\gamma_{k-i}.
$$

因此
$$
\gamma_k=\sum_{i=1}^{p}\phi_i\gamma_{k-i},
\qquad k\geq1.
$$

两边除以 $\gamma_0$，得到自相关版本：
$$
\rho_k=\sum_{i=1}^{p}\phi_i\rho_{k-i}.
$$

## $k=0$ 的方程

当 $k=0$ 时，
$$
E(\varepsilon_ty_t)=E(\varepsilon_t^2)=\sigma^2.
$$

所以
$$
\gamma_0=\sum_{i=1}^{p}\phi_i\gamma_i+\sigma^2.
$$

## 结论

Yule-Walker 方程把 AR 参数和 ACF 联系起来。

它可以用来：

- 已知 AR 参数，递推 ACF；
- 用样本 ACF 估计 AR 参数；
- 理解 AR 模型为什么 ACF 拖尾。

## 适用边界

- $k\ge1$ 时创新与过去信息不相关是关键；若创新存在条件异方差，二阶协方差推导仍可能成立，但估计推断需另行稳健处理。
- 递推式中的 $\gamma_{k-i}$ 对负下标使用 $\gamma_{-j}=\gamma_j$；实现时要显式处理索引，避免把负滞后误当成零。
- 样本 Yule–Walker 估计在近单位根、高阶模型或样本较短时可能不稳定，应与 OLS/最大似然估计和残差诊断比较。

## 复现规范

记录 AR 阶数、均值处理、ACF 估计口径、样本区间与求解方式；报告估计参数、根是否位于单位圆内，以及残差 ACF 和 Ljung–Box 结果。

## 来自课程位置

- [[03_平稳时间序列模型#3. ACF|时间序列 03：Yule-Walker 与 ACF]]

## 关联卡片

- [[Autoregressive Model]]
- [[ARMA]]
- [[Autocorrelation Function]]
- [[Partial Autocorrelation Function]]
- [[White Noise]]

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
