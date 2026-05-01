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
