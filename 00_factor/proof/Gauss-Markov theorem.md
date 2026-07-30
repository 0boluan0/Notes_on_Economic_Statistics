---
aliases:
- Gauss-Markov theorem
- Gauss-Markov 定理
- 高斯-马尔可夫定理
- OLS BLUE
tags:
- proof
- econometrics
---
# Gauss-Markov theorem

## 假设

线性模型：

$$
y=X\beta+u
$$

满足：

- $X$ 满列秩。
- $E[u\mid X]=0$。
- $Var(u\mid X)=\sigma^2 I$。

## 命题

在所有线性无偏估计量中，OLS 估计量：

$$
\hat\beta=(X'X)^{-1}X'y
$$

具有最小方差，即是 BLUE（Best Linear Unbiased Estimator）。

## 推导链

任取一个线性估计量：

$$
\tilde\beta=Cy
$$

无偏要求：

$$
E[\tilde\beta\mid X]=CX\beta=\beta
$$

因此：

$$
CX=I
$$

把 $C$ 写成 OLS 权重加一个偏离项：

$$
C=(X'X)^{-1}X'+D
$$

由 $CX=I$ 得：

$$
DX=0
$$

方差为：

$$
Var(\tilde\beta\mid X)=\sigma^2CC'
$$

代入 $C$ 并利用 $DX=0$：

$$
Var(\tilde\beta\mid X)
=\sigma^2(X'X)^{-1}+\sigma^2DD'
$$

由于 $DD'$ 半正定：

$$
Var(\tilde\beta\mid X)-Var(\hat\beta\mid X)=\sigma^2DD'\succeq 0
$$

## 结论

OLS 在同方差、无自相关和外生性条件下，是所有线性无偏估计量中方差最小的估计量。

## 适用边界

- BLUE 只在“线性、无偏估计量”这个比较集合内成立，不表示 OLS 在所有可能的非线性或有偏估计量中均方误差最小。
- 正态性不是 Gauss–Markov 定理的必要条件；正态性主要用于有限样本 t/F 精确推断。
- 若 $X$ 不是满列秩，公式不可逆；若存在异方差或自相关，需改用稳健协方差或 GLS，不能宣称仍为 BLUE。

## 复现规范

复现时检查设计矩阵秩、误差协方差结构和条件外生性，并明确报告使用经典、White HC 或 HAC 协方差；不要仅凭 $R^2$ 判断定理条件。

## 边界

- 若有 [[Heteroskedasticity]] 或 [[Autocorrelation]]，OLS 可仍无偏/一致，但不再有效，标准误也需调整。
- 若有 [[Endogeneity]]，无偏性和一致性都会出问题。

## 关联卡片

- [[OLS Estimator]]
- [[OLS unbiasedness]]
- [[Heteroskedasticity]]
- [[Autocorrelation]]
- [[White Robust Standard Errors]]
