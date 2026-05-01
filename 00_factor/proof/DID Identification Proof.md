---
aliases:
- DID Identification Proof
- DID识别证明
- Difference-in-Differences proof
- Parallel trends proof
- 双重差分识别
tags:
- proof
- econometrics
- causal-inference
---
# DID Identification Proof

## 假设

处理组 $G=1$，对照组 $G=0$，处理后 $Post=1$。

观察结果：

$$
Y=D Y(1)+(1-D)Y(0)
$$

关键假设是 [[Parallel Trends]]：

$$
E[Y_{post}(0)-Y_{pre}(0)\mid G=1]
=E[Y_{post}(0)-Y_{pre}(0)\mid G=0]
$$

并假设无预期效应、无溢出效应。

## 推导链

DID 估计量：

$$
\begin{aligned}
DID
&=[E(Y\mid G=1,post)-E(Y\mid G=1,pre)]\\
&\quad-[E(Y\mid G=0,post)-E(Y\mid G=0,pre)]
\end{aligned}
$$

处理组政策后观察到 $Y(1)$，政策前观察到 $Y(0)$：

$$
E(Y\mid G=1,post)-E(Y\mid G=1,pre)
=E[Y_{post}(1)-Y_{pre}(0)\mid G=1]
$$

加减 $E[Y_{post}(0)\mid G=1]$：

$$
=E[Y_{post}(1)-Y_{post}(0)\mid G=1]
+E[Y_{post}(0)-Y_{pre}(0)\mid G=1]
$$

由平行趋势，第二项等于对照组未处理趋势：

$$
E[Y_{post}(0)-Y_{pre}(0)\mid G=0]
$$

而对照组未被处理，所以可观察到：

$$
E(Y\mid G=0,post)-E(Y\mid G=0,pre)
$$

相减后共同趋势抵消，留下：

$$
DID=E[Y_{post}(1)-Y_{post}(0)\mid G=1]
$$

## 结论

在平行趋势、无预期和无溢出条件下，DID 识别处理组在处理后的平均处理效应，即 [[ATT]]。

## 关联卡片

- [[DID]]
- [[Parallel Trends]]
- [[ATT]]
- [[DID Diagnostics]]
