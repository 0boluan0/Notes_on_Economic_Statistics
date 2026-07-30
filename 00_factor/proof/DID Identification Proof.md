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

## 适用边界

- 证明针对两组、两期且处理时间统一的 DID；多期错位处理不能仅凭一个 TWFE 系数套用该结论。
- 平行趋势是关于未处理潜在结果的反事实假设，无法由政策后的显著性检验直接证明；处理前图形和事件研究只能提供支持或反证。
- 若处理组存在提前反应、对照组受到溢出，或两组在政策期发生不同的样本选择，识别式中的反事实分解失效。

## 复现规范

报告组别定义、政策生效日、结果变量窗口、聚类层级和处理前事件研究；同时保存原始样本筛选规则与安慰剂设定，使 ATT 的口径可复核。

## 关联卡片

- [[DID]]
- [[Parallel Trends]]
- [[ATT]]
- [[DID Diagnostics]]
