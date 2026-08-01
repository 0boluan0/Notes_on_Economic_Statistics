# 1. 第6章：比较多个均值向量（Comparisons of Multivariate Mean Vectors）
<!-- bilingual-en:start -->
*1. Chapter 6: Comparisons of Multivariate Mean Vectors*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 第 5 章检验一个总体均值向量，第 6 章比较两个或多个总体均值向量。两组问题优先识别“配对”还是“独立”；多组问题进入 one-way MANOVA。
> <!-- bilingual-en:start -->
> Chapter 5 tests one population mean vector; Chapter 6 compares the mean vectors of two or more populations. For two groups, first distinguish paired from independent samples. With several groups, use one-way MANOVA.
> <!-- bilingual-en:end -->

## 1.1. 回忆用
<!-- bilingual-en:start -->
*1.1. Quick Review*
<!-- bilingual-en:end -->

本章核心：
<!-- bilingual-en:start -->
The chapter has three core cases:
<!-- bilingual-en:end -->

1. 配对样本：先构造差值向量，再做单样本 Hotelling $T^2$。
2. 两独立样本：协方差相等时使用 pooled covariance。
3. 多个总体：使用单因子 MANOVA，核心统计量是 Wilks Lambda。
<!-- bilingual-en:start -->

&nbsp;
**1.** Paired samples: form difference vectors and apply a one-sample Hotelling $T^2$ test.<br>
**2.** Two independent samples: use a pooled covariance matrix when the population covariance matrices are equal.<br>
**3.** Several populations: use one-way MANOVA, with Wilks' Lambda as the central statistic.<br>
<!-- bilingual-en:end -->

>[!attention] 题型边界
> 若 $\Sigma_1\neq\Sigma_2$，不能直接使用 pooled covariance 两样本 $T^2$。
> <!-- bilingual-en:start -->
> If $\Sigma_1\neq\Sigma_2$, the pooled-covariance two-sample $T^2$ test cannot be used directly.
> <!-- bilingual-en:end -->

## 1.2. 配对样本均值向量比较
<!-- bilingual-en:start -->
*1.2. Comparing Mean Vectors with Paired Samples*
<!-- bilingual-en:end -->

### 1.2.1. 场景与符号
<!-- bilingual-en:start -->
*1.2.1. Setting and Notation*
<!-- bilingual-en:end -->

同一对象在两种处理下被测量多个变量。
<!-- bilingual-en:start -->
Several variables are measured on the same object under two treatments.
<!-- bilingual-en:end -->

记第 $j$ 个样本在两种处理下的观测为
<!-- bilingual-en:start -->
Denote the two observations on sample unit $j$ by
<!-- bilingual-en:end -->
$$
X_{1j},\qquad X_{2j}.
$$

### 1.2.2. 差值向量
<!-- bilingual-en:start -->
*1.2.2. Difference Vectors*
<!-- bilingual-en:end -->

定义差值向量：
<!-- bilingual-en:start -->
Define the difference vector:
<!-- bilingual-en:end -->
$$
D_j=X_{1j}-X_{2j}.
$$

总体差值均值为
<!-- bilingual-en:start -->
The population mean difference is
<!-- bilingual-en:end -->
$$
E(D_j)=\delta.
$$

检验：
<!-- bilingual-en:start -->
Test:
<!-- bilingual-en:end -->
$$
H_0:\delta=0
\quad\text{vs.}\quad
H_1:\delta\neq0.
$$

### 1.2.3. Hotelling T² 检验
<!-- bilingual-en:start -->
*1.2.3. Hotelling T² Test*
<!-- bilingual-en:end -->

计算
<!-- bilingual-en:start -->
Compute
<!-- bilingual-en:end -->
$$
\bar D=\frac1n\sum_{j=1}^nD_j,
$$
$$
S_d=\frac1{n-1}\sum_{j=1}^n(D_j-\bar D)(D_j-\bar D)'.
$$

统计量：
<!-- bilingual-en:start -->
The statistic is:
<!-- bilingual-en:end -->
$$
T^2=n\bar D'S_d^{-1}\bar D.
$$

在 $H_0$ 下：
<!-- bilingual-en:start -->
Under $H_0$:
<!-- bilingual-en:end -->
$$
\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.
$$

### 1.2.4. 同时置信区间
<!-- bilingual-en:start -->
*1.2.4. Simultaneous Confidence Intervals*
<!-- bilingual-en:end -->

第 $i$ 个均值差 $\delta_i$ 的同时置信区间：
<!-- bilingual-en:start -->
The simultaneous confidence interval for mean difference $\delta_i$ is:
<!-- bilingual-en:end -->
$$
\bar d_i\pm
\sqrt{\frac{p(n-1)}{n-p}F_{p,n-p}(1-\alpha)}
\sqrt{\frac{s_{d,ii}}{n}}.
$$

### 1.2.5. 案例：商业实验室 vs 国家实验室
<!-- bilingual-en:start -->
*1.2.5. Example: Commercial versus State Laboratory*
<!-- bilingual-en:end -->

旧笔记中的例子比较两个实验室对 BOD 和 SS 的测量差异。
<!-- bilingual-en:start -->
The example in the old note compares BOD and SS measurements from two laboratories.
<!-- bilingual-en:end -->

差值向量为
<!-- bilingual-en:start -->
The difference vector is
<!-- bilingual-en:end -->
$$
D_j=
\begin{bmatrix}
\text{Commercial BOD}-\text{State BOD}\\
\text{Commercial SS}-\text{State SS}
\end{bmatrix}.
$$

样本计算得到
<!-- bilingual-en:start -->
The sample calculations give
<!-- bilingual-en:end -->
$$
\bar d=
\begin{bmatrix}
-9.36\\
13.27
\end{bmatrix},
\qquad
S_d=
\begin{bmatrix}
199.26&88.38\\
88.38&418.61
\end{bmatrix}.
$$

计算得
<!-- bilingual-en:start -->
This yields
<!-- bilingual-en:end -->
$$
T^2=13.6.
$$

在 $\alpha=0.05$ 下，临界值约为 $9.47$，因此拒绝 $H_0$。
<!-- bilingual-en:start -->
At $\alpha=0.05$, the critical value is approximately $9.47$, so $H_0$ is rejected.
<!-- bilingual-en:end -->

>[!note] 解读
> 即使单个分量的同时区间可能包含 0，联合检验仍可能拒绝。这说明两个变量合在一起呈现了显著差异。
> <!-- bilingual-en:start -->
> The joint test may reject even when the simultaneous interval for an individual component contains zero. The two variables can be jointly different even when neither is decisive on its own.
> <!-- bilingual-en:end -->

## 1.3. 参数线性变换检验
<!-- bilingual-en:start -->
*1.3. Testing Linear Transformations of Parameters*
<!-- bilingual-en:end -->

有时比较的不是原始均值，而是线性组合：
<!-- bilingual-en:start -->
Sometimes the target is not the original mean vector but a linear combination:
<!-- bilingual-en:end -->
$$
\psi=C\mu.
$$

例如
<!-- bilingual-en:start -->
For example,
<!-- bilingual-en:end -->
$$
C=
\begin{bmatrix}
1&0&-1&0\\
0&1&0&-1
\end{bmatrix}
$$
可以比较两组变量的差异。
<!-- bilingual-en:start -->
can compare differences between two sets of variables.
<!-- bilingual-en:end -->

检验统计量仍使用 Hotelling 型：
<!-- bilingual-en:start -->
The test statistic retains the Hotelling form:
<!-- bilingual-en:end -->
$$
T^2=n(C\bar X-\psi_0)'(CSC')^{-1}(C\bar X-\psi_0).
$$

如果 $C$ 的秩为 $m$，则
<!-- bilingual-en:start -->
If $C$ has rank $m$, then
<!-- bilingual-en:end -->
$$
\frac{n-m}{m(n-1)}T^2\sim F_{m,n-m}.
$$

## 1.4. 两个独立总体均值向量比较
<!-- bilingual-en:start -->
*1.4. Comparing the Mean Vectors of Two Independent Populations*
<!-- bilingual-en:end -->

### 1.4.1. 协方差矩阵相等
<!-- bilingual-en:start -->
*1.4.1. Equal Covariance Matrices*
<!-- bilingual-en:end -->

设两组样本独立，且
<!-- bilingual-en:start -->
Suppose the two samples are independent and
<!-- bilingual-en:end -->
$$
\Sigma_1=\Sigma_2=\Sigma.
$$

检验：
<!-- bilingual-en:start -->
Test:
<!-- bilingual-en:end -->
$$
H_0:\mu_1-\mu_2=0.
$$

pooled covariance 为
<!-- bilingual-en:start -->
The pooled covariance matrix is
<!-- bilingual-en:end -->
$$
S_p=\frac{(n_1-1)S_1+(n_2-1)S_2}{n_1+n_2-2}.
$$

统计量：
<!-- bilingual-en:start -->
The statistic is:
<!-- bilingual-en:end -->
$$
T^2=\frac{n_1n_2}{n_1+n_2}
(\bar X_1-\bar X_2)'S_p^{-1}(\bar X_1-\bar X_2).
$$

F 转换：
<!-- bilingual-en:start -->
Its F transformation is:
<!-- bilingual-en:end -->
$$
\frac{n_1+n_2-p-1}{p(n_1+n_2-2)}T^2
\sim F_{p,n_1+n_2-p-1}.
$$

### 1.4.2. 协方差矩阵不等
<!-- bilingual-en:start -->
*1.4.2. Unequal Covariance Matrices*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->
$$
\Sigma_1\neq\Sigma_2,
$$
标准 pooled covariance 公式不再适用。
<!-- bilingual-en:start -->
the standard pooled-covariance formula no longer applies.
<!-- bilingual-en:end -->

>[!attention] 常见错误
> 题目明确协方差矩阵不等时，不要机械套 pooled 两样本 Hotelling $T^2$。
> <!-- bilingual-en:start -->
> When a question explicitly states that the covariance matrices differ, do not mechanically apply the pooled two-sample Hotelling $T^2$ test.
> <!-- bilingual-en:end -->

## 1.5. 多个总体均值向量比较：单因子 MANOVA
<!-- bilingual-en:start -->
*1.5. Comparing Several Population Mean Vectors: One-Way MANOVA*
<!-- bilingual-en:end -->

### 1.5.1. 问题设定
<!-- bilingual-en:start -->
*1.5.1. Problem Setup*
<!-- bilingual-en:end -->

有 $g$ 个总体，每个观测是 $p$ 维向量。
<!-- bilingual-en:start -->
There are $g$ populations, and each observation is a $p$-dimensional vector.
<!-- bilingual-en:end -->

检验：
<!-- bilingual-en:start -->
Test:
<!-- bilingual-en:end -->
$$
H_0:\mu_1=\mu_2=\cdots=\mu_g.
$$

### 1.5.2. 模型表达
<!-- bilingual-en:start -->
*1.5.2. Model Representation*
<!-- bilingual-en:end -->

单因子 MANOVA 可写为
<!-- bilingual-en:start -->
A one-way MANOVA can be written as
<!-- bilingual-en:end -->
$$
X_{ij}=\mu+\tau_i+e_{ij},
$$
其中 $\tau_i$ 表示第 $i$ 个组的效应。
<!-- bilingual-en:start -->
where $\tau_i$ is the effect of group $i$.
<!-- bilingual-en:end -->

检验等价于
<!-- bilingual-en:start -->
The equivalent hypothesis is
<!-- bilingual-en:end -->
$$
H_0:\tau_1=\cdots=\tau_g=0.
$$

### 1.5.3. SSP 矩阵分解
<!-- bilingual-en:start -->
*1.5.3. SSP Matrix Decomposition*
<!-- bilingual-en:end -->

总 SSP 矩阵分解为
<!-- bilingual-en:start -->
The total SSP matrix decomposes as
<!-- bilingual-en:end -->
$$
T=H+E.
$$

其中：
<!-- bilingual-en:start -->
where:
<!-- bilingual-en:end -->

- $T$：总变异；
- $H$：组间变异；
- $E$：组内误差变异。
<!-- bilingual-en:start -->
- $T$: total variation;
- $H$: between-group variation;
- $E$: within-group error variation.
<!-- bilingual-en:end -->

### 1.5.4. Wilks Lambda 检验
<!-- bilingual-en:start -->
*1.5.4. Wilks' Lambda Test*
<!-- bilingual-en:end -->

Wilks Lambda 定义为
<!-- bilingual-en:start -->
Wilks' Lambda is defined as
<!-- bilingual-en:end -->
$$
\Lambda^*=\frac{|E|}{|E+H|}.
$$

若 $\Lambda^*$ 很小，表示组间差异解释了较多总变异，倾向拒绝 $H_0$。
<!-- bilingual-en:start -->
A small $\Lambda^*$ means that between-group differences explain a substantial share of total variation and favours rejecting $H_0$.
<!-- bilingual-en:end -->

>[!attention] 方向
> Wilks Lambda 越小越显著，不是越大越显著。
> <!-- bilingual-en:start -->
> A smaller Wilks' Lambda is more significant, not a larger one.
> <!-- bilingual-en:end -->

### 1.5.5. MANOVA 表
<!-- bilingual-en:start -->
*1.5.5. MANOVA Table*
<!-- bilingual-en:end -->

MANOVA 表通常组织为：
<!-- bilingual-en:start -->
A MANOVA table is usually organised as follows:
<!-- bilingual-en:end -->

| 来源 | SSP 矩阵 | 自由度 |
|---|---|---|
| 组间 | $H$ | $g-1$ |
| 组内误差 | $E$ | $N-g$ |
| 总计 | $T$ | $N-1$ |
<!-- bilingual-en:start -->
| Source | SSP matrix | Degrees of freedom |
|---|---|---|
| Between groups | $H$ | $g-1$ |
| Within-group error | $E$ | $N-g$ |
| Total | $T$ | $N-1$ |
<!-- bilingual-en:end -->

## 1.6. 采样分布提示
<!-- bilingual-en:start -->
*1.6. Note on Sampling Distributions*
<!-- bilingual-en:end -->

MANOVA 的精确或近似分布依赖样本量、组数和维度。课程复习中优先掌握：
<!-- bilingual-en:start -->
The exact or approximate distribution in MANOVA depends on sample size, the number of groups, and dimension. For this course, prioritise:
<!-- bilingual-en:end -->

1. $T=H+E$ 的矩阵分解。
2. Wilks Lambda 的定义和方向。
3. 显著后需要进一步解释变量或线性组合。
<!-- bilingual-en:start -->

&nbsp;
**1.** The matrix decomposition $T=H+E$.<br>
**2.** The definition and direction of Wilks' Lambda.<br>
**3.** The need to interpret variables or linear combinations after a significant result.<br>
<!-- bilingual-en:end -->

## 1.7. 题型识别表
<!-- bilingual-en:start -->
*1.7. Question-Type Recognition Table*
<!-- bilingual-en:end -->

| 题目关键词 | 用法 |
|---|---|
| 同一对象前后比较 | 配对样本均值向量比较 |
| 两个独立总体，协方差相等 | 两样本 Hotelling $T^2$ |
| 两个独立总体，协方差不等 | 不直接 pooled |
| 三个及以上总体 | one-way MANOVA |
| 多个均值差区间 | 同时置信区间或 Bonferroni |
<!-- bilingual-en:start -->
| Wording in the question | Method |
|---|---|
| Before-and-after comparison on the same objects | Paired-sample mean-vector comparison |
| Two independent populations with equal covariances | Two-sample Hotelling $T^2$ |
| Two independent populations with unequal covariances | Do not pool directly |
| Three or more populations | One-way MANOVA |
| Intervals for several mean differences | Simultaneous confidence intervals or Bonferroni intervals |
<!-- bilingual-en:end -->

## 1.8. 关联卡片
<!-- bilingual-en:start -->
*1.8. Related Cards*
<!-- bilingual-en:end -->

- [[Hotelling T² 与多元均值推断#配对与两独立总体|Paired Mean Vector Comparison]]
- [[Hotelling T² 与多元均值推断#配对与两独立总体|Two-Sample Hotelling T2 Test]]
- [[Hotelling T² 与多元均值推断#单总体 Hotelling $T^2$|Hotelling T2 Test]]
- [[MANOVA 多元方差分析#MANOVA 的模型|MANOVA]]
- [[MANOVA 多元方差分析#MANOVA 的模型|One-way MANOVA Procedure]]
- [[MANOVA 多元方差分析#$H$ 与 $E$ 矩阵|Wilks Lambda]]
- [[MANOVA 多元方差分析#$H$ 与 $E$ 矩阵|SSP Matrix]]
- [[Hotelling T² 与多元均值推断|Multivariate Mean Inference Map]]
