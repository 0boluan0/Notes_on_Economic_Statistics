# 1. 第6章：比较多个均值向量（Comparisons of Multivariate Mean Vectors）

>[!summary] 本章主线
> 第 5 章检验一个总体均值向量，第 6 章比较两个或多个总体均值向量。两组问题优先识别“配对”还是“独立”；多组问题进入 one-way MANOVA。

## 1.1. 回忆用

本章核心：

1. 配对样本：先构造差值向量，再做单样本 Hotelling $T^2$。
2. 两独立样本：协方差相等时使用 pooled covariance。
3. 多个总体：使用单因子 MANOVA，核心统计量是 Wilks Lambda。

>[!warning] 题型边界
> 若 $\Sigma_1\neq\Sigma_2$，不能直接使用 pooled covariance 两样本 $T^2$。

## 1.2. 配对样本均值向量比较

### 1.2.1. 场景与符号

同一对象在两种处理下被测量多个变量。

记第 $j$ 个样本在两种处理下的观测为
$$
X_{1j},\qquad X_{2j}.
$$

### 1.2.2. 差值向量

定义差值向量：
$$
D_j=X_{1j}-X_{2j}.
$$

总体差值均值为
$$
E(D_j)=\delta.
$$

检验：
$$
H_0:\delta=0
\quad\text{vs.}\quad
H_1:\delta\neq0.
$$

### 1.2.3. Hotelling T² 检验

计算
$$
\bar D=\frac1n\sum_{j=1}^nD_j,
$$
$$
S_d=\frac1{n-1}\sum_{j=1}^n(D_j-\bar D)(D_j-\bar D)'.
$$

统计量：
$$
T^2=n\bar D'S_d^{-1}\bar D.
$$

在 $H_0$ 下：
$$
\frac{n-p}{p(n-1)}T^2\sim F_{p,n-p}.
$$

### 1.2.4. 同时置信区间

第 $i$ 个均值差 $\delta_i$ 的同时置信区间：
$$
\bar d_i\pm
\sqrt{\frac{p(n-1)}{n-p}F_{p,n-p}(1-\alpha)}
\sqrt{\frac{s_{d,ii}}{n}}.
$$

### 1.2.5. 案例：商业实验室 vs 国家实验室

旧笔记中的例子比较两个实验室对 BOD 和 SS 的测量差异。

差值向量为
$$
D_j=
\begin{bmatrix}
\text{Commercial BOD}-\text{State BOD}\\
\text{Commercial SS}-\text{State SS}
\end{bmatrix}.
$$

样本计算得到
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
$$
T^2=13.6.
$$

在 $\alpha=0.05$ 下，临界值约为 $9.47$，因此拒绝 $H_0$。

>[!note] 解读
> 即使单个分量的同时区间可能包含 0，联合检验仍可能拒绝。这说明两个变量合在一起呈现了显著差异。

## 1.3. 参数线性变换检验

有时比较的不是原始均值，而是线性组合：
$$
\psi=C\mu.
$$

例如
$$
C=
\begin{bmatrix}
1&0&-1&0\\
0&1&0&-1
\end{bmatrix}
$$
可以比较两组变量的差异。

检验统计量仍使用 Hotelling 型：
$$
T^2=n(C\bar X-\psi_0)'(CSC')^{-1}(C\bar X-\psi_0).
$$

如果 $C$ 的秩为 $m$，则
$$
\frac{n-m}{m(n-1)}T^2\sim F_{m,n-m}.
$$

## 1.4. 两个独立总体均值向量比较

### 1.4.1. 协方差矩阵相等

设两组样本独立，且
$$
\Sigma_1=\Sigma_2=\Sigma.
$$

检验：
$$
H_0:\mu_1-\mu_2=0.
$$

pooled covariance 为
$$
S_p=\frac{(n_1-1)S_1+(n_2-1)S_2}{n_1+n_2-2}.
$$

统计量：
$$
T^2=\frac{n_1n_2}{n_1+n_2}
(\bar X_1-\bar X_2)'S_p^{-1}(\bar X_1-\bar X_2).
$$

F 转换：
$$
\frac{n_1+n_2-p-1}{p(n_1+n_2-2)}T^2
\sim F_{p,n_1+n_2-p-1}.
$$

### 1.4.2. 协方差矩阵不等

若
$$
\Sigma_1\neq\Sigma_2,
$$
标准 pooled covariance 公式不再适用。

>[!warning] 常见错误
> 题目明确协方差矩阵不等时，不要机械套 pooled 两样本 Hotelling $T^2$。

## 1.5. 多个总体均值向量比较：单因子 MANOVA

### 1.5.1. 问题设定

有 $g$ 个总体，每个观测是 $p$ 维向量。

检验：
$$
H_0:\mu_1=\mu_2=\cdots=\mu_g.
$$

### 1.5.2. 模型表达

单因子 MANOVA 可写为
$$
X_{ij}=\mu+\tau_i+e_{ij},
$$
其中 $\tau_i$ 表示第 $i$ 个组的效应。

检验等价于
$$
H_0:\tau_1=\cdots=\tau_g=0.
$$

### 1.5.3. SSP 矩阵分解

总 SSP 矩阵分解为
$$
T=H+E.
$$

其中：

- $T$：总变异；
- $H$：组间变异；
- $E$：组内误差变异。

### 1.5.4. Wilks Lambda 检验

Wilks Lambda 定义为
$$
\Lambda^*=\frac{|E|}{|E+H|}.
$$

若 $\Lambda^*$ 很小，表示组间差异解释了较多总变异，倾向拒绝 $H_0$。

>[!warning] 方向
> Wilks Lambda 越小越显著，不是越大越显著。

### 1.5.5. MANOVA 表

MANOVA 表通常组织为：

| 来源 | SSP 矩阵 | 自由度 |
|---|---|---|
| 组间 | $H$ | $g-1$ |
| 组内误差 | $E$ | $N-g$ |
| 总计 | $T$ | $N-1$ |

## 1.6. 采样分布提示

MANOVA 的精确或近似分布依赖样本量、组数和维度。课程复习中优先掌握：

1. $T=H+E$ 的矩阵分解。
2. Wilks Lambda 的定义和方向。
3. 显著后需要进一步解释变量或线性组合。

## 1.7. 题型识别表

| 题目关键词 | 用法 |
|---|---|
| 同一对象前后比较 | 配对样本均值向量比较 |
| 两个独立总体，协方差相等 | 两样本 Hotelling $T^2$ |
| 两个独立总体，协方差不等 | 不直接 pooled |
| 三个及以上总体 | one-way MANOVA |
| 多个均值差区间 | 同时置信区间或 Bonferroni |

## 1.8. 关联卡片

- [[Paired Mean Vector Comparison]]
- [[Two-Sample Hotelling T2 Test]]
- [[Hotelling T2 Test]]
- [[MANOVA]]
- [[One-way MANOVA Procedure]]
- [[Wilks Lambda]]
- [[SSP Matrix]]
- [[Multivariate Mean Inference Map]]
