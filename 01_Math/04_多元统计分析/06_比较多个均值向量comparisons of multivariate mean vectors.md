# 0. 回忆用

- 本章核心：比较多个总体的均值向量（mean vectors）是否存在显著差异。
- 两组比较优先看 Hotelling \(T^2\)；多组比较优先看单因子多元方差分析（one-way MANOVA）与 Wilks Lambda。
- 若协方差矩阵不等（\(\Sigma_1 \neq \Sigma_2\)），不能直接用 pooled covariance 方法。

# 1. 比较配对样本的均值向量（Paired Mean Vector Comparison）

## 1.1 场景与符号

- 目标：比较同一对象在两种处理（treatment）下的多变量测量是否存在均值差异。
- 记第 \(j\) 个样本在两种处理下第 \(i\) 个变量的观测为：
  - \(x_{1ji}\)：处理 1（treatment 1）
  - \(x_{2ji}\)：处理 2（treatment 2）

## 1.2 差值向量与参数（Difference Vector and Parameter）

定义每个样本的差值向量（difference vector）：

\[
D_j=
\begin{pmatrix}
D_{j1}\\
D_{j2}\\
\vdots\\
D_{jp}
\end{pmatrix}
=
\begin{pmatrix}
x_{1j1}-x_{2j1}\\
x_{1j2}-x_{2j2}\\
\vdots\\
x_{1jp}-x_{2jp}
\end{pmatrix}.
\]

其总体参数为：

\[
E(D_j)=\delta=
\begin{pmatrix}
\delta_1\\
\delta_2\\
\vdots\\
\delta_p
\end{pmatrix},
\qquad
\operatorname{Cov}(D_j)=\Sigma_d.
\]

## 1.3 Hotelling \(T^2\) 检验（Hotelling's \(T^2\) Test）

检验问题：

\[
H_0:\delta=0
\quad \text{vs.} \quad
H_1:\delta\neq 0.
\]

统计量：

\[
T^2=n(\bar D-\delta)'S_d^{-1}(\bar D-\delta),
\]

其中：

\[
\bar D=\frac{1}{n}\sum_{j=1}^n D_j,
\qquad
S_d=\frac{1}{n-1}\sum_{j=1}^n(D_j-\bar D)(D_j-\bar D)'.
\]

==重点==

- 样本量较大时，\(T^2\) 可用 \(\chi_p^2\) 近似。
- 精确分布可写为：

\[
T^2\sim \frac{p(n-1)}{n-p}F_{p,n-p}.
\]

## 1.4 同时置信区间（Simultaneous Confidence Intervals）

第 \(i\) 个均值差 \(\delta_i\) 的同时置信区间：

\[
\bar d_i \pm
\sqrt{\frac{p(n-1)}{n-p}F_{p,n-p}(\alpha)}
\sqrt{\frac{S_{dii}}{n}}.
\]

其中：

- \(\bar d_i\)：第 \(i\) 个差值分量的样本均值。
- \(S_{dii}\)：\(S_d\) 的第 \(i\) 个对角元素。

## 1.5 案例：商业实验室 vs 国家实验室（Case Study）

### 1.5.1 题干与数据

| Sample \(j\) | Commercial Lab \((x_{1j1},x_{1j2})\) | State Lab \((x_{2j1},x_{2j2})\) |
|---|---|---|
| 1  | (6, 27)   | (25, 15) |
| 2  | (6, 23)   | (28, 13) |
| 3  | (18, 64)  | (36, 22) |
| 4  | (8, 44)   | (35, 29) |
| 5  | (11, 30)  | (15, 31) |
| 6  | (34, 75)  | (44, 64) |
| 7  | (28, 26)  | (42, 30) |
| 8  | (71, 124) | (54, 64) |
| 9  | (43, 54)  | (54, 64) |
| 10 | (33, 30)  | (34, 56) |
| 11 | (20, 14)  | (39, 21) |

变量（variables）：

- 变量 1：生化需氧量（Biochemical Oxygen Demand, BOD）
- 变量 2：悬浮固体（Suspended Solids, SS）

实验目标：检验两实验室在 BOD 与 SS 的均值测量是否存在系统差异。

### 1.5.2 参数计算

差值定义：

\[
d_{j1}=x_{1j1}-x_{2j1},
\qquad
d_{j2}=x_{1j2}-x_{2j2}.
\]

样本差值序列：

- \(d_{j1}\)：\(-19,-22,-18,-27,-4,-10,-14,17,9,4,-19\)
- \(d_{j2}\)：\(12,10,42,15,-1,11,-4,-60,-2,10,-7\)

样本均值向量与协方差矩阵：

\[
\bar{\mathbf d}=
\begin{bmatrix}
-9.36\\
13.27
\end{bmatrix},
\qquad
S_d=
\begin{bmatrix}
199.26 & 88.38\\
88.38 & 418.61
\end{bmatrix},
\qquad
S_d^{-1}=
\begin{bmatrix}
0.0055 & 0.0012\\
-0.0012 & 0.0026
\end{bmatrix}.
\]

### 1.5.3 \(T^2\) 统计量

\[
T^2=n\bar{\mathbf d}'S_d^{-1}\bar{\mathbf d}
=11
\begin{bmatrix}
-9.36 & 13.27
\end{bmatrix}
\begin{bmatrix}
0.0055 & 0.0012\\
-0.0012 & 0.0026
\end{bmatrix}
\begin{bmatrix}
-9.36\\
13.27
\end{bmatrix}
=13.6.
\]

### 1.5.4 检验结论与区间

在 \(\alpha=0.05\) 下，临界值为：

\[
c_\alpha=
\frac{p(n-1)}{n-p}F_{p,n-p}(0.05)
=
\frac{2\cdot 10}{11-2}F_{2,9}(0.05)
=9.47.
\]

由于 \(T^2=13.6>9.47\)，拒绝 \(H_0\)。

95% 同时置信区间：

\[
\delta_1:\ \bar d_1\pm\sqrt{c_\alpha\frac{S_{d,11}}{n}}
=-9.36\pm\sqrt{9.47\cdot\frac{199.26}{11}}
=(-22.46,\ 3.74),
\]

\[
\delta_2:\ \bar d_2\pm\sqrt{c_\alpha\frac{S_{d,22}}{n}}
=13.27\pm\sqrt{9.47\cdot\frac{418.61}{11}}
=(-5.71,\ 32.25).
\]

说明：区间可能包含 0，但联合检验 \(T^2\) 仍可拒绝 \(H_0\)，这反映了“联合差异”与“分量单独区间”判定口径不同。

## 1.6 参数线性变换检验（Transformed Parametric Functions）

给定变换矩阵：

\[
C=
\begin{bmatrix}
1&0&-1&0\\
0&1&0&-1
\end{bmatrix},
\qquad
\operatorname{rank}(C)=2.
\]

定义参数：

\[
\psi=C\mu=
\begin{bmatrix}
\mu_1-\mu_3\\
\mu_2-\mu_4
\end{bmatrix}.
\]

检验：

\[
H_0:\psi=0
\quad \text{vs.} \quad
H_1:\psi\neq 0.
\]

已知：

\[
\bar X=
\begin{bmatrix}
25.2727\\
46.4545\\
34.6364\\
33.1818
\end{bmatrix},
\quad
S=
\begin{bmatrix}
387.4 & 489.4 & 1014.1 & 148.7\\
489.4 & 225.4 & 109.3 & 296.0\\
1014.1 & 109.3 & 479.6 & 120.4\\
148.7 & 296.0 & 120.4 & 363.8
\end{bmatrix}.
\]

则

\[
\hat\psi=C\bar X=
\begin{bmatrix}
-9.3636\\
13.2727
\end{bmatrix},
\qquad
CSC'=
\begin{bmatrix}
199.2545 & 88.3091\\
88.3091 & 418.6182
\end{bmatrix}.
\]

统计量：

\[
T^2=n\hat\psi'(CSC')^{-1}\hat\psi=13.6.
\]

临界值：

\[
C_\alpha=\frac{m(n-1)}{n-m}F_{m,n-m}(\alpha),
\]

当 \(\alpha=0.05,m=2,n=11\) 时，\(C_\alpha=9.47\)。

结论：\(T^2=13.6>9.47\)，拒绝 \(H_0\)。

# 2. 比较两个总体均值向量（Two-Sample Mean Vector Comparison）

## 2.1 协方差矩阵相等（Equal Covariance Matrices）

设：

\[
X_{11},X_{12},\dots,X_{1n_1}\sim N_p(\mu_1,\Sigma),
\quad
X_{21},X_{22},\dots,X_{2n_2}\sim N_p(\mu_2,\Sigma).
\]

样本均值：

\[
\bar X_1=\frac{1}{n_1}\sum_{j=1}^{n_1}X_{1j},
\qquad
\bar X_2=\frac{1}{n_2}\sum_{j=1}^{n_2}X_{2j}.
\]

样本协方差：

\[
S_1=\frac{1}{n_1-1}\sum_{j=1}^{n_1}(X_{1j}-\bar X_1)(X_{1j}-\bar X_1)',
\]

\[
S_2=\frac{1}{n_2-1}\sum_{j=1}^{n_2}(X_{2j}-\bar X_2)(X_{2j}-\bar X_2)'.
\]

合并协方差矩阵（pooled covariance matrix）：

\[
S_{\text{pooled}}=\frac{(n_1-1)S_1+(n_2-1)S_2}{n_1+n_2-2}.
\]

检验：

\[
H_0:\mu_1=\mu_2
\quad \text{vs.} \quad
H_1:\mu_1\neq\mu_2,
\]

或更一般地：

\[
H_0:\mu_1-\mu_2=\delta_0
\quad \text{vs.} \quad
H_1:\mu_1-\mu_2\neq\delta_0.
\]

统计量：

\[
T^2=(\bar X_1-\bar X_2-\delta_0)'
\left[\left(\frac{1}{n_1}+\frac{1}{n_2}\right)S_{\text{pooled}}\right]^{-1}
(\bar X_1-\bar X_2-\delta_0).
\]

分布：

\[
T^2\sim
\frac{p(n_1+n_2-2)}{n_1+n_2-1-p}
F_{p,\,n_1+n_2-1-p}.
\]

拒绝域：

\[
T^2>
\frac{p(n_1+n_2-2)}{n_1+n_2-1-p}
F_{p,\,n_1+n_2-1-p}(\alpha).
\]

同时区间：

\[
a'(\bar X_1-\bar X_2)
\pm C_\alpha
\sqrt{a'\left(\frac{1}{n_1}+\frac{1}{n_2}\right)S_{\text{pooled}}a},
\]

其中

\[
C_\alpha^2=
\frac{p(n_1+n_2-2)}{n_1+n_2-1-p}
F_{p,\,n_1+n_2-1-p}(\alpha).
\]

第 \(i\) 个分量区间：

\[
(\mu_{1i}-\mu_{2i})
\in
(\bar x_{1i}-\bar x_{2i})
\pm
C_\alpha\sqrt{\left(\frac{1}{n_1}+\frac{1}{n_2}\right)s_{ii,\text{pooled}}}.
\]

Bonferroni 同时区间：

\[
(\mu_{1i}-\mu_{2i})
\in
(\bar x_{1i}-\bar x_{2i})
\pm
t_{n_1+n_2-2}\!\left(\frac{\alpha}{2p}\right)
\sqrt{\left(\frac{1}{n_1}+\frac{1}{n_2}\right)s_{ii,\text{pooled}}}.
\]

## 2.2 协方差矩阵不等（Unequal Covariance Matrices）

==课堂备注：这部分以理解思路为主，考试权重通常较低。==

当 \(\Sigma_1\neq\Sigma_2\) 时，不能使用 pooled covariance 矩阵，需改用广义方法。

检验：

\[
H_0:\mu_1=\mu_2
\quad \text{vs.} \quad
H_1:\mu_1\neq\mu_2.
\]

可用广义 Hotelling \(T^2\) 统计量：

\[
T^2=
[(\bar X_1-\bar X_2)-(\mu_1-\mu_2)]'
\left(\frac{1}{n_1}S_1+\frac{1}{n_2}S_2\right)^{-1}
[(\bar X_1-\bar X_2)-(\mu_1-\mu_2)].
\]

其中

\[
\frac{1}{n_1}S_1+\frac{1}{n_2}S_2
\]

是均值差的估计协方差矩阵。

小样本下该统计量不再严格服从简单的 \(F\) 分布，可用近似方法（如多变量 Behrens-Fisher 近似）：

\[
T^2\sim \frac{\nu p}{\nu-p+1}F_{p,\nu-p+1},
\]

\[
\nu=
\frac{p+p^2}{
\sum_{i=1}^2\frac{1}{n_i}
\left\{
\operatorname{tr}\left[\frac{1}{n_i}S_i\left(\frac{1}{n_1}S_1+\frac{1}{n_2}S_2\right)^{-1}\right]^2
+
\left(\operatorname{tr}\left[\frac{1}{n_i}S_i\left(\frac{1}{n_1}S_1+\frac{1}{n_2}S_2\right)^{-1}\right]\right)^2
\right\}
}.
\]

并有范围：

\[
\min(n_1,n_2)\le \nu\le n_1+n_2.
\]

对应的同时置信区间可写为：

\[
(\mu_1-\mu_2)_i
\in
(\bar x_{1i}-\bar x_{2i})
\pm
C_\alpha\sqrt{\left(\frac{S_1}{n_1}+\frac{S_2}{n_2}\right)_{ii}},
\]

其中 \(C_\alpha\) 可由 \(\chi^2\) 近似或重抽样（resampling）方法获得。

# 3. 比较多个总体均值向量：单因子 MANOVA（One-Way MANOVA）

==理解检验逻辑与矩阵分解即可，重点不在机械背诵。==

## 3.1 问题设定（Problem Setup）

设有 \(g\) 个总体：

\[
X_{l1},X_{l2},\dots,X_{ln_l}\sim N_p(\mu_l,\Sigma),
\quad l=1,2,\dots,g,
\]

并假设各组独立、协方差矩阵相同。

检验：

\[
H_0:\mu_1=\mu_2=\cdots=\mu_g
\quad \text{vs.} \quad
H_1:\text{至少有一个均值向量不同}.
\]

## 3.2 模型表达（Model Representation）

\[
x_{lj}=\mu+\gamma_l+e_{lj},
\]

其中：

- \(\mu\)：总体中心（grand mean）
- \(\gamma_l\)：第 \(l\) 组处理效应（treatment effect）
- \(e_{lj}\sim N_p(0,\Sigma)\)：组内残差（within-group residual）

分解形式：

\[
x_{lj}=\bar X+(\bar X_l-\bar X)+(x_{lj}-\bar X_l).
\]

## 3.3 SSP 矩阵分解（Sum of Squares and Cross Products）

总 SSP：

\[
T=
\sum_{l=1}^g\sum_{j=1}^{n_l}
(x_{lj}-\bar X)(x_{lj}-\bar X)'.
\]

处理 SSP（between）：

\[
B=
\sum_{l=1}^g
n_l(\bar X_l-\bar X)(\bar X_l-\bar X)'.
\]

组内 SSP（within）：

\[
W=
\sum_{l=1}^g\sum_{j=1}^{n_l}
(x_{lj}-\bar X_l)(x_{lj}-\bar X_l)'.
\]

关系：

\[
T=B+W.
\]

## 3.4 Wilks Lambda 检验（Wilks' Lambda Test）

定义统计量：

\[
\Lambda^*=\frac{|W|}{|T|}.
\]

- \(\Lambda^*\) 越小，说明组间差异越显著。

大样本近似：

\[
\left(n-1-\frac{p+g}{2}\right)\ln\Lambda^*
\sim \chi^2_{p(g-1)}.
\]

拒绝域：

\[
\left(n-1-\frac{p+g}{2}\right)\ln\Lambda^*>
\chi^2_{p(g-1)}(\alpha).
\]

## 3.5 MANOVA 表（One-Way MANOVA Table）

| Source | SSP 矩阵 | 自由度 |
|---|---|---|
| Treatment (Between) | \(B=\sum_{l=1}^g n_l(\bar X_l-\bar X)(\bar X_l-\bar X)'\) | \(g-1\) |
| Residual (Within) | \(W=\sum_{l=1}^g\sum_{j=1}^{n_l}(x_{lj}-\bar X_l)(x_{lj}-\bar X_l)'\) | \(\sum_{l=1}^g n_l-g\) |
| Total (Corrected) | \(T=B+W\) | \(\sum_{l=1}^g n_l-1\) |

## 3.6 采样分布提示（Sampling Distribution Notes）

- Wilks Lambda 的精确分布依赖 \(p\)、\(g\) 与样本量结构。
- 实务中常使用近似 \(F\) 变换或 \(\chi^2\) 近似。
- 本章考试更关注“建模-统计量-拒绝域-解释”这一完整链条。
