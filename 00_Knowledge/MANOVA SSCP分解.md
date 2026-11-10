---
aliases:
- MANOVA SSCP decomposition
- One-way MANOVA SSCP decomposition
- MANOVA 的 H、E 与 T 矩阵
student_os: knowledge-atom
atom_id: STAT-MAN-002
atom_set: manova
atom_type: decomposition
status: source-checked
part_of:
- '[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]'
requires:
- '[[MANOVA]]'
- '[[多响应残差与误差SSP]]'
implies:
- '[[MANOVA四种统计量]]'
related:
- '[[误差SSP秩边界]]'
- '[[Wishart分布]]'
---

<!-- bilingual-en:start -->
# SSCP decomposition in one-way MANOVA
<!-- bilingual-en:end -->

*单因素 MANOVA 的平方和与交叉乘积分解*

<!-- bilingual-en:start -->
Suppose three groups each provide two measurements per person. Some variation comes from people differing within the same group; some comes from the group averages being different. The SSCP decomposition separates these two sources while keeping track of how the two measurements vary together. It is the matrix version of “total squared variation = within-group variation + between-group variation.”
<!-- bilingual-en:end -->

假设有三组人，每个人都测量两项指标。有些变动来自同组内人与人的差别，有些来自各组平均水平不同。SSCP 分解把这两个来源拆开，同时保留两项测量共同变动的信息。它是“总平方变动 = 组内变动 + 组间变动”的矩阵版本。

<!-- bilingual-en:start -->
[[MANOVA|MANOVA]] uses several responses together to assess a group or model effect. This card develops the **one-way group decomposition**. The matrix identity itself is arithmetic: it does not require normal observations, independent observations, or equal population covariance matrices. Those conditions become relevant when using the matrices for statistical inference.
<!-- bilingual-en:end -->

[[MANOVA]]把多个响应一起用于判断组别或模型效应。本卡解释的是**单因素分组分解**。矩阵恒等式本身是算术结论，不需要观测正态、观测独立或总体协方差相同；这些条件关系到之后的统计推断。

<!-- bilingual-en:start -->
## What a square-and-product matrix stores
<!-- bilingual-en:end -->

*平方和与交叉乘积矩阵记录什么*

<!-- bilingual-en:start -->
Let $Y_{ij}\in\mathbb R^p$ be the column of $p\ge1$ measurements on observation $j$ in group $i$. There are $g\ge1$ nonempty groups, with positive integer sizes $n_i$, and $N=\sum_{i=1}^g n_i$ observations altogether. Write the group [[样本均值向量|sample means]] and the overall mean as
$$\bar Y_i=\frac1{n_i}\sum_{j=1}^{n_i}Y_{ij},\qquad
\bar Y=\frac1N\sum_{i=1}^g n_i\bar Y_i.$$
The overall mean weights a group by its size. An unweighted average of group means is generally wrong when the sizes differ.
<!-- bilingual-en:end -->

令 $Y_{ij}\in\mathbb R^p$ 为第 $i$ 组第 $j$ 个观测的列向量，含 $p\ge1$ 项测量。有 $g\ge1$ 个非空组，组样本量 $n_i$ 为正整数，总数为 $N=\sum_i n_i$。$\bar Y_i$ 是各组[[样本均值向量]]，$\bar Y$ 是所有观测的总均值。总均值按组大小加权；各组大小不同时，通常不能直接把组均值不加权取平均。

<!-- bilingual-en:start -->
For a deviation column $d=(d_1,d_2)^T$, its **outer product** is
$$dd^T=\begin{bmatrix}d_1^2&d_1d_2\\d_1d_2&d_2^2\end{bmatrix}.$$
Summing these matrices gives sums of squares on the diagonal and sums of cross-products off the diagonal. This is what **SSCP**, “sums of squares and cross-products,” means; some sources and linked cards shorten it to SSP. Off-diagonal entries can be negative, and a zero cross-product sum alone does not establish independence.
<!-- bilingual-en:end -->

偏差列 $d=(d_1,d_2)^T$ 与自己的转置相乘，得到上述**外积**。把这些矩阵相加，对角线上累积平方和，非对角线上累积交叉乘积和。这就是 **SSCP**（sums of squares and cross-products，平方和与交叉乘积）的意思；部分来源及链接卡片简写为 SSP。非对角元可以为负，交叉乘积和为零本身也不等于独立。

> [!important] Three matrices, three centres · 三张矩阵与三种偏差
> <!-- bilingual-en:start -->
> All three matrices below are $p\times p$. The superscript $T$ means [[矩阵转置|transpose]]; the standalone letter $T$ names the total matrix.
> $$\begin{aligned}
> E&=\sum_{i=1}^g\sum_{j=1}^{n_i}(Y_{ij}-\bar Y_i)(Y_{ij}-\bar Y_i)^T,\\
> H&=\sum_{i=1}^g n_i(\bar Y_i-\bar Y)(\bar Y_i-\bar Y)^T,\\
> T&=\sum_{i=1}^g\sum_{j=1}^{n_i}(Y_{ij}-\bar Y)(Y_{ij}-\bar Y)^T.
> \end{aligned}$$
> $E$ measures deviations from each observation's **own group mean**. $H$ measures deviations of **group means from the overall mean**, weighted by group size. $T$ measures deviations of **all observations from the overall mean**. They satisfy $\boxed{T=E+H}$.
> <!-- bilingual-en:end -->
>
> 三者都是 $p\times p$ 矩阵；上标 $T$ 表示[[矩阵转置|转置]]，单独字母 $T$ 则表示总矩阵。$E$ 围绕**本组均值**，$H$ 比较**组均值与总均值**并按人数加权，$T$ 围绕**所有观测的总均值**。它们满足 $T=E+H$。

<!-- bilingual-en:start -->
$E$ is called the error or within-group SSCP; $H$ is the hypothesis or between-group SSCP. Here “error” means variation left after fitting a separate group mean, not necessarily a measurement mistake. If the residual vectors are stored as rows of an $N\times p$ matrix $R$, then $E=R^TR$. The [[多响应残差与误差SSP|residual matrix and the error SSCP]] have different dimensions: $R$ retains individual observations, whereas $E$ aggregates them. None of $E,H,T$ has yet been divided by degrees of freedom, so these are not sample covariance matrices.
<!-- bilingual-en:end -->

$E$ 称为误差或组内 SSCP，$H$ 称为假设或组间 SSCP。这里的“误差”表示拟合各组均值后留下的变动，不一定是测量失误。把残差向量逐行放入 $N\times p$ 矩阵 $R$，就有 $E=R^TR$。[[多响应残差与误差SSP|残差矩阵与误差 SSCP]]维度不同：$R$ 保留逐个观测，$E$ 则把它们汇总。$E,H,T$ 都尚未除以自由度，因此不是样本协方差矩阵。

<!-- bilingual-en:start -->
## Why the cross terms disappear
<!-- bilingual-en:end -->

*交叉项为什么消失*

<!-- bilingual-en:start -->
Each observation's total deviation splits into a within-group deviation and a group-mean deviation:
$$Y_{ij}-\bar Y=(Y_{ij}-\bar Y_i)+(\bar Y_i-\bar Y).$$
In group $i$, the second vector is constant across its $n_i$ observations, while the first vectors sum to zero. Thus one of the cross-term sums is
$$\sum_j(Y_{ij}-\bar Y_i)(\bar Y_i-\bar Y)^T
=\left[\sum_j(Y_{ij}-\bar Y_i)\right](\bar Y_i-\bar Y)^T=0.$$
Its transpose also vanishes. The two remaining outer-product sums are exactly $E$ and $H$, proving the decomposition. “Cross terms vanish” here refers to these terms between the two deviation sources; it does **not** say that off-diagonal entries inside $E$ or $H$ vanish.
<!-- bilingual-en:end -->

每个观测相对总均值的偏差，分成“观测减本组均值”与“本组均值减总均值”。同组内第二个向量固定，而第一个向量加起来为零，因此展开外积后，上述交叉项及其转置都消失。剩下的两个外积和正好是 $E$ 与 $H$，于是得到分解。这里“交叉项消失”说的是**两类偏差之间**的项，不是说 $E$ 或 $H$ 内部的非对角元为零。

> [!example] Six observations you can calculate by hand · 六个可手算的观测
> <!-- bilingual-en:start -->
> Take three groups with two coordinates and two observations each:
>
> | Group | Observations | Group mean |
> |---|---|---|
> | A | $(-2,-1)^T,(0,-1)^T$ | $(-1,-1)^T$ |
> | B | $(1,-1)^T,(1,1)^T$ | $(1,0)^T$ |
> | C | $(-1,0)^T,(1,2)^T$ | $(0,1)^T$ |
>
> The overall mean is zero. The within-group residual pairs are $\pm(1,0)^T$, $\pm(0,1)^T$, and $\pm(1,1)^T$. Hence
> $$E=\begin{bmatrix}2&0\\0&0\end{bmatrix}+\begin{bmatrix}0&0\\0&2\end{bmatrix}+\begin{bmatrix}2&2\\2&2\end{bmatrix}=\begin{bmatrix}4&2\\2&4\end{bmatrix}.$$
> Weighting the three mean outer products by $2$ gives the same matrix, $H=\begin{bmatrix}4&2\\2&4\end{bmatrix}$. Directly summing the six observations' outer products gives $T=\begin{bmatrix}8&4\\4&8\end{bmatrix}=E+H$. The off-diagonal cross-products remain visible.
> <!-- bilingual-en:end -->
>
> 三组均值分别为 $(-1,-1)^T,(1,0)^T,(0,1)^T$，总均值为零。各组残差分别为 $\pm(1,0)^T$、$\pm(0,1)^T$、$\pm(1,1)^T$，因此按上式得到 $E$。各组均值的外积乘人数 $2$ 后相加，恰好得到相同的 $H$；六个观测直接算总外积则得到 $T=E+H$。可以看到，非对角交叉乘积并没有消失。

<!-- bilingual-en:start -->
## Degrees of freedom are not matrix rank
<!-- bilingual-en:end -->

*自由度不等于矩阵秩*

<!-- bilingual-en:start -->
For each response coordinate, subtracting $g$ separate group means leaves $N-g$ residual degrees of freedom. Subtracting only the overall mean leaves $N-1$. Their difference is the $g-1$ between-group degrees of freedom. These count independent directions on the **observation axis**, not the number $p$ of response coordinates:
<!-- bilingual-en:end -->

对每个响应坐标，分别减去各组均值（共 $g$ 个），留下 $N-g$ 个残差自由度；只减一个总均值，留下 $N-1$ 个自由度；两者之差是 $g-1$ 个组间自由度。这数的是**观测轴上**独立变化的方向，不是响应坐标数 $p$。

<!-- bilingual-en:start -->
| Source | SSCP | Degrees of freedom |
|---|---|---|
| Within groups | $E$ | $N-g$ |
| Between groups | $H$ | $g-1$ |
| Total | $T$ | $N-1$ |
<!-- bilingual-en:end -->

<!-- bilingual-en:start -->
In the six-observation example, these degrees of freedom are $3,2,5$, although each two-dimensional matrix has rank at most $2$. Likewise, with four groups and $N=40$, the degrees of freedom for $H,E,T$ are $3,36,39$.
<!-- bilingual-en:end -->

上表依次给出组内、组间与总计的矩阵及自由度。六观测例子中，自由度分别为 $3,2,5$；两维矩阵的秩却至多为 $2$。四组、$N=40$ 的例子同理给出 $H,E,T$ 自由度 $3,36,39$。

<!-- bilingual-en:start -->

Each matrix is [[外积和的半正定性|a nonnegative sum of outer products]], hence [[半正定矩阵|positive semidefinite]]: every projected squared deviation is nonnegative. The centering constraints also give
$$\operatorname{rank}(E)\le\min(p,N-g),\qquad
\operatorname{rank}(H)\le\min(p,g-1).$$
In particular, $N-g<p$ forces $E$ to be singular. Having $N-g\ge p$ only permits full rank; exact collinearity can still prevent it. The [[误差SSP秩边界|error-SSCP rank boundary]] separates this algebraic condition from almost-sure full rank under a suitable continuous sampling model.
<!-- bilingual-en:end -->

每张矩阵都是[[外积和的半正定性|外积的非负加权和]]，因此[[半正定矩阵|半正定]]：投影后的平方偏差不可能为负。中心化约束还给出上述秩上界。特别地，$N-g<p$ 时 $E$ 必奇异；$N-g\ge p$ 只让满秩成为可能，精确共线仍可能阻止它。[[误差SSP秩边界]]区分了这一代数门槛与适当连续抽样模型下的几乎必然满秩。

<!-- bilingual-en:start -->
If $N>g$, dividing $E$ by $N-g$ gives the usual pooled within-group covariance estimate under the common-covariance model. That division is a later normalization, not part of $T=E+H$. With one group, $H=0$; if every group contains one observation, $E=0$. The decomposition still holds, but division by a zero degree of freedom does not. How the matrices become scalar evidence against equal population means is the separate task of [[MANOVA四种统计量|the four MANOVA criteria]] and their [[MANOVA检验校准|reference-distribution calibration]].
<!-- bilingual-en:end -->

若 $N>g$，$E/(N-g)$ 在共同协方差模型下是通常的 pooled 组内协方差估计；这是之后的归一化，不属于 $T=E+H$ 本身。只有一组时 $H=0$；每组只有一个观测时 $E=0$。分解仍成立，但不能除以零自由度。如何把矩阵变成反对总体均值相等的标量证据，由[[MANOVA四种统计量]]及其[[MANOVA检验校准|参考分布校准]]另行说明。

## 来源与核验

<!-- bilingual-en:start -->
- [Penn State STAT505, Lesson8 §8.2](https://online.stat.psu.edu/stat505/Lesson08): the one-way response-vector notation, weighted grand mean, three SSCP definitions, decomposition and degrees-of-freedom table. The cancellation, rank explanation and worked example are shown directly above.
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm): hypothesis/error matrices for a shared-design multivariate model, and their diagonals as the corresponding univariate sums of squares. The present card restricts the explicit decomposition to one-way groups.
- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.5.3. SSCP 矩阵分解|Course §1.5.3]] and [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.5.5. MANOVA 表|§1.5.5]]: course notation and the MANOVA table. No sampling distribution is inferred merely from the arithmetic identity.
<!-- bilingual-en:end -->

- Penn State 第8课8.2节核对向量记号、加权总均值、三张矩阵、分解与自由度表；交叉项消失、秩说明与数值例子在正文直接核验。
- SAS 核对共享设计多响应模型中的假设/误差矩阵，以及对角元对应单响应平方和；本卡的显式分解限定于单因素分组。
- 课程第1.5.3与1.5.5节核对课程记号和 MANOVA 表；没有仅凭算术恒等式推断抽样分布。
