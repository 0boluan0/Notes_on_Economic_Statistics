# 1. 第3章：样本几何与随机抽样（Sample Geometry and Random Sampling）
<!-- bilingual-en:start -->
*1. Chapter 3: Sample Geometry and Random Sampling*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 本章把样本矩阵看成几何对象：每个变量是一根中心化后的偏差向量，协方差来自偏差向量内积，相关系数来自偏差向量夹角，广义方差来自这些向量张成的面积或体积。
> <!-- bilingual-en:start -->
> [!note] Chapter backbone
> This chapter treats the sample matrix as a geometric object. Each variable becomes a centred deviation vector; covariance comes from inner products, correlation from angles between deviation vectors, and generalised variance from the area or volume spanned by those vectors.
> <!-- bilingual-en:end -->

## 1.1. 样本的几何结构
<!-- bilingual-en:start -->
*1.1. Geometric Structure of a Sample*
<!-- bilingual-en:end -->

### 1.1.1. 数据矩阵表示
<!-- bilingual-en:start -->
*1.1.1. Data-Matrix Representation*
<!-- bilingual-en:end -->

数据矩阵为
<!-- bilingual-en:start -->
The data matrix is
<!-- bilingual-en:end -->
$$
X_{n\times p}=
\begin{bmatrix}
x_{11}&x_{12}&\cdots&x_{1p}\\
x_{21}&x_{22}&\cdots&x_{2p}\\
\vdots&\vdots&\ddots&\vdots\\
x_{n1}&x_{n2}&\cdots&x_{np}
\end{bmatrix}.
$$

每一行是一个样本，每一列是一个变量。
<!-- bilingual-en:start -->
Each row is an observation, and each column is a variable.
<!-- bilingual-en:end -->

### 1.1.2. 均值修正向量与偏差向量
<!-- bilingual-en:start -->
*1.1.2. Mean-Corrected and Deviation Vectors*
<!-- bilingual-en:end -->

第 $i$ 个变量的观测向量为
<!-- bilingual-en:start -->
The observation vector for variable $i$ is
<!-- bilingual-en:end -->
$$
y_i=
\begin{bmatrix}
x_{1i}\\x_{2i}\\\vdots\\x_{ni}
\end{bmatrix}.
$$

其偏差向量为
<!-- bilingual-en:start -->
Its deviation vector is
<!-- bilingual-en:end -->
$$
d_i=y_i-\bar x_i\mathbf1
=
\begin{bmatrix}
x_{1i}-\bar x_i\\
x_{2i}-\bar x_i\\
\vdots\\
x_{ni}-\bar x_i
\end{bmatrix}.
$$

>[!note] 几何读法
> 一个变量不是只被看成一列数字，而是被看成 $n$ 维空间中的一根偏差向量。
> <!-- bilingual-en:start -->
> [!note] Geometric interpretation
> A variable is viewed not merely as a column of numbers but as a deviation vector in an $n$-dimensional space.
> <!-- bilingual-en:end -->

### 1.1.3. 偏差平方和
<!-- bilingual-en:start -->
*1.1.3. Sum of Squared Deviations*
<!-- bilingual-en:end -->

变量 $i$ 的偏差平方和为
<!-- bilingual-en:start -->
The sum of squared deviations for variable $i$ is
<!-- bilingual-en:end -->
$$
L^2(d_i)=d_i'd_i=\sum_{j=1}^n(x_{ji}-\bar x_i)^2.
$$

这就是该变量围绕均值的总偏离程度。
<!-- bilingual-en:start -->
This is the variable's total deviation around its mean.
<!-- bilingual-en:end -->

### 1.1.4. 偏差向量夹角与相关系数
<!-- bilingual-en:start -->
*1.1.4. Angles Between Deviation Vectors and Correlation*
<!-- bilingual-en:end -->

两个变量 $i,k$ 的偏差向量内积为
<!-- bilingual-en:start -->
The inner product of the deviation vectors for variables $i$ and $k$ is
<!-- bilingual-en:end -->
$$
d_i'd_k=\sum_{j=1}^n(x_{ji}-\bar x_i)(x_{jk}-\bar x_k).
$$

夹角余弦为
<!-- bilingual-en:start -->
The cosine of the angle is
<!-- bilingual-en:end -->
$$
\cos(\theta_{ik})
=\frac{d_i'd_k}{L(d_i)L(d_k)}
=\frac{s_{ik}}{\sqrt{s_{ii}s_{kk}}}
=r_{ik}.
$$

>[!item] 关键结论
> 样本相关系数就是两个中心化变量向量夹角的余弦。
> <!-- bilingual-en:start -->
> [!item] Key conclusion
> The sample correlation coefficient is the cosine of the angle between two centred variable vectors.
> <!-- bilingual-en:end -->

## 1.2. 随机样本与样本矩
<!-- bilingual-en:start -->
*1.2. Random Samples and Sample Moments*
<!-- bilingual-en:end -->

### 1.2.1. 样本协方差矩阵
<!-- bilingual-en:start -->
*1.2.1. Sample Covariance Matrix*
<!-- bilingual-en:end -->

样本协方差矩阵 $S=((s_{ik}))$ 的元素为
<!-- bilingual-en:start -->
The entries of the sample covariance matrix $S=((s_{ik}))$ are
<!-- bilingual-en:end -->
$$
s_{ik}
=\frac{1}{n-1}\sum_{j=1}^n(x_{ji}-\bar x_i)(x_{jk}-\bar x_k)
=\frac{1}{n-1}d_i'd_k.
$$

令偏差矩阵
<!-- bilingual-en:start -->
Let the deviation matrix be
<!-- bilingual-en:end -->
$$
D=
\begin{bmatrix}
|&|&&|\\
d_1&d_2&\cdots&d_p\\
|&|&&|
\end{bmatrix},
$$
则
<!-- bilingual-en:start -->
Then
<!-- bilingual-en:end -->
$$
S=\frac{1}{n-1}D'D.
$$

>[!attention] 易错点
> $D$ 是中心化后的 deviation matrix。协方差矩阵不是直接由未中心化的 $X'X$ 得到。
> <!-- bilingual-en:start -->
> [!attention] Common error
> $D$ is the centred deviation matrix. The covariance matrix is not obtained directly from an uncentred $X'X$.
> <!-- bilingual-en:end -->

### 1.2.2. 样本均值向量的期望
<!-- bilingual-en:start -->
*1.2.2. Expectation of the Sample Mean Vector*
<!-- bilingual-en:end -->

若 $X_1,\ldots,X_n$ 是来自总体均值 $\mu$ 的随机样本，则
<!-- bilingual-en:start -->
If $X_1,\ldots,X_n$ is a random sample from a population with mean $\mu$, then
<!-- bilingual-en:end -->
$$
E(\bar X)=\mu.
$$

### 1.2.3. 样本协方差矩阵的期望
<!-- bilingual-en:start -->
*1.2.3. Expectation of the Sample Covariance Matrix*
<!-- bilingual-en:end -->

在常规随机抽样条件下，
<!-- bilingual-en:start -->
Under ordinary random-sampling conditions,
<!-- bilingual-en:end -->
$$
E(S)=\Sigma.
$$

这解释了为什么分母用 $n-1$。
<!-- bilingual-en:start -->
This explains the denominator $n-1$.
<!-- bilingual-en:end -->

## 1.3. 广义方差
<!-- bilingual-en:start -->
*1.3. Generalised Variance*
<!-- bilingual-en:end -->

### 1.3.1. 行列式作为广义方差
<!-- bilingual-en:start -->
*1.3.1. The Determinant as Generalised Variance*
<!-- bilingual-en:end -->

样本广义方差定义为
<!-- bilingual-en:start -->
The sample generalised variance is defined as
<!-- bilingual-en:end -->
$$
|S|.
$$

它把多维变异压缩成一个数。
<!-- bilingual-en:start -->
It compresses multivariate variation into one number.
<!-- bilingual-en:end -->

二维情形：
<!-- bilingual-en:start -->
In two dimensions:
<!-- bilingual-en:end -->
$$
|S|=s_{11}s_{22}-s_{12}^2
=s_{11}s_{22}(1-r_{12}^2).
$$

当 $|r_{12}|=1$ 时，$|S|=0$，说明两个变量完全线性相关，二维面积退化为 0。
<!-- bilingual-en:start -->
When $|r_{12}|=1$, $|S|=0$: the variables are perfectly linearly related and the two-dimensional area collapses to zero.
<!-- bilingual-en:end -->

### 1.3.2. 迹作为总变异
<!-- bilingual-en:start -->
*1.3.2. The Trace as Total Variation*
<!-- bilingual-en:end -->

另一种汇总变异的方式是迹：
<!-- bilingual-en:start -->
Another summary of variation is the trace:
<!-- bilingual-en:end -->
$$
\operatorname{tr}(S)=s_{11}+s_{22}+\cdots+s_{pp}.
$$

它只加总各变量自己的方差，不惩罚变量之间的相关性。
<!-- bilingual-en:start -->
It simply adds the variables' own variances and does not penalise correlation among them.
<!-- bilingual-en:end -->

>[!note] 对比
> $|S|$ 看联合体积，受相关性影响；$\operatorname{tr}(S)$ 看总方差，不直接看变量之间是否重叠。
> <!-- bilingual-en:start -->
> [!note] Comparison
> $|S|$ measures joint volume and is affected by correlation; $\operatorname{tr}(S)$ measures total marginal variance without directly reflecting overlap among variables.
> <!-- bilingual-en:end -->

## 1.4. 广义方差的几何解释
<!-- bilingual-en:start -->
*1.4. Geometric Interpretation of Generalised Variance*
<!-- bilingual-en:end -->

### 1.4.1. 二维面积
<!-- bilingual-en:start -->
*1.4.1. Two-Dimensional Area*
<!-- bilingual-en:end -->

当 $p=2$ 时，两个偏差向量张成平行四边形。面积为
<!-- bilingual-en:start -->
When $p=2$, the two deviation vectors span a parallelogram with area
<!-- bilingual-en:end -->
$$
\operatorname{Area}
=(n-1)\sqrt{s_{11}s_{22}(1-r_{12}^2)}
=(n-1)|S|^{1/2}.
$$

若两个变量正交，面积较大；若两个变量完全相关，面积为 0。
<!-- bilingual-en:start -->
The area is large when the variables are orthogonal and zero when they are perfectly correlated.
<!-- bilingual-en:end -->

### 1.4.2. 高维超体积
<!-- bilingual-en:start -->
*1.4.2. High-Dimensional Hypervolume*
<!-- bilingual-en:end -->

当 $p>2$ 时，$d_1,\ldots,d_p$ 张成高维体积。
<!-- bilingual-en:start -->
When $p>2$, $d_1,\ldots,d_p$ span a high-dimensional volume.
<!-- bilingual-en:end -->

有
<!-- bilingual-en:start -->
Specifically,
<!-- bilingual-en:end -->
$$
|S|=(\operatorname{Volume})^2(n-1)^{-p}.
$$

### 1.4.3. 秩亏缺情形
<!-- bilingual-en:start -->
*1.4.3. Rank-Deficient Cases*
<!-- bilingual-en:end -->

若 $n\leq p$，或某个变量是其他变量的线性组合，则 $S$ 可能秩亏：
<!-- bilingual-en:start -->
If $n\leq p$, or if one variable is a linear combination of others, $S$ may be rank-deficient:
<!-- bilingual-en:end -->
$$
|S|=0.
$$

此时很多需要 $S^{-1}$ 的方法会失效。
<!-- bilingual-en:start -->
Many methods requiring $S^{-1}$ then fail.
<!-- bilingual-en:end -->

可选处理：
<!-- bilingual-en:start -->
Possible responses include:
<!-- bilingual-en:end -->

1. 删除冗余变量。
2. 先做 PCA 降维。
3. 使用正则化协方差矩阵。
4. 在尺度问题明显时改用相关矩阵。
<!-- bilingual-en:start -->
1. Remove redundant variables.
2. Apply PCA first to reduce dimension.
3. Use a regularised covariance matrix.
4. Use the correlation matrix when scale differences are substantial.
<!-- bilingual-en:end -->

## 1.5. 样本协方差矩阵的矩阵表示
<!-- bilingual-en:start -->
*1.5. Matrix Representation of the Sample Covariance Matrix*
<!-- bilingual-en:end -->

### 1.5.1. 中心化矩阵
<!-- bilingual-en:start -->
*1.5.1. Centring Matrix*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->
$$
H=I-\frac1n\mathbf1\mathbf1'.
$$

中心化后的数据矩阵可写作
<!-- bilingual-en:start -->
The centred data matrix can be written as
<!-- bilingual-en:end -->
$$
D=HX.
$$

所以
<!-- bilingual-en:start -->
Therefore,
<!-- bilingual-en:end -->
$$
S=\frac{1}{n-1}X'HX
=\frac{1}{n-1}D'D.
$$

### 1.5.2. 行列式与特征值
<!-- bilingual-en:start -->
*1.5.2. Determinant and Eigenvalues*
<!-- bilingual-en:end -->

若 $S$ 的特征值为 $\lambda_1,\ldots,\lambda_p$，则
<!-- bilingual-en:start -->
If the eigenvalues of $S$ are $\lambda_1,\ldots,\lambda_p$, then
<!-- bilingual-en:end -->
$$
|S|=\prod_{i=1}^p\lambda_i,
\qquad
\operatorname{tr}(S)=\sum_{i=1}^p\lambda_i.
$$

在 PCA 中，$\lambda_i$ 表示第 $i$ 个主成分方向上的样本方差。
<!-- bilingual-en:start -->
In PCA, $\lambda_i$ is the sample variance along principal-component direction $i$.
<!-- bilingual-en:end -->

## 1.6. 标准化协方差矩阵与相关矩阵
<!-- bilingual-en:start -->
*1.6. Standardised Covariance and Correlation Matrices*
<!-- bilingual-en:end -->

### 1.6.1. 标准化
<!-- bilingual-en:start -->
*1.6.1. Standardisation*
<!-- bilingual-en:end -->

变量 $X_j$ 的标准化形式为
<!-- bilingual-en:start -->
The standardised form of variable $X_j$ is
<!-- bilingual-en:end -->
$$
Z_j=\frac{X_j-\bar X_j}{\sqrt{s_{jj}}}.
$$

标准化后均值为 0，方差为 1。
<!-- bilingual-en:start -->
After standardisation, its mean is zero and its variance is one.
<!-- bilingual-en:end -->

### 1.6.2. 样本相关矩阵
<!-- bilingual-en:start -->
*1.6.2. Sample Correlation Matrix*
<!-- bilingual-en:end -->

若 $Z$ 是标准化后的数据矩阵，则
<!-- bilingual-en:start -->
If $Z$ is the standardised data matrix, then
<!-- bilingual-en:end -->
$$
R=\frac{1}{n-1}Z'Z.
$$

元素为
<!-- bilingual-en:start -->
Its entries are
<!-- bilingual-en:end -->
$$
r_{ij}=\frac{s_{ij}}{\sqrt{s_{ii}s_{jj}}}.
$$

### 1.6.3. 协方差矩阵与相关矩阵转换
<!-- bilingual-en:start -->
*1.6.3. Converting Between Covariance and Correlation Matrices*
<!-- bilingual-en:end -->

令
<!-- bilingual-en:start -->
Let
<!-- bilingual-en:end -->
$$
D_s^{1/2}=\operatorname{diag}(\sqrt{s_{11}},\ldots,\sqrt{s_{pp}}).
$$

则
<!-- bilingual-en:start -->
Then
<!-- bilingual-en:end -->
$$
R=D_s^{-1/2}SD_s^{-1/2},
$$
$$
S=D_s^{1/2}RD_s^{1/2}.
$$

## 1.7. 关联卡片
<!-- bilingual-en:start -->
*1.7. Related Cards*
<!-- bilingual-en:end -->

- [[Sample Mean Vector]]
- [[Sample Covariance Matrix]]
- [[Generalized Variance]]
- [[Correlation Matrix]]
- [[Choosing Covariance vs Correlation Matrix]]
- [[PCA]]
- [[Matrix Rank]]
