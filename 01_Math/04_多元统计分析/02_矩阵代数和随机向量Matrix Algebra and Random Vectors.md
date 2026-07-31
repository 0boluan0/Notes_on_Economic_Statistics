# 1. 第2章：矩阵代数与随机向量（Matrix Algebra and Random Vectors）
<!-- bilingual-en:start -->
*Chapter 2: Matrix Algebra and Random Vectors*
<!-- bilingual-en:end -->

>[!note] 本章主线
> 多元统计把“多个变量一起变化”写成向量和矩阵。本章是后续多元正态、Hotelling $T^2$、PCA 和因子分析的线性代数入口。
> <!-- bilingual-en:start -->
> Multivariate statistics represents the joint variation of several variables with vectors and matrices. This chapter provides the linear-algebra foundation for the multivariate normal distribution, Hotelling's $T^2$, PCA, and factor analysis.
> <!-- bilingual-en:end -->

## 1.1. 矩阵与向量代数基础
<!-- bilingual-en:start -->
*Foundations of Vector and Matrix Algebra*
<!-- bilingual-en:end -->

### 1.1.1. 向量的定义与表示
<!-- bilingual-en:start -->
*Definition and Representation of Vectors*
<!-- bilingual-en:end -->

列向量写作
<!-- bilingual-en:start -->
A column vector is written as
<!-- bilingual-en:end -->
$$
x=
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_n
\end{bmatrix},
\qquad
x'=(x_1,x_2,\ldots,x_n).
$$

向量长度为
<!-- bilingual-en:start -->
The length of a vector is
<!-- bilingual-en:end -->
$$
L_x=\sqrt{x'x}=\sqrt{x_1^2+\cdots+x_n^2}.
$$

单位向量为
<!-- bilingual-en:start -->
The corresponding unit vector is
<!-- bilingual-en:end -->
$$
x^*=\frac{x}{L_x}.
$$

### 1.1.2. 夹角、内积与正交
<!-- bilingual-en:start -->
*Angles, Inner Products, and Orthogonality*
<!-- bilingual-en:end -->

两个向量 $x,y$ 的夹角 $\theta$ 满足
<!-- bilingual-en:start -->
The angle $\theta$ between two vectors $x$ and $y$ satisfies
<!-- bilingual-en:end -->
$$
\cos\theta=\frac{x'y}{L_xL_y}.
$$

当 $x'y=0$ 时，两个向量正交。
<!-- bilingual-en:start -->
When $x'y=0$, the two vectors are orthogonal.
<!-- bilingual-en:end -->

>[!note] 几何直觉
> 相关系数可以看成两个中心化变量向量夹角的余弦。第 3 章会把这个直觉用于样本几何。
> <!-- bilingual-en:start -->
> A correlation coefficient can be viewed as the cosine of the angle between two centered variable vectors. Chapter 3 applies this intuition to sample geometry.
> <!-- bilingual-en:end -->

### 1.1.3. 线性相关与投影
<!-- bilingual-en:start -->
*Linear Dependence and Projection*
<!-- bilingual-en:end -->

向量组 $\{x_1,\ldots,x_k\}$ 线性相关，指存在不全为 0 的系数 $c_1,\ldots,c_k$，使
<!-- bilingual-en:start -->
A set of vectors $\{x_1,\ldots,x_k\}$ is linearly dependent if there are coefficients $c_1,\ldots,c_k$, not all zero, such that
<!-- bilingual-en:end -->
$$
c_1x_1+\cdots+c_kx_k=0.
$$

向量 $x$ 在 $y$ 上的投影为
<!-- bilingual-en:start -->
The projection of $x$ onto $y$ is
<!-- bilingual-en:end -->
$$
\operatorname{proj}_y(x)=\frac{x'y}{y'y}y.
$$

## 1.2. 矩阵（Matrices）
<!-- bilingual-en:start -->
*Matrices*
<!-- bilingual-en:end -->

### 1.2.1. 矩阵的定义
<!-- bilingual-en:start -->
*Definition of a Matrix*
<!-- bilingual-en:end -->

矩阵是按行列排列的数值数组：
<!-- bilingual-en:start -->
A matrix is a rectangular array of numbers arranged in rows and columns:
<!-- bilingual-en:end -->
$$
A_{n\times p}=
\begin{bmatrix}
a_{11}&a_{12}&\cdots&a_{1p}\\
a_{21}&a_{22}&\cdots&a_{2p}\\
\vdots&\vdots&\ddots&\vdots\\
a_{n1}&a_{n2}&\cdots&a_{np}
\end{bmatrix}.
$$

若 $n=p$，称为方阵。
<!-- bilingual-en:start -->
If $n=p$, the matrix is square.
<!-- bilingual-en:end -->

### 1.2.2. 常见特殊矩阵
<!-- bilingual-en:start -->
*Common Special Matrices*
<!-- bilingual-en:end -->

单位矩阵：
<!-- bilingual-en:start -->
The identity matrix is
<!-- bilingual-en:end -->
$$
I_p=
\begin{bmatrix}
1&0&\cdots&0\\
0&1&\cdots&0\\
\vdots&\vdots&\ddots&\vdots\\
0&0&\cdots&1
\end{bmatrix}.
$$

对角矩阵：
<!-- bilingual-en:start -->
A diagonal matrix is
<!-- bilingual-en:end -->
$$
D=\operatorname{diag}(d_1,\ldots,d_p).
$$

若 $d_i\neq0$，则
<!-- bilingual-en:start -->
If every $d_i\neq0$, then
<!-- bilingual-en:end -->
$$
D^{-1}=\operatorname{diag}\left(\frac1{d_1},\ldots,\frac1{d_p}\right).
$$

### 1.2.3. 矩阵运算
<!-- bilingual-en:start -->
*Matrix Operations*
<!-- bilingual-en:end -->

矩阵加法要求维度相同。
<!-- bilingual-en:start -->
Matrix addition requires the matrices to have the same dimensions.
<!-- bilingual-en:end -->

矩阵乘法要求内维匹配：
<!-- bilingual-en:start -->
Matrix multiplication requires the inner dimensions to agree:
<!-- bilingual-en:end -->
$$
A_{n\times p}B_{p\times m}=C_{n\times m},
\qquad
c_{ik}=\sum_{j=1}^p a_{ij}b_{jk}.
$$

转置把行列互换：
<!-- bilingual-en:start -->
Transposition interchanges rows and columns:
<!-- bilingual-en:end -->
$$
(A')_{ij}=A_{ji}.
$$

对称矩阵满足
<!-- bilingual-en:start -->
A symmetric matrix satisfies
<!-- bilingual-en:end -->
$$
A'=A.
$$

### 1.2.4. 逆矩阵
<!-- bilingual-en:start -->
*Inverse Matrices*
<!-- bilingual-en:end -->

若方阵 $A$ 存在 $A^{-1}$，使
<!-- bilingual-en:start -->
If a square matrix $A$ has a matrix $A^{-1}$ such that
<!-- bilingual-en:end -->
$$
AA^{-1}=A^{-1}A=I,
$$
则 $A$ 可逆。
<!-- bilingual-en:start -->
then $A$ is invertible.
<!-- bilingual-en:end -->

>[!attention] 多元统计里的关键边界
> 总体协方差矩阵常假设为正定，因此可逆；样本协方差矩阵不一定可逆，尤其当 $n\leq p$ 或变量完全线性相关时。
> <!-- bilingual-en:start -->
> A population covariance matrix is often assumed to be positive definite and is therefore invertible. A sample covariance matrix need not be invertible, especially when $n\leq p$ or some variables are perfectly linearly dependent.
> <!-- bilingual-en:end -->

### 1.2.5. 正交矩阵
<!-- bilingual-en:start -->
*Orthogonal Matrices*
<!-- bilingual-en:end -->

若
<!-- bilingual-en:start -->
If
<!-- bilingual-en:end -->
$$
Q'Q=QQ'=I,
$$
则 $Q$ 是正交矩阵。
<!-- bilingual-en:start -->
then $Q$ is an orthogonal matrix.
<!-- bilingual-en:end -->

正交矩阵满足
<!-- bilingual-en:start -->
An orthogonal matrix satisfies
<!-- bilingual-en:end -->
$$
Q^{-1}=Q'.
$$

它的列向量构成标准正交组。
<!-- bilingual-en:start -->
Its columns form an orthonormal set.
<!-- bilingual-en:end -->

### 1.2.6. 矩阵的秩
<!-- bilingual-en:start -->
*Matrix Rank*
<!-- bilingual-en:end -->

矩阵 $A$ 的秩是线性无关列向量的最大个数：
<!-- bilingual-en:start -->
The rank of a matrix $A$ is the maximum number of linearly independent columns:
<!-- bilingual-en:end -->
$$
\operatorname{rank}(A)\leq\min(n,p).
$$

满秩表示达到最大可能秩；否则称为秩亏。
<!-- bilingual-en:start -->
A matrix is full rank when its rank is as large as possible; otherwise it is rank deficient.
<!-- bilingual-en:end -->

### 1.2.7. 矩阵的迹
<!-- bilingual-en:start -->
*Matrix Trace*
<!-- bilingual-en:end -->

方阵 $A$ 的迹为对角线元素之和：
<!-- bilingual-en:start -->
The trace of a square matrix $A$ is the sum of its diagonal entries:
<!-- bilingual-en:end -->
$$
\operatorname{tr}(A)=\sum_{i=1}^n a_{ii}.
$$

常用性质：

1. $\operatorname{tr}(A+B)=\operatorname{tr}(A)+\operatorname{tr}(B)$。
2. $\operatorname{tr}(cA)=c\operatorname{tr}(A)$。
3. $\operatorname{tr}(A)=\operatorname{tr}(A')$。
4. $\operatorname{tr}(AB)=\operatorname{tr}(BA)$。
<!-- bilingual-en:start -->
Useful properties:

1. $\operatorname{tr}(A+B)=\operatorname{tr}(A)+\operatorname{tr}(B)$.
2. $\operatorname{tr}(cA)=c\operatorname{tr}(A)$.
3. $\operatorname{tr}(A)=\operatorname{tr}(A')$.
4. $\operatorname{tr}(AB)=\operatorname{tr}(BA)$.
<!-- bilingual-en:end -->

## 1.3. 行列式（Determinants）
<!-- bilingual-en:start -->
*Determinants*
<!-- bilingual-en:end -->

### 1.3.1. 定义与计算
<!-- bilingual-en:start -->
*Definition and Calculation*
<!-- bilingual-en:end -->

二维方阵的行列式：
<!-- bilingual-en:start -->
The determinant of a two-dimensional square matrix is
<!-- bilingual-en:end -->
$$
\begin{vmatrix}
a&b\\c&d
\end{vmatrix}
=ad-bc.
$$

对角矩阵行列式为对角线元素之积：
<!-- bilingual-en:start -->
The determinant of a diagonal matrix is the product of its diagonal entries:
<!-- bilingual-en:end -->
$$
|D|=\prod_{i=1}^p d_i.
$$

### 1.3.2. 核心性质
<!-- bilingual-en:start -->
*Core Properties*
<!-- bilingual-en:end -->

1. $|AB|=|A||B|$。
2. 若 $A$ 可逆，则 $|A^{-1}|=1/|A|$。
3. $|A'|=|A|$。
4. 若 $|A|\neq0$，则 $A$ 可逆。
<!-- bilingual-en:start -->
1. $|AB|=|A||B|$.
2. If $A$ is invertible, then $|A^{-1}|=1/|A|$.
3. $|A'|=|A|$.
4. If $|A|\neq0$, then $A$ is invertible.
<!-- bilingual-en:end -->

正交矩阵满足
<!-- bilingual-en:start -->
An orthogonal matrix satisfies
<!-- bilingual-en:end -->
$$
|Q|=\pm1.
$$

### 1.3.3. 与特征值的关系
<!-- bilingual-en:start -->
*Relationship to Eigenvalues*
<!-- bilingual-en:end -->

若 $A$ 的特征值为 $\lambda_1,\ldots,\lambda_p$，则
<!-- bilingual-en:start -->
If the eigenvalues of $A$ are $\lambda_1,\ldots,\lambda_p$, then
<!-- bilingual-en:end -->
$$
|A|=\prod_{i=1}^p\lambda_i,
\qquad
\operatorname{tr}(A)=\sum_{i=1}^p\lambda_i.
$$

>[!note] 后续用途
> 协方差矩阵的行列式就是 [[多元数据、随机向量与样本协方差#广义方差与可视化|Generalized Variance]]；它衡量多维数据联合变异的体积。
> <!-- bilingual-en:start -->
> The determinant of a covariance matrix is the [[多元数据、随机向量与样本协方差#广义方差与可视化|Generalized Variance]]. It measures the volume of joint variation in multivariate data.
> <!-- bilingual-en:end -->

## 1.4. 特征值与特征向量
<!-- bilingual-en:start -->
*Eigenvalues and Eigenvectors*
<!-- bilingual-en:end -->

### 1.4.1. 定义
<!-- bilingual-en:start -->
*Definition*
<!-- bilingual-en:end -->

若存在非零向量 $x$ 和标量 $\lambda$，使
<!-- bilingual-en:start -->
If there is a nonzero vector $x$ and a scalar $\lambda$ such that
<!-- bilingual-en:end -->
$$
Ax=\lambda x,
$$
则 $\lambda$ 是 $A$ 的特征值，$x$ 是对应特征向量。
<!-- bilingual-en:start -->
then $\lambda$ is an eigenvalue of $A$, and $x$ is a corresponding eigenvector.
<!-- bilingual-en:end -->

特征值由
<!-- bilingual-en:start -->
Eigenvalues are obtained from
<!-- bilingual-en:end -->
$$
|A-\lambda I|=0
$$
求得。
<!-- bilingual-en:start -->
by solving this characteristic equation.
<!-- bilingual-en:end -->

### 1.4.2. 几何意义
<!-- bilingual-en:start -->
*Geometric Meaning*
<!-- bilingual-en:end -->

特征向量是在矩阵作用下方向不变的方向；特征值表示该方向上的伸缩倍数。
<!-- bilingual-en:start -->
An eigenvector points in a direction that a matrix transformation leaves unchanged, while its eigenvalue gives the scale factor along that direction.
<!-- bilingual-en:end -->

当 $A$ 是实对称矩阵时，不同特征值对应的特征向量可取为正交。
<!-- bilingual-en:start -->
When $A$ is real and symmetric, eigenvectors associated with distinct eigenvalues can be chosen to be orthogonal.
<!-- bilingual-en:end -->

### 1.4.3. 谱分解（Spectral Decomposition）
<!-- bilingual-en:start -->
*Spectral Decomposition*
<!-- bilingual-en:end -->

若 $A$ 是实对称矩阵，则
<!-- bilingual-en:start -->
If $A$ is real and symmetric, then
<!-- bilingual-en:end -->
$$
A=Q\Lambda Q',
$$
其中 $Q$ 是正交特征向量矩阵，$\Lambda$ 是特征值对角矩阵。
<!-- bilingual-en:start -->
where $Q$ is the orthogonal matrix of eigenvectors and $\Lambda$ is the diagonal matrix of eigenvalues.
<!-- bilingual-en:end -->

也可写作
<!-- bilingual-en:start -->
Equivalently,
<!-- bilingual-en:end -->
$$
A=\sum_{i=1}^p\lambda_i q_iq_i'.
$$

>[!item] 复习连接
> PCA 正是对协方差矩阵做谱分解：特征向量给方向，特征值给该方向的方差。
> <!-- bilingual-en:start -->
> PCA performs a spectral decomposition of the covariance matrix: eigenvectors determine the directions, and eigenvalues give the variance along those directions.
> <!-- bilingual-en:end -->

### 1.4.4. 奇异值分解（SVD）
<!-- bilingual-en:start -->
*Singular Value Decomposition (SVD)*
<!-- bilingual-en:end -->

对任意矩阵 $A$，
<!-- bilingual-en:start -->
For any matrix $A$,
<!-- bilingual-en:end -->
$$
A=U\Sigma V'.
$$

其中：

- $V$ 给输入空间的正交方向；
- $\Sigma$ 给非负奇异值；
- $U$ 给输出空间的正交方向。
<!-- bilingual-en:start -->
Here:

- $V$ gives orthogonal directions in the input space;
- $\Sigma$ contains the nonnegative singular values;
- $U$ gives orthogonal directions in the output space.
<!-- bilingual-en:end -->

奇异值满足
<!-- bilingual-en:start -->
The singular values satisfy
<!-- bilingual-en:end -->
$$
\sigma_i=\sqrt{\lambda_i(A'A)}.
$$

SVD 可用于读秩、做低秩近似和处理不可逆问题。
<!-- bilingual-en:start -->
SVD can be used to determine rank, construct low-rank approximations, and handle problems involving noninvertible matrices.
<!-- bilingual-en:end -->

## 1.5. 正定与半正定矩阵
<!-- bilingual-en:start -->
*Positive-Definite and Positive-Semidefinite Matrices*
<!-- bilingual-en:end -->

### 1.5.1. 定义
<!-- bilingual-en:start -->
*Definition*
<!-- bilingual-en:end -->

对称矩阵 $A$ 若对任意非零向量 $x$ 满足
<!-- bilingual-en:start -->
A symmetric matrix $A$ is positive definite if, for every nonzero vector $x$,
<!-- bilingual-en:end -->
$$
x'Ax>0,
$$
则为正定矩阵。
<!-- bilingual-en:start -->
holds.
<!-- bilingual-en:end -->

若满足
<!-- bilingual-en:start -->
If instead
<!-- bilingual-en:end -->
$$
x'Ax\geq0,
$$
则为半正定矩阵。
<!-- bilingual-en:start -->
holds, then $A$ is positive semidefinite.
<!-- bilingual-en:end -->

### 1.5.2. 特征值判定
<!-- bilingual-en:start -->
*Eigenvalue Criterion*
<!-- bilingual-en:end -->

对称矩阵 $A$：

- 所有特征值 $>0$ 当且仅当 $A$ 正定；
- 所有特征值 $\geq0$ 当且仅当 $A$ 半正定。
<!-- bilingual-en:start -->
For a symmetric matrix $A$:

- every eigenvalue is $>0$ if and only if $A$ is positive definite;
- every eigenvalue is $\geq0$ if and only if $A$ is positive semidefinite.
<!-- bilingual-en:end -->

### 1.5.3. 证明思路
<!-- bilingual-en:start -->
*Proof Idea*
<!-- bilingual-en:end -->

若 $A=Q\Lambda Q'$，令 $y=Q'x$，则
<!-- bilingual-en:start -->
If $A=Q\Lambda Q'$ and $y=Q'x$, then
<!-- bilingual-en:end -->
$$
x'Ax=y'\Lambda y=\sum_{i=1}^p\lambda_i y_i^2.
$$

所以特征值全正时，任意非零 $x$ 都使二次型为正。
<!-- bilingual-en:start -->
Therefore, when all eigenvalues are positive, the quadratic form is positive for every nonzero $x$.
<!-- bilingual-en:end -->

## 1.6. 矩阵平方根与二次型
<!-- bilingual-en:start -->
*Matrix Square Roots and Quadratic Forms*
<!-- bilingual-en:end -->

### 1.6.1. 矩阵平方根（Matrix Square Root）
<!-- bilingual-en:start -->
*Matrix Square Root*
<!-- bilingual-en:end -->

若 $A$ 是对称正定矩阵，且
<!-- bilingual-en:start -->
If $A$ is symmetric and positive definite, with
<!-- bilingual-en:end -->
$$
A=Q\Lambda Q',
$$
则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->
$$
A^{1/2}=Q\Lambda^{1/2}Q',
$$
满足
<!-- bilingual-en:start -->
and it satisfies
<!-- bilingual-en:end -->
$$
A^{1/2}A^{1/2}=A.
$$

逆平方根为
<!-- bilingual-en:start -->
The inverse square root is
<!-- bilingual-en:end -->
$$
A^{-1/2}=Q\Lambda^{-1/2}Q'.
$$

### 1.6.2. 二次型（Quadratic Form）
<!-- bilingual-en:start -->
*Quadratic Form*
<!-- bilingual-en:end -->

形如
<!-- bilingual-en:start -->
A scalar of the form
<!-- bilingual-en:end -->
$$
x'Ax
$$
的标量称为二次型。
<!-- bilingual-en:start -->
is called a quadratic form.
<!-- bilingual-en:end -->

它在多元正态中形成马哈拉诺比斯距离：
<!-- bilingual-en:start -->
In the multivariate normal distribution, it forms the Mahalanobis distance:
<!-- bilingual-en:end -->
$$
(X-\mu)'\Sigma^{-1}(X-\mu).
$$

### 1.6.3. Rayleigh 商
<!-- bilingual-en:start -->
*Rayleigh Quotient*
<!-- bilingual-en:end -->

Rayleigh 商定义为
<!-- bilingual-en:start -->
The Rayleigh quotient is defined as
<!-- bilingual-en:end -->
$$
R(x)=\frac{x'Bx}{x'x}.
$$

若 $B$ 的特征值满足
<!-- bilingual-en:start -->
If the eigenvalues of $B$ satisfy
<!-- bilingual-en:end -->
$$
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p,
$$
则
<!-- bilingual-en:start -->
then
<!-- bilingual-en:end -->
$$
\lambda_p\leq R(x)\leq\lambda_1.
$$

最大值在最大特征值对应的特征向量方向取得。
<!-- bilingual-en:start -->
The maximum is attained in the direction of an eigenvector associated with the largest eigenvalue.
<!-- bilingual-en:end -->

## 1.7. 随机向量与随机矩阵
<!-- bilingual-en:start -->
*Random Vectors and Random Matrices*
<!-- bilingual-en:end -->

### 1.7.1. 随机向量
<!-- bilingual-en:start -->
*Random Vectors*
<!-- bilingual-en:end -->

随机向量写作
<!-- bilingual-en:start -->
A random vector is written as
<!-- bilingual-en:end -->
$$
X=
\begin{bmatrix}
X_1\\X_2\\\vdots\\X_p
\end{bmatrix}.
$$

均值向量为
<!-- bilingual-en:start -->
Its mean vector is
<!-- bilingual-en:end -->
$$
E(X)=
\begin{bmatrix}
E(X_1)\\
E(X_2)\\
\vdots\\
E(X_p)
\end{bmatrix}
=\mu.
$$

若 $Y=AX+b$，则
<!-- bilingual-en:start -->
If $Y=AX+b$, then
<!-- bilingual-en:end -->
$$
E(Y)=AE(X)+b.
$$

### 1.7.2. 协方差矩阵
<!-- bilingual-en:start -->
*Covariance Matrix*
<!-- bilingual-en:end -->

$$
\Sigma=\operatorname{Cov}(X)=E[(X-\mu)(X-\mu)'].
$$

展开为
<!-- bilingual-en:start -->
Expanded entry by entry, this is
<!-- bilingual-en:end -->
$$
\Sigma=
\begin{bmatrix}
\operatorname{Var}(X_1)&\operatorname{Cov}(X_1,X_2)&\cdots&\operatorname{Cov}(X_1,X_p)\\
\operatorname{Cov}(X_2,X_1)&\operatorname{Var}(X_2)&\cdots&\operatorname{Cov}(X_2,X_p)\\
\vdots&\vdots&\ddots&\vdots\\
\operatorname{Cov}(X_p,X_1)&\operatorname{Cov}(X_p,X_2)&\cdots&\operatorname{Var}(X_p)
\end{bmatrix}.
$$

### 1.7.3. 线性函数的方差
<!-- bilingual-en:start -->
*Variance of a Linear Function*
<!-- bilingual-en:end -->

若 $b$ 是常数向量，则
<!-- bilingual-en:start -->
If $b$ is a constant vector, then
<!-- bilingual-en:end -->
$$
\operatorname{Var}(b'X)=b'\Sigma b.
$$

这表示随机向量在方向 $b$ 上的分布宽度。
<!-- bilingual-en:start -->
This quantity measures the spread of the random vector in direction $b$.
<!-- bilingual-en:end -->

### 1.7.4. 独立与不相关
<!-- bilingual-en:start -->
*Independence and Uncorrelatedness*
<!-- bilingual-en:end -->

独立一定推出不相关：
<!-- bilingual-en:start -->
Independence always implies zero covariance:
<!-- bilingual-en:end -->
$$
X_i\perp X_j\Rightarrow \operatorname{Cov}(X_i,X_j)=0.
$$

但一般情况下，不相关不一定推出独立。
<!-- bilingual-en:start -->
In general, however, zero covariance does not imply independence.
<!-- bilingual-en:end -->

>[!note] 特例
> 在多元正态分布中，零协方差可以推出独立。
> <!-- bilingual-en:start -->
> In a multivariate normal distribution, zero covariance does imply independence.
> <!-- bilingual-en:end -->

## 1.8. 相关矩阵（Correlation Matrix）
<!-- bilingual-en:start -->
*Correlation Matrix*
<!-- bilingual-en:end -->

相关系数定义为
<!-- bilingual-en:start -->
The correlation coefficient is defined as
<!-- bilingual-en:end -->
$$
\rho_{ij}
=\frac{\operatorname{Cov}(X_i,X_j)}
{\sqrt{\operatorname{Var}(X_i)\operatorname{Var}(X_j)}}
=\frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}}.
$$

相关矩阵为
<!-- bilingual-en:start -->
The correlation matrix is
<!-- bilingual-en:end -->
$$
\rho=
\begin{bmatrix}
1&\rho_{12}&\cdots&\rho_{1p}\\
\rho_{21}&1&\cdots&\rho_{2p}\\
\vdots&\vdots&\ddots&\vdots\\
\rho_{p1}&\rho_{p2}&\cdots&1
\end{bmatrix}.
$$

若 $D=\operatorname{diag}(\sigma_{11},\ldots,\sigma_{pp})$，则
<!-- bilingual-en:start -->
If $D=\operatorname{diag}(\sigma_{11},\ldots,\sigma_{pp})$, then
<!-- bilingual-en:end -->
$$
\rho=D^{-1/2}\Sigma D^{-1/2},
\qquad
\Sigma=D^{1/2}\rho D^{1/2}.
$$

>[!item] 选择矩阵
> 变量量纲差异大时，用相关矩阵更稳；量纲相同且方差大小本身有意义时，用协方差矩阵更自然。
> <!-- bilingual-en:start -->
> Use the correlation matrix when variables have very different measurement scales. Use the covariance matrix when the variables share a scale and the magnitudes of their variances are substantively meaningful.
> <!-- bilingual-en:end -->

## 1.9. 关联卡片
<!-- bilingual-en:start -->
*Related Cards*
<!-- bilingual-en:end -->

- [[多元数据、随机向量与样本协方差#数据矩阵与随机向量|Matrix Operations]]
- [[广义逆与最小范数解#左逆、右逆与可逆|Matrix Inverse]]
- [[线性方程组与四个基本子空间#基、维数与秩|Matrix Rank]]
- [[行列式#行列式的结构含义|Determinant]]
- [[对称矩阵与正定二次型#对称矩阵与谱定理|Eigenvalues]]
- [[对称矩阵与正定二次型#对称矩阵与谱定理|Eigenvectors]]
- [[对称矩阵与正定二次型#对称矩阵与谱定理|Spectral Decomposition]]
- [[奇异值分解与低秩近似#SVD 的三层结构|Singular Value Decomposition]]
- [[对称矩阵与正定二次型#二次型与正定性|Positive Definite Matrix]]
- [[对称矩阵与正定二次型#二次型与正定性|Matrix Square Root]]
- [[对称矩阵与正定二次型#二次型与正定性|Quadratic Form]]
- [[主成分分析 PCA#PCA 的方差最大化|Rayleigh Quotient]]
- [[多元数据、随机向量与样本协方差#数据矩阵与随机向量|Random Vector]]
- [[多元数据、随机向量与样本协方差#均值、协方差与相关|Mean Vector]]
- [[多元数据、随机向量与样本协方差#均值、协方差与相关|Covariance Matrix]]
- [[多元数据、随机向量与样本协方差#均值、协方差与相关|Correlation Matrix]]
