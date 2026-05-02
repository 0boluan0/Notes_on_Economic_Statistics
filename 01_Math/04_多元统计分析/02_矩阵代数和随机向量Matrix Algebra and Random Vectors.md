# 1. 第2章：矩阵代数与随机向量（Matrix Algebra and Random Vectors）

>[!summary] 本章主线
> 多元统计把“多个变量一起变化”写成向量和矩阵。本章是后续多元正态、Hotelling $T^2$、PCA 和因子分析的线性代数入口。

## 1.1. 矩阵与向量代数基础

### 1.1.1. 向量的定义与表示

列向量写作
$$
x=
\begin{bmatrix}
x_1\\x_2\\\vdots\\x_n
\end{bmatrix},
\qquad
x'=(x_1,x_2,\ldots,x_n).
$$

向量长度为
$$
L_x=\sqrt{x'x}=\sqrt{x_1^2+\cdots+x_n^2}.
$$

单位向量为
$$
x^*=\frac{x}{L_x}.
$$

### 1.1.2. 夹角、内积与正交

两个向量 $x,y$ 的夹角 $\theta$ 满足
$$
\cos\theta=\frac{x'y}{L_xL_y}.
$$

当 $x'y=0$ 时，两个向量正交。

>[!note] 几何直觉
> 相关系数可以看成两个中心化变量向量夹角的余弦。第 3 章会把这个直觉用于样本几何。

### 1.1.3. 线性相关与投影

向量组 $\{x_1,\ldots,x_k\}$ 线性相关，指存在不全为 0 的系数 $c_1,\ldots,c_k$，使
$$
c_1x_1+\cdots+c_kx_k=0.
$$

向量 $x$ 在 $y$ 上的投影为
$$
\operatorname{proj}_y(x)=\frac{x'y}{y'y}y.
$$

## 1.2. 矩阵（Matrices）

### 1.2.1. 矩阵的定义

矩阵是按行列排列的数值数组：
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

### 1.2.2. 常见特殊矩阵

单位矩阵：
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
$$
D=\operatorname{diag}(d_1,\ldots,d_p).
$$

若 $d_i\neq0$，则
$$
D^{-1}=\operatorname{diag}\left(\frac1{d_1},\ldots,\frac1{d_p}\right).
$$

### 1.2.3. 矩阵运算

矩阵加法要求维度相同。

矩阵乘法要求内维匹配：
$$
A_{n\times p}B_{p\times m}=C_{n\times m},
\qquad
c_{ik}=\sum_{j=1}^p a_{ij}b_{jk}.
$$

转置把行列互换：
$$
(A')_{ij}=A_{ji}.
$$

对称矩阵满足
$$
A'=A.
$$

### 1.2.4. 逆矩阵

若方阵 $A$ 存在 $A^{-1}$，使
$$
AA^{-1}=A^{-1}A=I,
$$
则 $A$ 可逆。

>[!warning] 多元统计里的关键边界
> 总体协方差矩阵常假设为正定，因此可逆；样本协方差矩阵不一定可逆，尤其当 $n\leq p$ 或变量完全线性相关时。

### 1.2.5. 正交矩阵

若
$$
Q'Q=QQ'=I,
$$
则 $Q$ 是正交矩阵。

正交矩阵满足
$$
Q^{-1}=Q'.
$$

它的列向量构成标准正交组。

### 1.2.6. 矩阵的秩

矩阵 $A$ 的秩是线性无关列向量的最大个数：
$$
\operatorname{rank}(A)\leq\min(n,p).
$$

满秩表示达到最大可能秩；否则称为秩亏。

### 1.2.7. 矩阵的迹

方阵 $A$ 的迹为对角线元素之和：
$$
\operatorname{tr}(A)=\sum_{i=1}^n a_{ii}.
$$

常用性质：

1. $\operatorname{tr}(A+B)=\operatorname{tr}(A)+\operatorname{tr}(B)$。
2. $\operatorname{tr}(cA)=c\operatorname{tr}(A)$。
3. $\operatorname{tr}(A)=\operatorname{tr}(A')$。
4. $\operatorname{tr}(AB)=\operatorname{tr}(BA)$。

## 1.3. 行列式（Determinants）

### 1.3.1. 定义与计算

二维方阵的行列式：
$$
\begin{vmatrix}
a&b\\c&d
\end{vmatrix}
=ad-bc.
$$

对角矩阵行列式为对角线元素之积：
$$
|D|=\prod_{i=1}^p d_i.
$$

### 1.3.2. 核心性质

1. $|AB|=|A||B|$。
2. 若 $A$ 可逆，则 $|A^{-1}|=1/|A|$。
3. $|A'|=|A|$。
4. 若 $|A|\neq0$，则 $A$ 可逆。

正交矩阵满足
$$
|Q|=\pm1.
$$

### 1.3.3. 与特征值的关系

若 $A$ 的特征值为 $\lambda_1,\ldots,\lambda_p$，则
$$
|A|=\prod_{i=1}^p\lambda_i,
\qquad
\operatorname{tr}(A)=\sum_{i=1}^p\lambda_i.
$$

>[!note] 后续用途
> 协方差矩阵的行列式就是 [[Generalized Variance]]；它衡量多维数据联合变异的体积。

## 1.4. 特征值与特征向量

### 1.4.1. 定义

若存在非零向量 $x$ 和标量 $\lambda$，使
$$
Ax=\lambda x,
$$
则 $\lambda$ 是 $A$ 的特征值，$x$ 是对应特征向量。

特征值由
$$
|A-\lambda I|=0
$$
求得。

### 1.4.2. 几何意义

特征向量是在矩阵作用下方向不变的方向；特征值表示该方向上的伸缩倍数。

当 $A$ 是实对称矩阵时，不同特征值对应的特征向量可取为正交。

### 1.4.3. 谱分解（Spectral Decomposition）

若 $A$ 是实对称矩阵，则
$$
A=Q\Lambda Q',
$$
其中 $Q$ 是正交特征向量矩阵，$\Lambda$ 是特征值对角矩阵。

也可写作
$$
A=\sum_{i=1}^p\lambda_i q_iq_i'.
$$

>[!tip] 复习连接
> PCA 正是对协方差矩阵做谱分解：特征向量给方向，特征值给该方向的方差。

### 1.4.4. 奇异值分解（SVD）

对任意矩阵 $A$，
$$
A=U\Sigma V'.
$$

其中：

- $V$ 给输入空间的正交方向；
- $\Sigma$ 给非负奇异值；
- $U$ 给输出空间的正交方向。

奇异值满足
$$
\sigma_i=\sqrt{\lambda_i(A'A)}.
$$

SVD 可用于读秩、做低秩近似和处理不可逆问题。

## 1.5. 正定与半正定矩阵

### 1.5.1. 定义

对称矩阵 $A$ 若对任意非零向量 $x$ 满足
$$
x'Ax>0,
$$
则为正定矩阵。

若满足
$$
x'Ax\geq0,
$$
则为半正定矩阵。

### 1.5.2. 特征值判定

对称矩阵 $A$：

- 所有特征值 $>0$ 当且仅当 $A$ 正定；
- 所有特征值 $\geq0$ 当且仅当 $A$ 半正定。

### 1.5.3. 证明思路

若 $A=Q\Lambda Q'$，令 $y=Q'x$，则
$$
x'Ax=y'\Lambda y=\sum_{i=1}^p\lambda_i y_i^2.
$$

所以特征值全正时，任意非零 $x$ 都使二次型为正。

## 1.6. 矩阵平方根与二次型

### 1.6.1. 矩阵平方根（Matrix Square Root）

若 $A$ 是对称正定矩阵，且
$$
A=Q\Lambda Q',
$$
则
$$
A^{1/2}=Q\Lambda^{1/2}Q',
$$
满足
$$
A^{1/2}A^{1/2}=A.
$$

逆平方根为
$$
A^{-1/2}=Q\Lambda^{-1/2}Q'.
$$

### 1.6.2. 二次型（Quadratic Form）

形如
$$
x'Ax
$$
的标量称为二次型。

它在多元正态中形成马哈拉诺比斯距离：
$$
(X-\mu)'\Sigma^{-1}(X-\mu).
$$

### 1.6.3. Rayleigh 商

Rayleigh 商定义为
$$
R(x)=\frac{x'Bx}{x'x}.
$$

若 $B$ 的特征值满足
$$
\lambda_1\geq\lambda_2\geq\cdots\geq\lambda_p,
$$
则
$$
\lambda_p\leq R(x)\leq\lambda_1.
$$

最大值在最大特征值对应的特征向量方向取得。

## 1.7. 随机向量与随机矩阵

### 1.7.1. 随机向量

随机向量写作
$$
X=
\begin{bmatrix}
X_1\\X_2\\\vdots\\X_p
\end{bmatrix}.
$$

均值向量为
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
$$
E(Y)=AE(X)+b.
$$

### 1.7.2. 协方差矩阵

$$
\Sigma=\operatorname{Cov}(X)=E[(X-\mu)(X-\mu)'].
$$

展开为
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

若 $b$ 是常数向量，则
$$
\operatorname{Var}(b'X)=b'\Sigma b.
$$

这表示随机向量在方向 $b$ 上的分布宽度。

### 1.7.4. 独立与不相关

独立一定推出不相关：
$$
X_i\perp X_j\Rightarrow \operatorname{Cov}(X_i,X_j)=0.
$$

但一般情况下，不相关不一定推出独立。

>[!note] 特例
> 在多元正态分布中，零协方差可以推出独立。

## 1.8. 相关矩阵（Correlation Matrix）

相关系数定义为
$$
\rho_{ij}
=\frac{\operatorname{Cov}(X_i,X_j)}
{\sqrt{\operatorname{Var}(X_i)\operatorname{Var}(X_j)}}
=\frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}}.
$$

相关矩阵为
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
$$
\rho=D^{-1/2}\Sigma D^{-1/2},
\qquad
\Sigma=D^{1/2}\rho D^{1/2}.
$$

>[!tip] 选择矩阵
> 变量量纲差异大时，用相关矩阵更稳；量纲相同且方差大小本身有意义时，用协方差矩阵更自然。

## 1.9. 关联卡片

- [[Matrix Operations]]
- [[Matrix Inverse]]
- [[Matrix Rank]]
- [[Determinant]]
- [[Eigenvalues]]
- [[Eigenvectors]]
- [[Spectral Decomposition]]
- [[Singular Value Decomposition]]
- [[Positive Definite Matrix]]
- [[Matrix Square Root]]
- [[Quadratic Form]]
- [[Rayleigh Quotient]]
- [[Random Vector]]
- [[Mean Vector]]
- [[Covariance Matrix]]
- [[Correlation Matrix]]
