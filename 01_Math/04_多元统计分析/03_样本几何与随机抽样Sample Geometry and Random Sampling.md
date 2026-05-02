# 1. 第3章：样本几何与随机抽样（Sample Geometry and Random Sampling）

>[!summary] 本章主线
> 本章把样本矩阵看成几何对象：每个变量是一根中心化后的偏差向量，协方差来自偏差向量内积，相关系数来自偏差向量夹角，广义方差来自这些向量张成的面积或体积。

## 1.1. 样本的几何结构

### 1.1.1. 数据矩阵表示

数据矩阵为
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

### 1.1.2. 均值修正向量与偏差向量

第 $i$ 个变量的观测向量为
$$
y_i=
\begin{bmatrix}
x_{1i}\\x_{2i}\\\vdots\\x_{ni}
\end{bmatrix}.
$$

其偏差向量为
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

### 1.1.3. 偏差平方和

变量 $i$ 的偏差平方和为
$$
L^2(d_i)=d_i'd_i=\sum_{j=1}^n(x_{ji}-\bar x_i)^2.
$$

这就是该变量围绕均值的总偏离程度。

### 1.1.4. 偏差向量夹角与相关系数

两个变量 $i,k$ 的偏差向量内积为
$$
d_i'd_k=\sum_{j=1}^n(x_{ji}-\bar x_i)(x_{jk}-\bar x_k).
$$

夹角余弦为
$$
\cos(\theta_{ik})
=\frac{d_i'd_k}{L(d_i)L(d_k)}
=\frac{s_{ik}}{\sqrt{s_{ii}s_{kk}}}
=r_{ik}.
$$

>[!tip] 关键结论
> 样本相关系数就是两个中心化变量向量夹角的余弦。

## 1.2. 随机样本与样本矩

### 1.2.1. 样本协方差矩阵

样本协方差矩阵 $S=((s_{ik}))$ 的元素为
$$
s_{ik}
=\frac{1}{n-1}\sum_{j=1}^n(x_{ji}-\bar x_i)(x_{jk}-\bar x_k)
=\frac{1}{n-1}d_i'd_k.
$$

令偏差矩阵
$$
D=
\begin{bmatrix}
|&|&&|\\
d_1&d_2&\cdots&d_p\\
|&|&&|
\end{bmatrix},
$$
则
$$
S=\frac{1}{n-1}D'D.
$$

>[!warning] 易错点
> $D$ 是中心化后的 deviation matrix。协方差矩阵不是直接由未中心化的 $X'X$ 得到。

### 1.2.2. 样本均值向量的期望

若 $X_1,\ldots,X_n$ 是来自总体均值 $\mu$ 的随机样本，则
$$
E(\bar X)=\mu.
$$

### 1.2.3. 样本协方差矩阵的期望

在常规随机抽样条件下，
$$
E(S)=\Sigma.
$$

这解释了为什么分母用 $n-1$。

## 1.3. 广义方差

### 1.3.1. 行列式作为广义方差

样本广义方差定义为
$$
|S|.
$$

它把多维变异压缩成一个数。

二维情形：
$$
|S|=s_{11}s_{22}-s_{12}^2
=s_{11}s_{22}(1-r_{12}^2).
$$

当 $|r_{12}|=1$ 时，$|S|=0$，说明两个变量完全线性相关，二维面积退化为 0。

### 1.3.2. 迹作为总变异

另一种汇总变异的方式是迹：
$$
\operatorname{tr}(S)=s_{11}+s_{22}+\cdots+s_{pp}.
$$

它只加总各变量自己的方差，不惩罚变量之间的相关性。

>[!note] 对比
> $|S|$ 看联合体积，受相关性影响；$\operatorname{tr}(S)$ 看总方差，不直接看变量之间是否重叠。

## 1.4. 广义方差的几何解释

### 1.4.1. 二维面积

当 $p=2$ 时，两个偏差向量张成平行四边形。面积为
$$
\operatorname{Area}
=(n-1)\sqrt{s_{11}s_{22}(1-r_{12}^2)}
=(n-1)|S|^{1/2}.
$$

若两个变量正交，面积较大；若两个变量完全相关，面积为 0。

### 1.4.2. 高维超体积

当 $p>2$ 时，$d_1,\ldots,d_p$ 张成高维体积。

有
$$
|S|=(\operatorname{Volume})^2(n-1)^{-p}.
$$

### 1.4.3. 秩亏缺情形

若 $n\leq p$，或某个变量是其他变量的线性组合，则 $S$ 可能秩亏：
$$
|S|=0.
$$

此时很多需要 $S^{-1}$ 的方法会失效。

可选处理：

1. 删除冗余变量。
2. 先做 PCA 降维。
3. 使用正则化协方差矩阵。
4. 在尺度问题明显时改用相关矩阵。

## 1.5. 样本协方差矩阵的矩阵表示

### 1.5.1. 中心化矩阵

令
$$
H=I-\frac1n\mathbf1\mathbf1'.
$$

中心化后的数据矩阵可写作
$$
D=HX.
$$

所以
$$
S=\frac{1}{n-1}X'HX
=\frac{1}{n-1}D'D.
$$

### 1.5.2. 行列式与特征值

若 $S$ 的特征值为 $\lambda_1,\ldots,\lambda_p$，则
$$
|S|=\prod_{i=1}^p\lambda_i,
\qquad
\operatorname{tr}(S)=\sum_{i=1}^p\lambda_i.
$$

在 PCA 中，$\lambda_i$ 表示第 $i$ 个主成分方向上的样本方差。

## 1.6. 标准化协方差矩阵与相关矩阵

### 1.6.1. 标准化

变量 $X_j$ 的标准化形式为
$$
Z_j=\frac{X_j-\bar X_j}{\sqrt{s_{jj}}}.
$$

标准化后均值为 0，方差为 1。

### 1.6.2. 样本相关矩阵

若 $Z$ 是标准化后的数据矩阵，则
$$
R=\frac{1}{n-1}Z'Z.
$$

元素为
$$
r_{ij}=\frac{s_{ij}}{\sqrt{s_{ii}s_{jj}}}.
$$

### 1.6.3. 协方差矩阵与相关矩阵转换

令
$$
D_s^{1/2}=\operatorname{diag}(\sqrt{s_{11}},\ldots,\sqrt{s_{pp}}).
$$

则
$$
R=D_s^{-1/2}SD_s^{-1/2},
$$
$$
S=D_s^{1/2}RD_s^{1/2}.
$$

## 1.7. 关联卡片

- [[Sample Mean Vector]]
- [[Sample Covariance Matrix]]
- [[Generalized Variance]]
- [[Correlation Matrix]]
- [[Choosing Covariance vs Correlation Matrix]]
- [[PCA]]
- [[Matrix Rank]]
