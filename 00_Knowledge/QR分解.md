---
aliases:
  - QR factorization
  - QR decomposition
student_os: knowledge-atom
atom_id: LA-PROJ-016
atom_set: orthogonal-projection-least-squares
atom_type: definition
status: source-checked
requires:
  - "[[标准正交组]]"
related:
  - "[[Gram-Schmidt正交化]]"
leads_to:
  - "[[薄QR分解]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---
<!-- bilingual-en:start -->
# QR factorization
<!-- bilingual-en:end -->

*QR 分解*

<!-- bilingual-en:start -->
A **QR factorization** describes the columns of a matrix using perpendicular unit directions. The factor $Q$ supplies those directions; $R$ records how much of each direction is needed to rebuild each original column. This separates the geometry of the coordinate system from the coefficients used in it.
<!-- bilingual-en:end -->

**QR 分解**用彼此垂直的单位方向描述矩阵的各列。因子 $Q$ 提供这些方向，$R$ 记录重建每个原列时，各方向需要多少份。这样就把坐标方向的几何关系与其中的组合系数分开了。

<!-- bilingual-en:start -->
## A vector of length five
<!-- bilingual-en:end -->

## 最小例子

<!-- bilingual-en:start -->
Take the one-column matrix $A=(3,4)^T$, where $T$ denotes [[矩阵转置|transpose]], so the coordinates are written as a column. Its [[欧氏范数|Euclidean length]] is $\sqrt{3^2+4^2}=5$. Dividing by 5 gives a unit vector $q_1=(3/5,4/5)^T$ in the same direction. Choose $q_2=(-4/5,3/5)^T$ perpendicular to it. Both have length 1, and their [[向量点积|dot product]] is $-12/25+12/25=0$.
<!-- bilingual-en:end -->

取单列矩阵 $A=(3,4)^T$，其中 $T$ 表示[[矩阵转置|转置]]，把坐标写成列。它的[[欧氏范数|欧氏长度]]为 $\sqrt{3^2+4^2}=5$。除以5后，得到同方向的单位向量 $q_1=(3/5,4/5)^T$；再取垂直于它的 $q_2=(-4/5,3/5)^T$。两者长度都为1，[[向量点积|点积]]为 $-12/25+12/25=0$。

$$
Q=[q_1\ q_2]=\begin{bmatrix}3/5&-4/5\\4/5&3/5\end{bmatrix},\qquad
R=\begin{bmatrix}5\\0\end{bmatrix},\qquad
QR=5q_1+0q_2=\begin{bmatrix}3\\4\end{bmatrix}=A.
$$

<!-- bilingual-en:start -->
The original vector needs five units in the first direction and none in the second. The zero in $R$ expresses that fact. A **full** QR keeps both directions of the surrounding plane, even though this particular matrix only uses one of them.
<!-- bilingual-en:end -->

原向量沿第一方向需要5份，沿第二方向不需要任何分量；$R$ 中的零正是在表达这件事。**完整 QR** 保留周围平面的两个方向，即使这个矩阵只用到了其中一个。

<!-- bilingual-en:start -->
## Read the two factors
<!-- bilingual-en:end -->

## 分解在做什么

<!-- bilingual-en:start -->
Let $A\in\mathbb R^{m\times n}$ have $m\ge1$ rows and $n\ge1$ columns. A full QR factorization is $A=QR$ with an $m\times m$ [[正交矩阵|orthogonal matrix]] $Q$ and an $m\times n$ upper-trapezoidal matrix $R$. Orthogonal means that the columns of $Q$ form an [[标准正交组|orthonormal basis]] of $\mathbb R^m$: they are perpendicular unit vectors spanning the entire output space. Thus
<!-- bilingual-en:end -->

设 $A\in\mathbb R^{m\times n}$ 有 $m\ge1$ 行、$n\ge1$ 列。完整 QR 分解写成 $A=QR$，其中 $Q$ 是 $m\times m$ 的[[正交矩阵]]，$R$ 是 $m\times n$ 的上梯形矩阵。$Q$ 的各列构成 $\mathbb R^m$ 的[[标准正交组|标准正交基]]：它们是彼此垂直的单位向量，且张成整个输出空间。因此

$$
Q^TQ=QQ^T=I_m,\qquad R=Q^TA.
$$

<!-- bilingual-en:start -->
Here $I_m$ is the [[单位矩阵|identity matrix]], which leaves every $m$-coordinate vector unchanged. Multiplying a coordinate list by $Q$ rebuilds a vector in the original space; multiplying that vector by $Q^T$ reads its coordinates in the new basis. Because $Q$ is square, $Q^T$ is its [[逆矩阵|two-sided inverse]].
<!-- bilingual-en:end -->

$I_m$ 是[[单位矩阵]]，保持每个含 $m$ 个坐标的向量不变。用 $Q$ 乘一列组合系数，会在原空间中合成向量；再左乘 $Q^T$，则读出它在新基下的坐标。由于 $Q$ 是方阵，$Q^T$ 是其[[逆矩阵|双侧逆]]。

<!-- bilingual-en:start -->
Write $a_j$ for column $j$ of $A$, $q_i$ for column $i$ of $Q$, and $r_{ij}$ for the entry in row $i$, column $j$ of $R$. By [[矩阵乘法|matrix multiplication]],
<!-- bilingual-en:end -->

记 $a_j$ 为 $A$ 的第 $j$ 列，$q_i$ 为 $Q$ 的第 $i$ 列，$r_{ij}$ 为 $R$ 第 $i$ 行、第 $j$ 列的元素。按照[[矩阵乘法]]，

$$
a_j=\sum_{i=1}^{m}r_{ij}q_i,\qquad r_{ij}=q_i^Ta_j,\qquad r_{ij}=0\ \text{when }i>j.
$$

<!-- bilingual-en:start -->
The coefficient $r_{ij}$ is the signed amount of direction $q_i$ in the original column $a_j$. **Upper trapezoidal** means the entries below the main diagonal vanish, even when the array is rectangular. Column $j$ therefore uses only the first $\min(j,m)$ directions. For $m\ge n$, the bottom $m-n$ rows are zero and the top $n\times n$ block is [[三角矩阵|upper triangular]]. For $m<n$, the first $m$ columns form an upper-triangular block and the remaining columns form a rectangular block.
<!-- bilingual-en:end -->

$r_{ij}$ 是原列 $a_j$ 沿方向 $q_i$ 的带符号分量。**上梯形**表示主对角线下方的元素全为零，也允许整个数组为长方形。因此第 $j$ 列只使用前 $\min(j,m)$ 个方向。若 $m\ge n$，底部 $m-n$ 行全为零，上方 $n\times n$ 块为[[三角矩阵|上三角]]；若 $m<n$，前 $m$ 列组成上三角块，其余列组成长方块。

> [!note] A change of coordinates preserves length / 换坐标保持长度
> <!-- bilingual-en:start -->
> For every real column vector $y$ with $m$ coordinates, $\|Q^Ty\|_2^2=y^TQQ^Ty=y^Ty=\|y\|_2^2$. Thus QR lets us express a fitting error in convenient coordinates without changing its length. This is the reason it is useful for [[薄QR最小二乘|least squares]].
> <!-- bilingual-en:end -->
>
> 对任意含 $m$ 个坐标的实列向量 $y$，有 $\|Q^Ty\|_2^2=y^TQQ^Ty=y^Ty=\|y\|_2^2$。QR 允许把拟合误差换到方便的坐标中，同时保持长度，这正是它适合[[薄QR最小二乘|最小二乘计算]]的原因。

<!-- bilingual-en:start -->
A full QR exists even when the original columns are dependent or $A$ is wide. It does not assert that the columns of $A$ are a basis. Algorithms such as [[Gram-Schmidt正交化|Gram–Schmidt]], Householder reflections and Givens rotations construct the factors; ordinary Gram–Schmidt applied without a dependence-handling step requires independent input columns. When $m\ge n$, [[薄QR分解|thin QR]] keeps only the first $n$ columns of $Q$ and the corresponding triangular block. That convention and its rank boundary are a separate definition.
<!-- bilingual-en:end -->

即使原列线性相关，或 $A$ 的列数多于行数，也存在完整 QR；它并不声称 $A$ 的各列构成基。[[Gram-Schmidt正交化]]、Householder 反射与 Givens 旋转等算法都可构造因子；直接使用、未加入相关列处理步骤的普通 Gram–Schmidt 要求输入列独立。当 $m\ge n$ 时，[[薄QR分解]]只保留 $Q$ 的前 $n$ 列及对应上三角块；其尺寸约定与秩边界另有定义。

<!-- bilingual-en:start -->
For complex matrices, use a unitary $Q$ and conjugate transpose $Q^*$ in place of $Q^T$: transpose and conjugate every entry. Then $Q^*Q=QQ^*=I_m$, and the norm uses the sum of squared absolute values. Ordinary transpose alone does not provide these complex orthogonality identities; see [[Unitary 矩阵|unitary matrices]].
<!-- bilingual-en:end -->

复矩阵情形使用酉矩阵 $Q$，并把 $Q^T$ 换成共轭转置 $Q^*$：转置后再对每个元素取共轭。此时 $Q^*Q=QQ^*=I_m$，范数采用绝对值平方和。普通转置不能提供这些复数正交关系，见 [[Unitary 矩阵|酉矩阵]]。

## 来源与核验

<!-- bilingual-en:start -->
- [LAPACK Users’ Guide, QR Factorization](https://www.netlib.org/lapack/lug/node40.html): full and thin dimensions for tall and wide matrices, real/complex factors, and the coordinate-based least-squares solve.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|MIT 18.06SC Session 2.4 summary, PDF pp.1–3]]: orthonormal columns, the square inverse identity, Gram–Schmidt and the coefficient relation $R=Q^TA$.
- [Axler, *Linear Algebra Done Right*, 4th ed., 7.56–7.58, PDF pp.277–278](https://linear.axler.net/LADR4e.pdf#page=277): unitary matrices, norm preservation, and QR for independent square columns. The rectangular extension is supported by LAPACK; the $3$–$4$ example is direct multiplication.
<!-- bilingual-en:end -->

- LAPACK 核对高矩阵与宽矩阵的完整/薄分解尺寸、实复因子及换坐标后的最小二乘求解。
- MIT Session 2.4（PDF第1–3页）核对标准正交列、方阵逆关系、Gram–Schmidt 及 $R=Q^TA$。
- Axler 第7.56–7.58条（PDF第277–278页）核对酉矩阵、长度保持及独立方阵列的 QR；长方情形另由 LAPACK 支持，$3$–$4$ 例子直接乘法核验。
