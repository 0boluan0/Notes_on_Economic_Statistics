---
aliases:
  - Thin QR factorization
  - Reduced QR factorization
  - 经济 QR 分解
student_os: knowledge-atom
atom_id: LA-PROJ-026
atom_set: orthogonal-projection-least-squares
atom_type: definition
status: source-checked
requires:
  - "[[QR分解]]"
  - "[[标准正交组]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
leads_to:
  - "[[薄QR最小二乘]]"
related:
  - "[[Gram-Schmidt正交化]]"
---
<!-- bilingual-en:start -->
# Thin QR factorization
<!-- bilingual-en:end -->

*薄 QR 分解*

<!-- bilingual-en:start -->
A **thin QR factorization** keeps the part of a full [[QR分解|QR factorization]] needed to reconstruct a matrix with at least as many rows as columns. It removes the unused trailing directions, while retaining perpendicular unit directions and a square triangular table of coefficients. The word “thin” specifies a shape; it does not by itself certify that the original columns are independent.
<!-- bilingual-en:end -->

**薄 QR 分解**针对行数不少于列数的矩阵，只保留完整 [[QR分解|QR 分解]]中重建该矩阵所需的部分。它去掉末尾未参与重建的方向，保留彼此垂直的单位方向和方形上三角系数表。“薄”首先规定尺寸，并不自动保证原矩阵各列独立。

<!-- bilingual-en:start -->
## Rebuild two columns in three coordinates
<!-- bilingual-en:end -->

## 最小例子

$$
A=\begin{bmatrix}1&1\\0&1\\0&0\end{bmatrix},\qquad
Q_1=\begin{bmatrix}1&0\\0&1\\0&0\end{bmatrix},\qquad
R_1=\begin{bmatrix}1&1\\0&1\end{bmatrix},\qquad A=Q_1R_1.
$$

<!-- bilingual-en:start -->
The first column of $A$ is $q_1=(1,0,0)^T$. The second is $q_1+q_2$, where $q_2=(0,1,0)^T$. Those two coefficient lists, $(1,0)^T$ and $(1,1)^T$, are the columns of $R_1$. The directions $q_1,q_2$ have length 1 and are perpendicular: they are an [[标准正交组|orthonormal family]]. Here $T$ means [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

$A$ 的第一列是 $q_1=(1,0,0)^T$；第二列是 $q_1+q_2$，其中 $q_2=(0,1,0)^T$。两列组合系数 $(1,0)^T$、$(1,1)^T$ 就组成了 $R_1$。方向 $q_1,q_2$ 长度为1且彼此垂直，构成[[标准正交组]]；这里 $T$ 表示[[矩阵转置|转置]]。

<!-- bilingual-en:start -->
A full factorization adds the third direction $q_3=(0,0,1)^T$. Its coefficients in both original columns are zero, so the full triangular factor adds a zero row. Dropping that direction and that row leaves exactly the thin factorization above.
<!-- bilingual-en:end -->

完整分解还会加入第三方向 $q_3=(0,0,1)^T$。原来的两列沿该方向的分量都为零，所以完整三角因子还要添加一行零。删去这一方向和零行，就得到上面的薄分解。

<!-- bilingual-en:start -->
For $A\in\mathbb R^{m\times n}$ with $m\ge n\ge1$, the convention used here is
<!-- bilingual-en:end -->

对 $A\in\mathbb R^{m\times n}$、$m\ge n\ge1$，这里采用的薄分解约定为

$$
A=Q_1R_1,\qquad
Q_1\in\mathbb R^{m\times n},\quad Q_1^TQ_1=I_n,\qquad
R_1\in\mathbb R^{n\times n}\text{ upper triangular}.
$$

<!-- bilingual-en:start -->
The subscript 1 labels the retained block, not the rank. $I_n$ is the [[单位矩阵|identity matrix]]. The [[三角矩阵|upper-triangular]] factor has zeros below its diagonal. In a full factorization, $Q_2$ denotes the other $m-n$ columns, and
<!-- bilingual-en:end -->

下标1标识保留下来的块，并不是秩。$I_n$ 是[[单位矩阵]]；[[三角矩阵|上三角]]因子的对角线下方全为零。若用 $Q_2$ 表示完整分解中其余 $m-n$ 列，则

$$
A=[Q_1\ Q_2]\begin{bmatrix}R_1\\0\end{bmatrix}=Q_1R_1.
$$

<!-- bilingual-en:start -->
When $m=n$, there is no omitted block: thin and full have the same dimensions. For $m<n$, an $m$-dimensional space cannot contain $n$ orthonormal vectors, so this thin shape is unavailable; the wide-matrix form is described in [[QR分解|full QR]].
<!-- bilingual-en:end -->

若 $m=n$，就没有可省略的块，薄分解与完整分解尺寸相同。若 $m<n$，$m$ 维空间容纳不下 $n$ 个标准正交向量，因而不能采用这里的薄形状；宽矩阵形式见 [[QR分解|完整 QR]]。

<!-- bilingual-en:start -->
## Which space do the retained directions span?
<!-- bilingual-en:end -->

## 秩边界

<!-- bilingual-en:start -->
The [[列空间|column space]] $C(A)$ contains all linear combinations of the columns of $A$. Since $A=Q_1R_1$, every such vector belongs to $C(Q_1)$. If the original columns are [[线性无关|linearly independent]]—that is, $A$ has [[矩阵秩|full column rank]] $n$—then $R_1$ is invertible and the two spaces are equal. Indeed, $Ax=0$ is equivalent to $R_1x=0$, because multiplying by $Q_1^T$ recovers $R_1x$. Thus $A$ has no nonzero zero-output coefficient vector exactly when the square $R_1$ is invertible.
<!-- bilingual-en:end -->

[[列空间]] $C(A)$ 包含 $A$ 各列的全部线性组合。由于 $A=Q_1R_1$，这些向量都在 $C(Q_1)$ 中。若原列[[线性无关]]，也就是 $A$ [[矩阵秩|满列秩]]且秩为 $n$，则 $R_1$ 可逆，两个空间相等。原因是：$Ax=0$ 等价于 $R_1x=0$，左乘 $Q_1^T$ 就能恢复 $R_1x$。因此，$A$ 不存在非零的零输出系数，恰好等价于方阵 $R_1$ 可逆。

> [!warning] Thin is not rank-adapted / 薄分解不等于按秩截取
> <!-- bilingual-en:start -->
> Keep the same $Q_1=[(1,0,0)^T\ (0,1,0)^T]$, but take $R_1=\begin{bmatrix}1&1\\0&0\end{bmatrix}$. Then $A=Q_1R_1=\begin{bmatrix}1&1\\0&0\\0&0\end{bmatrix}$ has two identical columns. Its column space is a line, while $C(Q_1)$ is a plane. The factorization still has the prescribed thin shape, but $R_1$ is singular.
> <!-- bilingual-en:end -->
>
> 保持同一 $Q_1$，改用上述第二行全零的 $R_1$，便得到两列相同的 $A$。$C(A)$ 只有一条直线，$C(Q_1)$ 却是一个平面。分解仍符合薄形状，但 $R_1$ 已奇异。

<!-- bilingual-en:start -->
This matters when taking a projection. By the [[标准正交基投影公式|orthonormal-basis formula]], $Q_1Q_1^T$ always projects onto **$C(Q_1)$**. Under full column rank this is also $C(A)$. In the rank-deficient example it leaves $(0,1,0)^T$ unchanged, whereas projection onto $C(A)$ sends that vector to zero. A basis for the actual column space must then keep only as many independent directions as the rank; the $n$-column thin factor cannot simply stand in for it.
<!-- bilingual-en:end -->

这一区别会直接影响投影。根据[[标准正交基投影公式]]，$Q_1Q_1^T$ 总是投影到 **$C(Q_1)$**；满列秩时，这也就是 $C(A)$。但在秩亏例子中，该乘积会保留 $(0,1,0)^T$，而投影到 $C(A)$ 应把它送到零。此时描述实际列空间的基只能保留与秩相同数量的独立方向，不能直接用含 $n$ 列的薄因子代替。

<!-- bilingual-en:start -->
For $m>n$, $Q_1^TQ_1=I_n$ makes $Q_1^T$ a [[左逆|left inverse]], not a two-sided inverse. The reverse product removes directions perpendicular to $C(Q_1)$ and is not $I_m$. In the first example it is the diagonal matrix with diagonal $(1,1,0)$, so it removes the third coordinate.
<!-- bilingual-en:end -->

当 $m>n$ 时，$Q_1^TQ_1=I_n$ 说明 $Q_1^T$ 是[[左逆]]，并非双侧逆。反向乘积会消去垂直于 $C(Q_1)$ 的方向，不是 $I_m$。在第一个例子中，它是对角元为 $(1,1,0)$ 的对角矩阵，会消去第三坐标。

<!-- bilingual-en:start -->
For a real full-column-rank $A$, requiring every diagonal entry of $R_1$ to be positive makes the thin factors unique. Without this convention, reversing a column of $Q_1$ and the corresponding row of $R_1$ leaves their product unchanged. With positive diagonals, Gram–Schmidt fixes the first unit direction along the first column and then each new unit direction along the remaining perpendicular component. Independence makes every such component nonzero. This fixes all retained directions and their coefficients; it does not make the unused full-QR completion $Q_2$ unique.
<!-- bilingual-en:end -->

对实数满列秩 $A$，要求 $R_1$ 的每个对角元为正，就能使薄因子唯一。没有这一约定时，同时反转 $Q_1$ 一列和 $R_1$ 对应一行的符号，乘积仍不变。正对角约定使 Gram–Schmidt 的第一单位方向沿第一原列，其后每个单位方向沿剩下的垂直分量；列独立保证这些分量非零。这就固定了所有保留方向及其系数，但不会使完整 QR 中未使用的补充块 $Q_2$ 也唯一。

<!-- bilingual-en:start -->
The full-column-rank case leads to the triangular solve in [[薄QR最小二乘|thin-QR least squares]]. For complex entries, replace $Q_1^T$ by conjugate transpose $Q_1^*$; the same dimensions and rank distinctions apply with the usual complex inner product.
<!-- bilingual-en:end -->

满列秩情形接下来可用于[[薄QR最小二乘]]的三角求解。对复数元素，把 $Q_1^T$ 换为共轭转置 $Q_1^*$；使用通常复内积时，尺寸与秩的区别保持不变。

## 来源与核验

<!-- bilingual-en:start -->
- [LAPACK Users’ Guide, QR Factorization](https://www.netlib.org/lapack/lug/node40.html): the block reduction $[Q_1\ Q_2][R_1;0]=Q_1R_1$ and invertibility under full column rank.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|MIT 18.06SC Session 2.4 summary, PDF pp.1–3]]: rectangular orthonormal columns, projection onto their own column space, and the Gram–Schmidt coefficient construction. The projector statement uses the correct $P=QQ^T$ line on p.2.
- [Axler, *Linear Algebra Done Right*, 7.58 and its proof, PDF p.278](https://linear.axler.net/LADR4e.pdf#page=278): positive-diagonal uniqueness for square invertible matrices; the same sequential Gram–Schmidt argument above applies to independent rectangular columns. The two displayed rectangular examples and their different projectors are direct calculations.
<!-- bilingual-en:end -->

- LAPACK 核对完整到薄分解的分块关系及满列秩下三角因子可逆。
- MIT Session 2.4（PDF第1–3页）核对长方标准正交列、投影到这些列自身张成的空间，以及 Gram–Schmidt 系数构造；投影采用第2页正确的 $P=QQ^T$ 一行。
- Axler 第7.58条及证明（PDF第278页）给出可逆方阵的正对角唯一性，上文把同一逐列论证用于独立长方列；两个长方例子与投影差异直接计算核验。
