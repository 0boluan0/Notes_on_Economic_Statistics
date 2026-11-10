---
aliases:
- Hand computation of an SVD
- 奇异值分解计算流程
student_os: knowledge-atom
atom_id: LA-SVD-021
atom_set: singular-value-decomposition-low-rank
atom_type: method
status: source-checked
requires:
- '[[SVD存在性]]'
- '[[奇异值]]'
- '[[奇异向量]]'
- '[[奇异值与Gram谱]]'
- '[[奇异向量与Gram特征向量]]'
part_of:
- '[[奇异值分解与低秩近似.canvas]]'
related:
- '[[SVD零块补基不唯一]]'
- '[[SVD与四个基本子空间]]'
- '[[正规方程条件数平方]]'
---

<!-- bilingual-en:start -->
# Computing an SVD by hand
<!-- bilingual-en:end -->

*SVD 手算流程*

<!-- bilingual-en:start -->
A [[奇异值分解|singular value decomposition (SVD)]] describes a matrix by perpendicular input directions, their stretch amounts, and matching output directions. Hand calculation finds these pieces one at a time. The useful trick is to begin with a square Gram matrix, even when the original matrix is rectangular, and then use the original matrix to pair the two sets of directions.
<!-- bilingual-en:end -->

[[奇异值分解]]把矩阵描述成一组相互垂直的输入方向、各自的伸缩量，以及配对的输出方向。手算时逐步找出这些部分。关键技巧是：即使原矩阵为长方形，也先处理一个方形 Gram 矩阵，再用原矩阵把两侧方向配起来。

<!-- bilingual-en:start -->
For a real matrix $A$ with $m\ge1$ rows and $n\ge1$ columns, a full SVD has the form $A=U\Sigma V^T$. The columns of $V$ are the input directions; those of $U$ are the output directions. Both sets are [[标准正交组|orthonormal]]: mutually perpendicular with length 1. The diagonal entries of the $m\times n$ matrix $\Sigma$ are the nonnegative [[奇异值|singular values]], ordered from largest to smallest. Thus $U$ is $m\times m$, $V$ is $n\times n$, and both are [[正交矩阵|orthogonal matrices]]. The superscript $T$ means [[矩阵转置|transpose]].
<!-- bilingual-en:end -->

对有 $m\ge1$ 行、$n\ge1$ 列的实矩阵 $A$，完整 SVD 写成 $A=U\Sigma V^T$。$V$ 的列是输入方向，$U$ 的列是输出方向；两组都[[标准正交组|标准正交]]，即彼此垂直且长度为1。$m\times n$ 矩阵 $\Sigma$ 的主对角线上放非负、降序排列的[[奇异值]]。所以 $U$ 为 $m\times m$，$V$ 为 $n\times n$，二者都是[[正交矩阵]]；上标 $T$ 表示[[矩阵转置|转置]]。

> [!example] A matrix with two inputs and three outputs / 两个输入坐标、三个输出坐标
> <!-- bilingual-en:start -->
> We will compute a full SVD of the following matrix. It sends $(x_1,x_2)^T$ to $(x_1+x_2,x_1-x_2,x_1+x_2)^T$. The first and third outputs agree, so one output direction will never be reached. We should expect two input basis vectors and three output basis vectors.
> <!-- bilingual-en:end -->
>
> 下面矩阵把 $(x_1,x_2)^T$ 送到 $(x_1+x_2,x_1-x_2,x_1+x_2)^T$。第一与第三输出总相同，因此有一个输出方向永远到不了。完整分解需要两条输入基向量、三条输出基向量。
>
> $$A=\begin{pmatrix}1&1\\1&-1\\1&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
## Find the input directions and their stretches
<!-- bilingual-en:end -->

## 找输入方向及伸缩量

<!-- bilingual-en:start -->
Form the [[Gram矩阵正定判据|column Gram matrix]] $G=A^TA$. Its entries are dot products between the columns of $A$. The [[奇异值与Gram谱|Gram-spectrum identity]] says that its positive eigenvalues are squared singular values. An [[特征对|eigenvector]] is a nonzero vector $v$ satisfying $Gv=\lambda v$: $G$ only scales that direction. Here
<!-- bilingual-en:end -->

先形成[[Gram矩阵正定判据|列 Gram 矩阵]] $G=A^TA$，它的元素是 $A$ 各列之间的点积。[[奇异值与Gram谱|Gram 谱恒等式]]说明，它的正特征值是奇异值的平方。[[特征对|特征向量]]是满足下式的非零向量 $v$： $Gv=\lambda v$，表示 $G$ 只沿这个方向缩放。此处

$$
G=\begin{pmatrix}3&1\\1&3\end{pmatrix},\qquad
\det(G-\lambda I_2)=(3-\lambda)^2-1=(\lambda-4)(\lambda-2).
$$

<!-- bilingual-en:start -->
The [[行列式|determinant]] vanishes at the eigenvalues; $I_2$ is the two-dimensional [[单位矩阵|identity matrix]]. The eigenvalues are $4$ and $2$. Solving $(G-\lambda I_2)v=0$ gives directions $(1,1)^T$ and $(1,-1)^T$. Dividing each by its length $\sqrt2$ makes unit vectors. In descending order,
<!-- bilingual-en:end -->

[[行列式]]在特征值处为零，$I_2$ 是二维[[单位矩阵]]。特征值为 $4,2$；解 $(G-\lambda I_2)v=0$ 得方向 $(1,1)^T$、$(1,-1)^T$。各除以长度 $\sqrt2$，得到单位向量。按特征值降序排列，

$$
v_1=\frac1{\sqrt2}\begin{pmatrix}1\\1\end{pmatrix},\quad
v_2=\frac1{\sqrt2}\begin{pmatrix}1\\-1\end{pmatrix},\quad
\sigma_1=2,\quad \sigma_2=\sqrt2.
$$

<!-- bilingual-en:start -->
For a general matrix, choose an orthonormal eigenbasis of $G$, which is possible by the [[Hermitian 谱定理|spectral theorem]]. Distinct eigenspaces are already perpendicular; within a repeated eigenspace, choose perpendicular unit vectors rather than merely normalizing an arbitrary independent list. The selected vectors become the columns of $V$.
<!-- bilingual-en:end -->

一般情形中，由[[Hermitian 谱定理|谱定理]]，可为 $G$ 选一组标准正交特征基。不同特征空间本就垂直；重复特征值对应的空间内部，也要选彼此垂直的单位向量，不能只把任意独立向量各自缩成单位长度。这些向量组成 $V$ 的列。

<!-- bilingual-en:start -->
## Pair the output directions and complete the basis
<!-- bilingual-en:end -->

## 配出输出方向，再补齐基

<!-- bilingual-en:start -->
For each positive singular value, the relation $Av_i=\sigma_i u_i$ determines the matching [[奇异向量|left singular vector]]: $u_i=Av_i/\sigma_i$. In our example,
<!-- bilingual-en:end -->

对每个正奇异值，关系 $Av_i=\sigma_i u_i$ 确定配对的[[奇异向量|左奇异向量]]：$u_i=Av_i/\sigma_i$。本例中

$$
Av_1=\begin{pmatrix}\sqrt2\\0\\\sqrt2\end{pmatrix},\qquad
Av_2=\begin{pmatrix}0\\\sqrt2\\0\end{pmatrix},\qquad
u_1=\frac1{\sqrt2}\begin{pmatrix}1\\0\\1\end{pmatrix},\quad
u_2=\begin{pmatrix}0\\1\\0\end{pmatrix}.
$$

<!-- bilingual-en:start -->
These output vectors have length 1 and are perpendicular. This is not a coincidence: $u_i^Tu_j=v_i^TGv_j/(\sigma_i\sigma_j)$ is 1 when $i=j$ and 0 otherwise. We still need one column for a full $3\times3$ matrix $U$. Choose a unit vector in the [[左零空间|left nullspace]] $N(A^T)=\{u:A^Tu=0\}$, perpendicular to both existing output directions. One choice is $u_3=(1,0,-1)^T/\sqrt2$.
<!-- bilingual-en:end -->

这两条输出向量长度为1且彼此垂直，并非巧合：$u_i^Tu_j=v_i^TGv_j/(\sigma_i\sigma_j)$ 在 $i=j$ 时为1，否则为0。完整的 $3\times3$ 矩阵 $U$ 还缺一列；从[[左零空间]] $N(A^T)=\{u:A^Tu=0\}$ 中选一个单位向量，便与已有输出方向垂直。可以取 $u_3=(1,0,-1)^T/\sqrt2$。

$$
U=\begin{pmatrix}1/\sqrt2&0&1/\sqrt2\\0&1&0\\1/\sqrt2&0&-1/\sqrt2\end{pmatrix},\qquad
\Sigma=\begin{pmatrix}2&0\\0&\sqrt2\\0&0\end{pmatrix},\qquad
V=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix}.
$$

<!-- bilingual-en:start -->
Now verify $U^TU=I_3$, $V^TV=I_2$, and the product $U\Sigma V^T=A$. A short reconstruction is $A=2u_1v_1^T+\sqrt2u_2v_2^T$: its first and third rows are $(1,1)$ and its second row is $(1,-1)$. The unused column $u_3$ contributes zero, but is necessary for a full output basis. The [[紧SVD|compact SVD]] retains only the two positive pairs.
<!-- bilingual-en:end -->

最后核对 $U^TU=I_3$、$V^TV=I_2$，以及乘积 $U\Sigma V^T=A$。可简算重构式 $A=2u_1v_1^T+\sqrt2u_2v_2^T$：第一、三行为 $(1,1)$，第二行为 $(1,-1)$。未参与输出的 $u_3$ 贡献为零，却是完整输出基所需的一列；[[紧SVD]]只保留两对正奇异方向。

<!-- bilingual-en:start -->
## Handle zero values and alternative answers
<!-- bilingual-en:end -->

## 处理零值与不同外观的答案

<!-- bilingual-en:start -->
For any $m\times n$ matrix, let $p=\min(m,n)$ and let $r$ be its [[矩阵秩|rank]], the number of positive singular values. A full $\Sigma$ has $p$ diagonal positions: $r$ positive values followed by $p-r$ zeros. The Gram matrix $A^TA$ has $n$ eigenvectors. If $n>m$, its additional $n-p$ zero eigenvalues do not create extra diagonal positions in $\Sigma$. All $n-r$ zero eigenvectors belong to the [[零空间|input nullspace]] $N(A)$; the remaining $m-r$ columns of $U$ come from $N(A^T)$.
<!-- bilingual-en:end -->

对任意 $m\times n$ 矩阵，令 $p=\min(m,n)$，$r$ 为[[矩阵秩]]，也就是正奇异值个数。完整 $\Sigma$ 有 $p$ 个对角位置：前 $r$ 个为正，其后 $p-r$ 个为零。$A^TA$ 有 $n$ 条特征基向量；当 $n>m$ 时，额外 $n-p$ 个零特征值不会让 $\Sigma$ 多出对角位置。全部 $n-r$ 条零特征向量位于[[零空间|输入零空间]] $N(A)$，而 $U$ 还需从 $N(A^T)$ 补入 $m-r$ 列。

> [!warning] Never divide by a zero singular value / 零奇异值不能作分母
> <!-- bilingual-en:start -->
> For $A=\begin{pmatrix}1&1\\0&0\end{pmatrix}$, take $v_1=(1,1)^T/\sqrt2$, $v_2=(1,-1)^T/\sqrt2$. The singular values are $\sqrt2,0$. The first pair gives $u_1=(1,0)^T$, but $Av_2=0$ gives no direction to normalize. Choose $u_2=(0,1)^T$ from $N(A^T)$. For the zero matrix, both bases may be any orthonormal bases and $\Sigma=0$.
> <!-- bilingual-en:end -->
>
> 对 $A=\begin{pmatrix}1&1\\0&0\end{pmatrix}$，取 $v_1=(1,1)^T/\sqrt2$、$v_2=(1,-1)^T/\sqrt2$，奇异值为 $\sqrt2,0$。第一对给出 $u_1=(1,0)^T$；但 $Av_2=0$ 没有可归一化的方向，应从 $N(A^T)$ 选 $u_2=(0,1)^T$。若整个矩阵为零，两侧可任取标准正交基，$\Sigma=0$。

<!-- bilingual-en:start -->
For a wide matrix it may be easier to start with the smaller $AA^T$: find the left eigenvectors first and use $v_i=A^Tu_i/\sigma_i$ only for positive values. If you diagonalize both Gram matrices separately, their vector signs or repeated-eigenvalue bases need not match. Enforce $Av_i=\sigma_i u_i$ before accepting the answer. Joint sign changes, [[奇异向量共同相位|joint phases]] in complex spaces, and [[SVD零块补基不唯一|different nullspace completions]] can produce different valid factors. With complex entries use conjugate transpose $*$ throughout; orthogonal factors become [[Unitary 矩阵|unitary]].
<!-- bilingual-en:end -->

对宽矩阵，从较小的 $AA^T$ 开始可能更省事：先找左侧特征向量，再仅对正值用 $v_i=A^Tu_i/\sigma_i$。若分别独立对角化两侧 Gram 矩阵，向量符号或重复特征空间中的基未必配对；必须核对 $Av_i=\sigma_i u_i$。共同变号、复空间的[[奇异向量共同相位|共同相位]]及[[SVD零块补基不唯一|不同零空间补基]]都能给出外观不同的正确答案。复矩阵全部使用共轭转置 $*$，正交因子相应成为[[Unitary 矩阵|酉矩阵]]。

<!-- bilingual-en:start -->
## An exact hand method is not a numerical prescription
<!-- bilingual-en:end -->

## 手算构造与数值算法

<!-- bilingual-en:start -->
Forming a Gram matrix can lose information about small singular values through rounding. For full column rank, $A^*A$ is invertible and its two-norm condition number is $\kappa_2(A^*A)=\kappa_2(A)^2$, where $\kappa_2(A)=\sigma_1/\sigma_n$. If $m>n$, the other Gram matrix $AA^*$ is singular, so it has no finite ordinary inverse-based condition number. For full row rank, $\kappa_2(AA^*)=(\sigma_1/\sigma_m)^2$. The [[正规方程条件数平方|normal-equation condition-number result]] explains the full-column-rank case in detail. Numerical SVD routines can reduce $A$ to bidiagonal form directly rather than form its Gram matrix. The present workflow explains an exact construction and is useful for small hand examples; it does not prescribe a stable implementation.
<!-- bilingual-en:end -->

形成 Gram 矩阵时，舍入可能丢掉小奇异值的信息。若 $A$ 满列秩，$A^*A$ 可逆，其二范数条件数满足 $\kappa_2(A^*A)=\kappa_2(A)^2$，其中 $\kappa_2(A)=\sigma_1/\sigma_n$。若 $m>n$，另一侧 $AA^*$ 却是奇异的，没有有限的通常逆矩阵条件数。满行秩时，$\kappa_2(AA^*)=(\sigma_1/\sigma_m)^2$；[[正规方程条件数平方]]详细解释满列秩情形。数值 SVD 程序可直接把 $A$ 化成双对角形，而不先形成 Gram 矩阵。本流程解释精确构造，适合小型手算例子，并非稳定实现的处方。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/07_Recitation_Transcripts/U3_S05_Recitation_Problem_Solving_Computing_the_Singular_Value_Decomposition.pdf|MIT 18.06SC SVD recitation, PDF pp.1–3]]: deriving $V,\Sigma$ from the column Gram matrix, then matching $U$ through $AV=U\Sigma$. The complete rectangular and rank-deficient examples above are calculated directly.
- [Axler, *Linear Algebra Done Right*, 7.64–7.70 and the basis completion following 7.73, PDF pp.284–288](https://linear.axler.net/LADR4e.pdf#page=284): nonnegative Gram spectrum, orthonormal positive pairs and zero-space completion. Axler lists as many singular values as the input dimension; this card uses LAPACK's $\min(m,n)$ matrix convention, with additional Gram zeros counted separately.
- [LAPACK Users' Guide, SVD definition](https://www.netlib.org/lapack/lug/node32.html) and [computational routines](https://www.netlib.org/lapack/lug/node53.html): rectangular factor sizes, real/complex pairing and direct bidiagonal reduction. [MIT 18.335J Week 4, Lecture 9](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-4/): condition-number squaring when forming a full-column-rank Gram matrix.
<!-- bilingual-en:end -->

- MIT 18.06SC SVD 习题讲解第1–3页：从列 Gram 矩阵求 $V,\Sigma$，再由 $AV=U\Sigma$ 配对 $U$；正文长方形与秩亏算例直接计算。
- Axler 第7.64–7.70条及7.73后的补基段，PDF284–288：核对 Gram 非负谱、正值配对及零空间补基。该书按输入维数列奇异值；本卡用 LAPACK 的 $\min(m,n)$ 矩阵口径，额外 Gram 零值单独计数。
- LAPACK 定义与计算章节：核对尺寸、实复配对与双对角化；MIT18.335J Week4 Lecture9：核对满列秩 Gram 矩阵的条件数平方关系。
