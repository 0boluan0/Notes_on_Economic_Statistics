---
aliases:
- Schur decomposition
- Schur 分解
student_os: knowledge-atom
atom_id: LA-EIG-035
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
requires:
- '[[相似矩阵]]'
- '[[Unitary 矩阵]]'
- '[[正交矩阵]]'
related:
- '[[Schur分解存在性]]'
- '[[Normal 矩阵谱定理]]'
- '[[Jordan、Schur与SVD用途边界]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# Schur decomposition
<!-- bilingual-en:end -->

*Schur 分解*

<!-- bilingual-en:start -->
**Schur decomposition** describes a square matrix in perpendicular, unit-length coordinate directions so that its action becomes triangular, or triangular in small blocks. This exposes its eigenvalues without requiring a whole basis of eigenvectors. The new coordinates preserve lengths and angles; the matrix's linear action is unchanged.
<!-- bilingual-en:end -->

**Schur 分解**选取相互垂直、长度为一的坐标方向，让方阵的作用写成三角形，或者由小块组成的三角形。这样可以看出特征值，却不必先找到整组特征向量基。新坐标保持长度与角度，改变的是表示方式，矩阵所描述的线性作用没有改变。

> [!example] Exchange the coordinate directions / 交换两条坐标方向
> <!-- bilingual-en:start -->
> Let $A$ act on two-coordinate column vectors, and let $Q$ exchange their two coordinate directions:
> <!-- bilingual-en:end -->
>
> 让 $A$ 作用于两个坐标的列向量，$Q$ 则交换两条坐标方向：
>
> $$
> A=\begin{pmatrix}2&0\\1&1\end{pmatrix},\qquad
> Q=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
> T=Q^TAQ=\begin{pmatrix}1&1\\0&2\end{pmatrix}.
> $$
>
> <!-- bilingual-en:start -->
> Here $Q^T$ is the [[矩阵转置|transpose]], and $Q^TQ=I_2$, where the [[单位矩阵|identity matrix]] $I_2$ leaves two coordinates unchanged. Thus $Q$ is an [[正交矩阵|orthogonal matrix]], and $A=QTQ^T$. In the new coordinates, input $(z_1,z_2)^T$ produces $(z_1+z_2,2z_2)^T$. The first input direction does not feed into the second output, explaining the zero below the diagonal. The entry above the diagonal still permits the second input to affect the first output.
> <!-- bilingual-en:end -->
>
> 这里 $Q^T$ 是[[矩阵转置|转置]]，且 $Q^TQ=I_2$；[[单位矩阵]] $I_2$ 保持两个坐标不变。因此 $Q$ 是[[正交矩阵]]，且 $A=QTQ^T$。在新坐标中，输入 $(z_1,z_2)^T$ 产生 $(z_1+z_2,2z_2)^T$。第一条输入方向不会进入第二个输出，所以主对角线下方为零；上方的非零元素仍允许第二个输入影响第一个输出。

<!-- bilingual-en:start -->
## Complex triangular form
<!-- bilingual-en:end -->

*复数域中的三角形*

<!-- bilingual-en:start -->
For a complex $n\times n$ matrix $A$, with positive integer $n$, a complex Schur decomposition is
<!-- bilingual-en:end -->

对复数 $n\times n$ 方阵 $A$，其中 $n$ 为正整数，复 Schur 分解是

$$A=QTQ^*,\qquad Q^*Q=QQ^*=I_n.$$

<!-- bilingual-en:start -->
The star denotes conjugate transpose: transpose and also conjugate each entry. The square matrix $Q$ is [[Unitary 矩阵|unitary]], so its columns $q_1,\ldots,q_n$ form an [[标准正交组|orthonormal basis]]. The matrix $T=(t_{ij})$ is [[三角矩阵|upper triangular]], meaning $t_{ij}=0$ whenever the row index $i$ exceeds the column index $j$. All three matrices are $n\times n$.
<!-- bilingual-en:end -->

星号表示共轭转置：转置后再把每个元素取复共轭。方阵 $Q$ 是[[Unitary 矩阵|酉矩阵]]，其列 $q_1,\ldots,q_n$ 构成[[标准正交组|标准正交基]]。$T=(t_{ij})$ 是[[三角矩阵|上三角矩阵]]，即行号 $i$ 大于列号 $j$ 时，$t_{ij}=0$。三个矩阵的尺寸都是 $n\times n$。

<!-- bilingual-en:start -->
If $x=Qz$, then $z$ lists the coordinates of $x$ in that basis, and $Ax=QTz$. Equivalently, $Q^*AQ=T$: first enter the old coordinates, apply $A$, then return to the new coordinates. This is a [[相似矩阵|similarity change of basis]] because $Q^{-1}=Q^*$. The diagonal entries of $T$ are the [[特征对|eigenvalues]] of $A$, including repetitions. They need not be real or positive.
<!-- bilingual-en:end -->

如果 $x=Qz$，$z$ 就是 $x$ 在这组基下的坐标，且 $Ax=QTz$。等价地，$Q^*AQ=T$：先进入旧坐标，施加 $A$，再返回新坐标。由于 $Q^{-1}=Q^*$，这属于[[相似矩阵|相似换基]]。$T$ 的对角元素就是 $A$ 的[[特征对|特征值]]，重复值也照数列出；它们不必为实数，更不要求非负。

<!-- bilingual-en:start -->
## Real arithmetic keeps conjugate pairs in one block
<!-- bilingual-en:end -->

*实数运算把共轭对保留在同一块中*

<!-- bilingual-en:start -->
A real matrix can have nonreal eigenvalues. To keep all factors real, use $A=QTQ^T$ with real orthogonal $Q$ and real **upper quasi-triangular** $T$. This means block upper triangular: entries below the diagonal blocks are zero, while the blocks themselves have size $1\times1$ or $2\times2$. A one-by-one block contains a real eigenvalue. Each two-by-two block represents a nonreal conjugate eigenvalue pair; compute that block's eigenvalues, rather than reading its two diagonal entries as the pair.
<!-- bilingual-en:end -->

实矩阵也可能有非实特征值。为了让全部因子保持为实数，使用 $A=QTQ^T$，其中 $Q$ 为实正交矩阵，$T$ 为实**上准三角矩阵**。这表示块上三角：对角块下方的元素为零，而对角块的尺寸为 $1\times1$ 或 $2\times2$。一阶块包含一个实特征值；每个二阶块表示一对非实共轭特征值，需要计算该块的特征值，不能把它的两个对角元素直接当成这对值。

> [!example] A rotation needs a two-dimensional real block / 旋转需要实二维块
> <!-- bilingual-en:start -->
> The matrix below turns $(x,y)^T$ into $(-y,x)^T$, a quarter-turn:
> <!-- bilingual-en:end -->
>
> 下列矩阵把 $(x,y)^T$ 变成 $(-y,x)^T$，即旋转四分之一圈：
>
> $$R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.$$
>
> <!-- bilingual-en:start -->
> It is already a real Schur form: take $Q=I_2$ and $T=R$. Its characteristic equation is $\lambda^2+1=0$, giving $i,-i$, where $i^2=-1$. The two diagonal zeros are not its eigenvalues. No nonzero real vector keeps its line under this rotation, so real triangular form with one-by-one blocks is impossible. Over complex coordinates it can instead be unitarily diagonalized to the [[对角矩阵|diagonal matrix]] $\operatorname{diag}(i,-i)$, with $i,-i$ on the diagonal and zeros elsewhere.
> <!-- bilingual-en:end -->
>
> 它本身已经是实 Schur 形：取 $Q=I_2$、$T=R$。特征方程为 $\lambda^2+1=0$，得到 $i,-i$，其中 $i^2=-1$；两个对角零并不是它的特征值。没有非零实向量的直线能在这种旋转下保持不变，所以不能用实数的一阶块将它三角化。换用复数坐标后，则可酉对角化为[[对角矩阵]] $\operatorname{diag}(i,-i)$，主对角为 $i,-i$，其余元素为零。

<!-- bilingual-en:start -->
## Schur vectors describe nested invariant spaces
<!-- bilingual-en:end -->

*Schur 向量描述逐步扩大的不变空间*

<!-- bilingual-en:start -->
The columns of $Q$ are called **Schur vectors**. In complex Schur form, $AQ=QT$ says
<!-- bilingual-en:end -->

$Q$ 的列叫 **Schur 向量**。在复 Schur 形中，$AQ=QT$ 逐列表示

$$Aq_j=t_{1j}q_1+\cdots+t_{jj}q_j.$$

<!-- bilingual-en:start -->
Thus $Aq_j$ may mix earlier directions; $q_j$ need not itself be an eigenvector. In the opening example, $q_1=(0,1)^T$ is an eigenvector, but $q_2=(1,0)^T$ is not: $Aq_2=q_1+2q_2$. What remains inside itself is the space spanned by the first $k$ columns, for $1\le k\le n$. This is an [[不变子空间|invariant subspace]]: applying $A$ to any vector in it stays inside it. More compactly, with $Q_k=[q_1\ \cdots\ q_k]$ and $T_k$ the leading $k\times k$ block, $AQ_k=Q_kT_k$.
<!-- bilingual-en:end -->

所以 $Aq_j$ 可以混入前面的方向，$q_j$ 本身未必是特征向量。开头的例子中，$q_1=(0,1)^T$ 是特征向量，但 $q_2=(1,0)^T$ 不是，因为 $Aq_2=q_1+2q_2$。真正保持在自身内部的是前 $k$ 列张成的空间（$1\le k\le n$），即[[不变子空间]]：其中任意向量经过 $A$ 后仍留在里面。简写为 $AQ_k=Q_kT_k$，其中 $Q_k=[q_1\ \cdots\ q_k]$，$T_k$ 为左上方的 $k\times k$ 块。

<!-- bilingual-en:start -->
For real Schur form, the same conclusion holds only when the cut after column $k$ ends at a **whole diagonal-block boundary**. Cutting halfway through the rotation block fails: its first coordinate axis rotates into the second. Neither triangularity nor an orthonormal basis makes every individual Schur vector an eigenvector.
<!-- bilingual-en:end -->

实 Schur 形只有在第 $k$ 列之后恰好结束于**完整对角块的边界**时，才有同样的结论。把旋转块切成一半就会失败：第一条坐标轴被转到第二条。三角结构和标准正交基，都不意味着每个 Schur 向量分别是特征向量。

<!-- bilingual-en:start -->
Every square real or complex matrix admits its corresponding form; [[Schur分解存在性|Schur's existence theorem]] explains why. A complex Schur form can be diagonal exactly when the matrix is [[Normal 矩阵|normal]], meaning $A^*A=AA^*$; this is the [[Normal 矩阵谱定理|normal spectral theorem]]. Schur form also supports numerical eigenvalue computations, but it does not make a sensitive eigenvalue or invariant subspace insensitive. The [[Jordan、Schur与SVD用途边界|Jordan–Schur–SVD comparison]] separates these different purposes.
<!-- bilingual-en:end -->

每个实或复方阵都有对应的这种形式，理由由[[Schur分解存在性]]解释。复 Schur 形能取为对角矩阵，当且仅当原矩阵是 [[Normal 矩阵|normal 矩阵]]，即 $A^*A=AA^*$；这就是 [[Normal 矩阵谱定理]]。Schur 形也用于数值特征值计算，但不会让本来敏感的特征值或不变子空间变得不敏感。[[Jordan、Schur与SVD用途边界]]进一步区分这些不同目的。

## 来源与核验

<!-- bilingual-en:start -->
- [LAPACK Users' Guide, Eigenvalues, Eigenvectors and Schur Factorization](https://netlib.org/lapack/lug/node50.html): complex triangular and real quasi-triangular definitions, Schur vectors, and the numerical eigenproblem context.
- [LAPACK95, Nonsymmetric Eigenproblems](https://netlib.org/lapack95/lug95/node34.html): orthonormal Schur coordinates and invariant-subspace uses. Its leading-column statement is qualified here by whole real diagonal blocks.
- [Axler, Linear Algebra Done Right, 4th ed., 6.38 and 7.31](https://linear.axler.net/LADR4e.pdf#page=218), printed pp.204 and 246 / PDF pp.218 and 260: complex Schur existence and the normal-matrix criterion for unitary diagonalization. The displayed coordinate-swap and rotation examples are verified directly.
<!-- bilingual-en:end -->

- LAPACK 的 Schur 说明核对复三角、实准三角定义、Schur 向量及数值特征问题背景。
- LAPACK95 非对称特征问题说明核对标准正交 Schur 坐标与不变子空间用途；本卡给前导列结论补上实数完整对角块限定。
- Axler 6.38与7.31，印刷204、246页／PDF218、260页，核对复 Schur 存在性与酉对角化的 normal 条件。交换坐标与旋转例子由显示矩阵直接核验。
