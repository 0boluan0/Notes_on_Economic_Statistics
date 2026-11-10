---
aliases:
  - 奇异值分解的秩一展开
  - Rank-one expansion of the SVD
student_os: knowledge-atom
atom_id: LA-SVD-023
atom_set: singular-value-decomposition-low-rank
atom_type: identity
status: source-checked
requires:
  - "[[紧SVD]]"
  - "[[秩一矩阵外积刻画]]"
  - "[[奇异向量]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[截断SVD]]"
  - "[[奇异向量共同相位]]"
related:
  - "[[矩阵乘积外积展开]]"
---

<!-- bilingual-en:start -->
# Rank-one expansion of the SVD
<!-- bilingual-en:end -->

*SVD 的秩一展开*

<!-- bilingual-en:start -->
An SVD describes a matrix one input–output direction at a time. Each piece reads one input coordinate, scales it, and sends it along one output direction. Adding these rank-one pieces reproduces the whole matrix. This makes each direction's contribution visible and gives a starting point for retaining only some directions.
<!-- bilingual-en:end -->

SVD 按一对输入—输出方向来描述矩阵：每一块读取一个输入坐标，作伸缩，再沿一个输出方向送出。把这些秩一小块相加，就恢复整个矩阵。这让各方向的贡献清晰可见，也为只保留部分方向提供起点。

> [!example] Two separate coordinate actions / 两个坐标分别作用
> <!-- bilingual-en:start -->
> Let $e_1=(1,0)^T$, $e_2=(0,1)^T$. Then
>
> $$A=\begin{pmatrix}3&0\\0&1\end{pmatrix}
> =3e_1e_1^T+e_2e_2^T
> =\begin{pmatrix}3&0\\0&0\end{pmatrix}+\begin{pmatrix}0&0\\0&1\end{pmatrix}.$$
>
> For input $(x_1,x_2)^T$, the first piece produces $(3x_1,0)^T$, the second $(0,x_2)^T$. Each has a one-dimensional output space and hence [[矩阵秩|rank]] one. The sum produces $(3x_1,x_2)^T$. Here $T$ means [[矩阵转置|transpose]].
> <!-- bilingual-en:end -->
>
> 令 $e_1=(1,0)^T$、$e_2=(0,1)^T$，得到上面的分解。输入为 $(x_1,x_2)^T$ 时，第一块输出 $(3x_1,0)^T$，第二块输出 $(0,x_2)^T$；各自输出空间为一维，所以[[矩阵秩|秩]]为1。相加得到 $(3x_1,x_2)^T$，$T$ 表示[[矩阵转置|转置]]。

<!-- bilingual-en:start -->
## Read a coordinate, then produce a direction
<!-- bilingual-en:end -->

*读取坐标，再输出方向*

<!-- bilingual-en:start -->
Let $A$ be a real or complex $m\times n$ matrix, with $m,n\ge1$ and [[奇异值与秩|rank]] $r>0$. Its [[紧SVD|compact SVD]] is $A=U_r\Sigma_rV_r^*$. The $m$-coordinate columns $u_i$ are perpendicular unit output directions, the $n$-coordinate columns $v_i$ perpendicular unit input directions, and $\sigma_1\ge\cdots\ge\sigma_r>0$ their stretches. The star means conjugate transpose, ordinary transpose for real matrices. Multiplying the factors [[矩阵乘积外积展开|column by row]] gives
<!-- bilingual-en:end -->

设 $A$ 为实或复 $m\times n$ 矩阵，$m,n\ge1$，[[奇异值与秩|秩]]为 $r>0$。其[[紧SVD]]为 $A=U_r\Sigma_rV_r^*$：含 $m$ 个坐标的列 $u_i$ 是相互垂直的单位输出方向，含 $n$ 个坐标的列 $v_i$ 是相互垂直的单位输入方向，$\sigma_1\ge\cdots\ge\sigma_r>0$ 是伸缩量。星号为共轭转置，实数时为普通转置。按[[矩阵乘积外积展开|列乘行]]展开，得到

$$A=\sum_{i=1}^{r}\sigma_i u_iv_i^*.$$

<!-- bilingual-en:start -->
The [[秩一矩阵外积刻画|outer product]] $u_iv_i^*$ has size $(m\times1)(1\times n)=m\times n$. Its entry in row $a$, column $b$ is $(u_i)_a\overline{(v_i)_b}$; the bar denotes conjugation and leaves real numbers unchanged. On input $x$,
<!-- bilingual-en:end -->

[[秩一矩阵外积刻画|外积]] $u_iv_i^*$ 的尺寸为 $(m\times1)(1\times n)=m\times n$，第 $a$ 行、第 $b$ 列元素为 $(u_i)_a\overline{(v_i)_b}$；上划线表示共轭，对实数不作改变。作用于输入 $x$ 时，

$$(\sigma_i u_iv_i^*)x=\sigma_i u_i(v_i^*x).$$

<!-- bilingual-en:start -->
The scalar $v_i^*x$ reads the input coordinate along $v_i$. Multiplying it by $\sigma_i u_i$ produces an output on the line spanned by $u_i$. The term sends $v_i$ to the nonzero vector $\sigma_i u_i$, so its rank is exactly one, not merely at most one. The sum formula follows entry by entry from ordinary matrix multiplication.
<!-- bilingual-en:end -->

标量 $v_i^*x$ 读取输入沿 $v_i$ 的坐标，乘 $\sigma_i u_i$ 后，输出落在 $u_i$ 张成的直线上。该项把 $v_i$ 映到非零的 $\sigma_i u_i$，所以秩恰为1，而非仅至多1。求和式逐元素来自普通矩阵乘法。

<!-- bilingual-en:start -->
## The pieces are perpendicular as matrices
<!-- bilingual-en:end -->

*矩阵小块之间也正交*

<!-- bilingual-en:start -->
For matrices of the same size, the Frobenius inner product multiplies corresponding entries, conjugating the first matrix's entries, and adds them:

$$\langle B,C\rangle_F=\sum_{a=1}^{m}\sum_{b=1}^{n}\overline{B_{ab}}C_{ab}.$$

Its squared length $\langle B,B\rangle_F$ is the square of the [[Frobenius范数|Frobenius norm]]. Set $E_i=u_iv_i^*$. Substituting entries separates the two sums:

$$\langle E_i,E_j\rangle_F=(u_i^*u_j)(v_j^*v_i)
=\begin{cases}1,&i=j,\\0,&i\ne j.\end{cases}$$

Thus the unweighted $E_i$ are orthonormal under this matrix inner product. The weighted pieces $\sigma_iE_i$ are perpendicular and have Frobenius lengths $\sigma_i$. They need not occupy disjoint entries: perpendicularity means a zero inner product, not disjoint patterns of nonzero entries.
<!-- bilingual-en:end -->

同尺寸矩阵的 Frobenius 内积，把第一个矩阵的元素取共轭，与第二个矩阵对应元素相乘，再全部相加，定义如上；$\langle B,B\rangle_F$ 是 [[Frobenius范数]]的平方。令 $E_i=u_iv_i^*$，代入后双重求和分离成 $(u_i^*u_j)(v_j^*v_i)$。两组向量都标准正交，故同项为1，不同项为0。

因此未加权的 $E_i$ 在矩阵内积下标准正交，加权项 $\sigma_iE_i$ 相互垂直，Frobenius 长度为 $\sigma_i$。它们不必占据互不重叠的元素位置；正交是内积为零，不是非零位置不能相交。

<!-- bilingual-en:start -->
For example, take $u_1=v_1=(1,1)^T/\sqrt2$ and $u_2=v_2=(1,-1)^T/\sqrt2$. Both outer products have four nonzero entries, yet their entrywise products add to zero. Orthogonality also gives $\|A\|_F^2=\sum_i\sigma_i^2$, discussed in [[Frobenius范数与奇异值|Frobenius norm and singular values]].
<!-- bilingual-en:end -->

例如取 $u_1=v_1=(1,1)^T/\sqrt2$、$u_2=v_2=(1,-1)^T/\sqrt2$，两个外积的四个元素都非零，但对应元素乘积之和仍为零。正交性也给出 $\|A\|_F^2=\sum_i\sigma_i^2$，详见[[Frobenius范数与奇异值]]。

<!-- bilingual-en:start -->
## Fixed matrix, possibly different pieces
<!-- bilingual-en:end -->

*矩阵固定，小块未必唯一*

<!-- bilingual-en:start -->
For $A=0$, rank is zero and the expansion is an empty sum. In a full SVD, zero-weight terms are zero matrices, not rank-one matrices. A general outer-product expansion can contain redundant or cancelling terms; the SVD has exactly $r$ positive pieces because both direction lists are orthonormal.

The matrix need not determine each singular vector uniquely. A [[奇异向量共同相位|common phase change]] of a paired $u_i,v_i$ leaves $u_iv_i^*$ unchanged. When positive singular values repeat, a paired rotation of the bases within that block can change individual pieces while preserving their sum. This expansion describes one valid choice, not a unique list of vectors.
<!-- bilingual-en:end -->

$A=0$ 时秩零，展开是空求和。完整 SVD 中权重为零的项是零矩阵，不是秩一矩阵。一般外积展开可能有多余项或抵消项；SVD 恰有 $r$ 个正项，依靠的是输入、输出两组方向的标准正交性。

矩阵未必唯一确定每个奇异向量：配对的 $u_i,v_i$ 作[[奇异向量共同相位|相同相位变化]]，$u_iv_i^*$ 不变；正奇异值重复时，成对旋转该块的基，甚至会改变各个秩一项，但总和不变。因此展开展示一组有效选择，不声称向量列表唯一。

<!-- bilingual-en:start -->
A [[截断SVD|truncated SVD]] keeps the first $k$ pieces, with $0\le k\le r$. If $k<r$, it changes the matrix by discarding nonzero components; a compact SVD instead keeps all $r$ pieces. Applying an inverse to the retained matrix is a further operation, explained in [[截断SVD反演|TSVD inversion]].
<!-- bilingual-en:end -->

[[截断SVD]]保留前 $k$ 项，$0\le k\le r$。$k<r$ 时舍去非零分量，改变原矩阵；紧 SVD 则保留全部 $r$ 项。对保留矩阵再作反演，是[[截断SVD反演]]讨论的另一项运算。

## 来源与核验

<!-- bilingual-en:start -->
- [Axler, *Linear Algebra Done Right*, 4th ed., §7E, Theorem7.70 and matrix version7.80, PDF287–288 and291](https://linear.axler.net/LADR4e.pdf#page=287): real/complex expansion into paired orthonormal directions and its compact matrix form. The outer-product entry formula and Frobenius orthogonality are direct algebraic consequences.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT18.06SC Session3.5, PDF pp.1,3–4]]: $Av_i=\sigma_i u_i$ and directional reconstruction. [LAPACK's SVD definition](https://netlib.org/lapack/lug/node32.html) confirms rectangular and conjugate-transpose conventions. Examples, the zero case and paired-basis invariance are checked directly.
<!-- bilingual-en:end -->

- Axler 第7E节定理7.70与矩阵版7.80，PDF287–288、291页：核对实复配对标准正交方向的展开及紧矩阵形式；外积元素式与 Frobenius 正交性由此直接推导。
- MIT18.06SC Session3.5第1、3–4页核对 $Av_i=\sigma_i u_i$ 与方向重构；LAPACK 定义核对矩形和共轭转置口径。例子、零边界与配对基变化直接验证。
