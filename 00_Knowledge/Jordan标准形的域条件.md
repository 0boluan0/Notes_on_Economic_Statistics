---
aliases:
  - Jordan canonical form field condition
  - Jordan form splitting criterion
  - Jordan 标准形的域条件
student_os: knowledge-atom
atom_id: LA-EIG-014
atom_set: eigenvalues-linear-dynamics
atom_type: boundary
status: source-checked
requires:
  - "[[Jordan标准形]]"
  - "[[特征多项式底层域]]"
  - "[[相似保持特征多项式]]"
  - "[[Jordan链]]"
related:
  - "[[Jordan形唯一性]]"
  - "[[Jordan、Schur与SVD用途边界]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# The field condition for Jordan form
<!-- bilingual-en:end -->

*Jordan 标准形的域条件*

<!-- bilingual-en:start -->
Whether a matrix has an ordinary [[Jordan标准形|Jordan form]] depends on which scalars its coordinates are allowed to use. A field is that chosen number system, such as the real numbers $\mathbb R$ or the complex numbers $\mathbb C$. Every [[Jordan块|Jordan block]] has an eigenvalue on its diagonal, so all the required eigenvalues must belong to the chosen field. The precise test is that the characteristic polynomial splits completely there.
<!-- bilingual-en:end -->

一个矩阵能否写成普通 [[Jordan标准形]]，取决于坐标允许使用哪些标量。这里的“域”是选定的数系，例如实数 $\mathbb R$ 或复数 $\mathbb C$。每个 [[Jordan块]]都把一个特征值放在对角线上，因此所需的全部特征值必须属于当前数域。精确的检验是：特征多项式必须在该域中完全分裂。

<!-- bilingual-en:start -->
## A real rotation needs complex coordinates
<!-- bilingual-en:end -->

*实旋转为什么需要复坐标*

<!-- bilingual-en:start -->
Consider the quarter-turn matrix
<!-- bilingual-en:end -->

考虑把平面转过四分之一圈的矩阵

$$R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}.$$

<!-- bilingual-en:start -->
It sends $(x,y)$ to $(-y,x)$. Its [[特征多项式|characteristic polynomial]], whose roots are eigenvalues, is
<!-- bilingual-en:end -->

它把 $(x,y)$ 送到 $(-y,x)$。它的[[特征多项式]]的根就是特征值，这个多项式为

$$p_R(t)=\det\begin{pmatrix}t&1\\-1&t\end{pmatrix}=t^2+1.$$

<!-- bilingual-en:start -->
Here $t$ is the polynomial variable and $\det$ denotes the [[行列式|determinant]]. For real $t$, $t^2+1>0$, so there is no real root. No real change of basis can turn $R$ into a matrix made of ordinary real Jordan blocks: there are no real eigenvalues to put on their diagonals.
<!-- bilingual-en:end -->

这里 $t$ 是多项式变量，$\det$ 表示[[行列式]]。对实数 $t$，$t^2+1>0$，所以没有实根。任何实换基都不能把 $R$ 变成由普通实 Jordan 块组成的矩阵，因为没有实特征值可填入它们的对角线。

> [!example] The same entries, a larger scalar field · 条目相同，允许的标量扩大
> <!-- bilingual-en:start -->
> Over $\mathbb C$, let $i^2=-1$. Now $t^2+1=(t-i)(t+i)$. The columns of
> $$S=\begin{pmatrix}1&1\\-i&i\end{pmatrix}$$
> are eigenvectors for $i$ and $-i$, respectively. Since $\det S=2i\ne0$, $S$ is invertible; $S^{-1}$ denotes its [[逆矩阵|inverse]]. Direct multiplication gives
> $$RS=S\begin{pmatrix}i&0\\0&-i\end{pmatrix},\qquad
> S^{-1}RS=\operatorname{diag}(i,-i).$$
> The two diagonal entries are size-one Jordan blocks. The matrix $R$ has not changed; the allowed coordinates and basis vectors have changed.
> <!-- bilingual-en:end -->
>
> 到 $\mathbb C$ 中，令 $i^2=-1$，便有 $t^2+1=(t-i)(t+i)$。显示矩阵 $S$ 的两列分别是 $i$、$-i$ 的特征向量。由于 $\det S=2i\ne0$，$S$ 可逆，$S^{-1}$ 表示它的[[逆矩阵]]；直接相乘给出上面的两条等式。两个对角元各是一阶 Jordan 块。$R$ 的条目没有改变，改变的是允许采用的坐标与基向量。

<!-- bilingual-en:start -->
## What “splits over the field” means
<!-- bilingual-en:end -->

*“在当前域上分裂”的准确含义*

<!-- bilingual-en:start -->
Let $A\in\mathbb F^{n\times n}$ with $n\ge1$: $A$ is a square matrix with entries in the chosen field $\mathbb F$. Write $I_n$ for the [[单位矩阵|identity matrix]] and $p_A(t)=\det(tI_n-A)$. Splitting means that this degree-$n$ polynomial can be written
<!-- bilingual-en:end -->

设 $A\in\mathbb F^{n\times n}$，$n\ge1$，即 $A$ 是条目属于所选域 $\mathbb F$ 的方阵。用 $I_n$ 表示[[单位矩阵]]，并令 $p_A(t)=\det(tI_n-A)$。“分裂”指这个 $n$ 次多项式能写成

$$p_A(t)=\prod_{k=1}^{m}(t-\lambda_k)^{a_k},\qquad
\lambda_k\in\mathbb F,\quad a_k\ge1,\quad a_1+\cdots+a_m=n.$$

<!-- bilingual-en:start -->
The $m$ values $\lambda_k$ are distinct roots in the field; the positive integers $a_k$ count their repetitions. A factor such as $(t-3)^2$ is allowed. The product must account for the **whole** polynomial, not just one available root. See [[特征多项式底层域|how roots depend on the field]] for this distinction.
<!-- bilingual-en:end -->

$m$ 个 $\lambda_k$ 是域中互不相同的根，正整数 $a_k$ 记录它们的重复次数。允许出现 $(t-3)^2$ 这样的重因子。乘积必须覆盖**整个**多项式，不能只找到一个根就算完成；这个区别也见[[特征多项式底层域]]。

> [!important] Necessary and sufficient · 充要条件
> <!-- bilingual-en:start -->
> There exists an invertible $S\in\mathbb F^{n\times n}$ such that $S^{-1}AS$ is a block diagonal matrix of ordinary Jordan blocks with eigenvalues in $\mathbb F$ **if and only if** $p_A(t)$ splits into linear factors over $\mathbb F$. The entries of the change-of-basis matrix must belong to that same field.
> <!-- bilingual-en:end -->
>
> 存在可逆 $S\in\mathbb F^{n\times n}$，使 $S^{-1}AS$ 为由特征值属于 $\mathbb F$ 的普通 Jordan 块组成的块对角矩阵，**当且仅当** $p_A(t)$ 在 $\mathbb F$ 中分解为一次因子的乘积。换基矩阵的条目也必须属于同一个域。

<!-- bilingual-en:start -->
## Why both directions hold
<!-- bilingual-en:end -->

*为什么两个方向都成立*

<!-- bilingual-en:start -->
For necessity, suppose such a Jordan form already exists. A block $J_r(\lambda)$ has characteristic polynomial $(t-\lambda)^r$, because $tI_r-J_r(\lambda)$ is triangular with $t-\lambda$ on every diagonal position. A block diagonal matrix multiplies these factors together. Since [[相似保持特征多项式|similarity preserves the characteristic polynomial]], $A$ must have exactly that product, whose roots all lie in $\mathbb F$.
<!-- bilingual-en:end -->

先看必要性。若已经有这样的 Jordan 形，每个块 $J_r(\lambda)$ 的特征多项式都是 $(t-\lambda)^r$，因为 $tI_r-J_r(\lambda)$ 是对角元全为 $t-\lambda$ 的三角矩阵。块对角矩阵再把各块的这些因子相乘。由于[[相似保持特征多项式]]，$A$ 必须有相同乘积，其中全部根都属于 $\mathbb F$。

<!-- bilingual-en:start -->
Sufficiency is the deeper part of the theorem. When the polynomial splits, the space can be separated into generalized eigenspaces. For a root $\lambda$, the corresponding space is $\ker\bigl((A-\lambda I_n)^n\bigr)$: it contains the vectors eventually sent to zero by repeated applications of $A-\lambda I_n$. Here $\ker$ is the [[零空间|nullspace]], the set of inputs sent to zero. On this subspace, $A-\lambda I_n$ is nilpotent, meaning that some positive power is zero. The theorem constructs a basis from [[Jordan链|Jordan chains]] for each such part. Putting these chain bases together produces an invertible $S$ over the same field and hence the Jordan blocks. Finding ordinary eigenvectors alone need not provide enough basis vectors.
<!-- bilingual-en:end -->

充分性是定理较深的一半。多项式分裂后，空间可以拆成各个广义特征空间。对于根 $\lambda$，相应空间是 $\ker\bigl((A-\lambda I_n)^n\bigr)$，包含反复施加 $A-\lambda I_n$ 后最终归零的向量。这里 $\ker$ 是[[零空间]]，即被送到零的全部输入。在这个子空间上，$A-\lambda I_n$ 是幂零算子，意思是某个正整数次幂为零。定理为每一部分构造由 [[Jordan链]]组成的基，再把各条链的基合在一起，就得到同一域上的可逆 $S$ 和相应的 Jordan 块。只找普通特征向量，未必够组成整组基。

<!-- bilingual-en:start -->
## Splitting is weaker than diagonalization
<!-- bilingual-en:end -->

*分裂不等于可对角化*

<!-- bilingual-en:start -->
For $J_2(3)=\begin{pmatrix}3&1\\0&3\end{pmatrix}$, the characteristic polynomial $(t-3)^2$ splits over $\mathbb R$, and the matrix is already in Jordan form. Nevertheless, it has only one independent eigenvector direction, so it cannot be [[对角化|diagonalized]]. Splitting permits nontrivial chains; diagonalization requires every Jordan block to have size one. A zero eigenvalue is also allowed: $J_2(0)$ already has an ordinary Jordan form although it is singular.
<!-- bilingual-en:end -->

对 $J_2(3)=\begin{pmatrix}3&1\\0&3\end{pmatrix}$，特征多项式 $(t-3)^2$ 在 $\mathbb R$ 中分裂，矩阵本来就是 Jordan 形。但它只有一个独立特征方向，所以不能[[对角化|对角化]]。分裂允许非平凡链，可对角化则要求每个 Jordan 块都只有一阶。零特征值同样允许：$J_2(0)$ 虽然奇异，却已经是普通 Jordan 形。

<!-- bilingual-en:start -->
Every nonconstant complex polynomial factors into linear factors, so every complex square matrix passes the test. Over $\mathbb R$, the test holds exactly when all characteristic roots are real. The chosen field need not make **every** polynomial split: for a given $A$, only its own characteristic polynomial matters. For example, a diagonal matrix with rational entries already has a Jordan form over the rational numbers.
<!-- bilingual-en:end -->

每个非常数复多项式都能分解成一次因子，因此每个复方阵都通过这个检验。对 $\mathbb R$，检验恰好要求全部特征根为实数。所选域不必让**所有**多项式都分裂：对于给定 $A$，只需它自身的特征多项式分裂。例如条目为有理数的对角矩阵，在有理数域中就已经有 Jordan 形。

<!-- bilingual-en:start -->
If a real matrix has nonreal roots and the computation must remain real, representations such as the [[Schur分解|real Schur form]] use real $2\times2$ blocks for conjugate pairs. Those are not ordinary scalar Jordan blocks. This is a different representation choice, not a failure of the splitting criterion. Once a Jordan form exists, [[Jordan形唯一性|its block-data uniqueness]] is a separate theorem.
<!-- bilingual-en:end -->

若实矩阵含非实根而计算必须留在实数域，[[Schur分解|实 Schur 形]]等表示会用实 $2\times2$ 块承载共轭复根。这些并不是普通的标量 Jordan 块。这是在选择另一种表示，并非分裂判据失效。Jordan 形存在后，[[Jordan形唯一性|块资料的唯一性]]由另一条定理负责。

## 来源与核验

<!-- bilingual-en:start -->
- [University of Colorado, Math 6140, §12.3, pp.1–2](https://math.colorado.edu/~mayr/teaching/math6140spring25/PID3.pdf): existence over a field where the characteristic polynomial splits, ordinary block structure and the distinction from diagonalization. Necessity is checked directly from the block characteristic polynomials and similarity.
- [Axler, *Linear Algebra Done Right*, 4th ed., 8.22 and 8.44–8.46, PDF pp.323,336–338](https://linear.axler.net/LADR4e.pdf#page=323): generalized-eigenspace decomposition, nilpotent chain construction and complex Jordan existence. The arbitrary-field statement is supported by the Colorado theorem, not inferred from a complex-only formulation.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=4|MIT 18.06SC Session 3.4 summary, p.4]] and [[03_Positive Definite Matrices and Applications#3.4.4 Jordan链与Jordan标准形|permanent course notes, §3.4.4]]: course block/chain notation. The rotation, complex change of basis and size-two boundary examples are verified directly.
<!-- bilingual-en:end -->

- 科罗拉多大学第12.3节第1–2页核对分裂域上的存在性、普通块结构及与对角化的区别；必要性由各块特征多项式和相似不变性直接核验。
- Axler 第8.22、8.44–8.46条，PDF323、336–338页，核对广义特征空间分解、幂零链构造与复数域存在性；任意数域的结论采用 Colorado 原定理，不由仅针对复数域的表述擅自推广。
- MIT Session3.4 摘要第4页与永久课程第3.4.4节核对块与链记号；旋转、复换基和二阶边界例子直接计算。
