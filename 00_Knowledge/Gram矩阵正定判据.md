---
aliases:
  - Gram matrix positive-definiteness criterion
  - Full column rank and positive-definite Gram matrix
student_os: knowledge-atom
atom_id: LA-PROJ-019
atom_set: orthogonal-projection-least-squares
atom_type: equivalence
status: source-checked
requires:
  - "[[正定矩阵]]"
  - "[[矩阵秩]]"
leads_to:
  - "[[满列秩投影公式]]"
related:
  - "[[正定与半正定可逆性]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---

<!-- bilingual-en:start -->
# Positive definiteness of a Gram matrix
<!-- bilingual-en:end -->

*Gram 矩阵的正定判据*

<!-- bilingual-en:start -->
A matrix can combine several columns into one output. The question is whether some nonzero choice of coefficients cancels that output completely. A **Gram matrix** records the pairwise dot products of the columns, and its quadratic form measures the squared length of their combination. It is positive definite exactly when no nonzero coefficient list can cancel the columns.
<!-- bilingual-en:end -->

矩阵可以把几列按系数组合成一个输出。问题在于：有没有一组不全为零的系数，能把输出完全抵消？**Gram 矩阵**记录各列两两的点积，它的二次型测量列组合的长度平方。恰好在任何非零系数组都不能抵消这些列时，这张矩阵是正定的。

> [!example] Two columns, three output coordinates · 两列与三个输出坐标
> <!-- bilingual-en:start -->
> Take the two columns of the following real matrix. With coefficients $s,t$, [[矩阵乘法|matrix multiplication]] gives $Ax=s a_1+t a_2$:
> <!-- bilingual-en:end -->
>
> 取下列实矩阵的两列，系数为 $s,t$。[[矩阵乘法]]给出 $Ax=s a_1+t a_2$：
>
> $$
> A=\begin{bmatrix}1&0\\0&1\\1&1\end{bmatrix},\qquad
> x=\begin{bmatrix}s\\t\end{bmatrix},\qquad
> Ax=\begin{bmatrix}s\\t\\s+t\end{bmatrix},\qquad
> G=A^TA=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
> $$
>
> <!-- bilingual-en:start -->
> Each column has squared length $2$ and their [[向量点积|dot product]] is $1$, giving the entries of $G$. The [[欧氏范数|squared Euclidean length]] of the output is
> <!-- bilingual-en:end -->
>
> 两列各自的长度平方为 $2$，两者的[[向量点积|点积]]为 $1$，因此得到 $G$ 的各元素。输出的[[欧氏范数|欧氏长度平方]]为
>
> $$x^TGx=s^2+t^2+(s+t)^2.$$
>
> <!-- bilingual-en:start -->
> This is strictly positive whenever $(s,t)\ne(0,0)$. The first two squares already prevent a nonzero coefficient pair from disappearing.
> <!-- bilingual-en:end -->
>
> 只要 $(s,t)\ne(0,0)$，这个值就严格为正；前两个平方已经保证非零系数对无法消失。

<!-- bilingual-en:start -->
For a real $m\times n$ matrix $A=[a_1\ \cdots\ a_n]$, its column Gram matrix is $G=A^TA$, with entries $G_{jk}=a_j^Ta_k$. It is an $n\times n$ matrix because it compares the $n$ columns, even when each column has $m$ coordinates. Here $m,n$ are positive integers. [[半正定矩阵|Positive semidefinite]] means symmetric with $x^TGx\ge0$ for every $x$; [[正定矩阵|positive definite]] requires strict positivity for every nonzero $x$.
<!-- bilingual-en:end -->

对于实 $m\times n$ 矩阵 $A=[a_1\ \cdots\ a_n]$，列 Gram 矩阵为 $G=A^TA$，元素是 $G_{jk}=a_j^Ta_k$。它比较的是 $n$ 列，所以尺寸为 $n\times n$，尽管每列有 $m$ 个坐标。这里 $m,n$ 为正整数。[[半正定矩阵]]要求对称且对每个 $x$ 都有 $x^TGx\ge0$；[[正定矩阵]]进一步要求每个非零 $x$ 都使结果严格为正。

> [!important] The criterion · 判据
> <!-- bilingual-en:start -->
> For every real $A\in\mathbb R^{m\times n}$, $A^TA$ is positive semidefinite, and
> <!-- bilingual-en:end -->
>
> 对任意实矩阵 $A\in\mathbb R^{m\times n}$，$A^TA$ 都半正定，而且
>
> $$
> \boxed{A^TA\succ0
> \iff N(A)=\{0\}
> \iff \operatorname{rank}(A)=n
> \iff A^TA\text{ is invertible}.}
> $$
>
> <!-- bilingual-en:start -->
> The [[零空间|nullspace]] $N(A)=\{x:Ax=0\}$ contains the coefficient lists that cancel the output. [[矩阵秩|Full column rank]], $\operatorname{rank}(A)=n$, means the columns are [[线性无关|linearly independent]]: only all-zero coefficients can cancel them. These are equivalent ways to test the same condition.
> <!-- bilingual-en:end -->
>
> [[零空间]] $N(A)=\{x:Ax=0\}$ 收集使输出抵消为零的系数组。[[矩阵秩|满列秩]]即 $\operatorname{rank}(A)=n$，表示各列[[线性无关]]：只有全零系数才能抵消它们。这几种说法检验的是同一个条件。

<!-- bilingual-en:start -->
## Why the squared length proves both directions
<!-- bilingual-en:end -->
## 一行证明中的两个方向

<!-- bilingual-en:start -->
The [[乘积的转置|transpose rule]] makes $G$ symmetric. For every real $x$, multiplication gives
<!-- bilingual-en:end -->

[[乘积的转置|转置规则]]保证 $G$ 对称；对每个实向量 $x$，直接相乘得到

$$x^TA^TAx=(Ax)^T(Ax)=\|Ax\|_2^2\ge0.$$

<!-- bilingual-en:start -->
This proves semidefiniteness without a rank assumption. If $N(A)=\{0\}$, every nonzero $x$ has $Ax\ne0$, whose squared length is strictly positive. Conversely, a nonzero $x\in N(A)$ gives a zero quadratic value, ruling out positive definiteness.
<!-- bilingual-en:end -->

这不需要任何秩假设，就证明了半正定性。若 $N(A)=\{0\}$，每个非零 $x$ 都有 $Ax\ne0$，其长度平方严格为正。反过来，若存在非零 $x\in N(A)$，就会得到零二次型值，从而排除正定性。

<!-- bilingual-en:start -->
The same identity gives the stronger nullspace statement $N(A^TA)=N(A)$. One direction is direct: $Ax=0$ implies $A^TAx=0$. In the other direction, $A^TAx=0$ implies $\|Ax\|_2^2=0$, hence $Ax=0$. Both matrices act on coefficient vectors with $n$ entries, so the [[秩零度定理|rank–nullity theorem]] also gives $\operatorname{rank}(A^TA)=\operatorname{rank}(A)$. A square matrix is [[非零齐次解与不可逆|invertible exactly when its nullspace is zero]], which proves the last equivalence.
<!-- bilingual-en:end -->

同一恒等式还给出 $N(A^TA)=N(A)$。一个方向直接可见：$Ax=0$ 推出 $A^TAx=0$。反方向中，$A^TAx=0$ 推出 $\|Ax\|_2^2=0$，因此 $Ax=0$。两张矩阵都以 $n$ 个系数为输入，[[秩零度定理]]便进一步给出 $\operatorname{rank}(A^TA)=\operatorname{rank}(A)$。方阵[[非零齐次解与不可逆|可逆当且仅当零空间只有零向量]]，这证明了最后一个等价关系。

<!-- bilingual-en:start -->
## Dependent columns and complex entries
<!-- bilingual-en:end -->
## 最小反例与复数情形

> [!warning] A square Gram matrix can still be singular · Gram 矩阵为方阵仍可能奇异
> <!-- bilingual-en:start -->
> Let $A=[a\ 2a]$ for any nonzero real column $a$. Then $z=(2,-1)^T$ is nonzero but $Az=2a-2a=0$. Hence $z^TA^TAz=0$ and $A^TA$ is singular. If $a=(1,0)^T$, it is explicitly $\begin{bmatrix}1&2\\2&4\end{bmatrix}$, with two positive diagonal entries. Positive column lengths do not guarantee column independence.
> <!-- bilingual-en:end -->
>
> 对任意非零实列向量 $a$，令 $A=[a\ 2a]$。非零向量 $z=(2,-1)^T$ 满足 $Az=2a-2a=0$，所以 $z^TA^TAz=0$，$A^TA$ 奇异。若 $a=(1,0)^T$，具体得到 $\begin{bmatrix}1&2\\2&4\end{bmatrix}$，两个对角元都为正。各列长度为正并不保证列独立。

<!-- bilingual-en:start -->
The dimension condition $m\ge n$ is necessary for full column rank, but is not sufficient: the preceding example is square and still fails. With complex entries, use $A^*A$, where $*$ means [[矩阵转置|conjugate transpose]], and $\|y\|_2^2=\sum_j|y_j|^2$. It is [[Hermitian 矩阵|Hermitian]] and $x^*A^*Ax=\|Ax\|_2^2$. Thus the same nullspace, rank and positive-definiteness conclusions hold over $\mathbb C$. Ordinary transpose cannot replace $*$: for $A=[i]$, $A^TA=[-1]$ whereas $A^*A=[1]$.
<!-- bilingual-en:end -->

满列秩必须有 $m\ge n$，但仅此还不够：上例是方阵，仍不满足条件。复数情形使用 $A^*A$，其中 $*$ 表示[[矩阵转置|共轭转置]]，且 $\|y\|_2^2=\sum_j|y_j|^2$。它是 [[Hermitian 矩阵]]，并满足 $x^*A^*Ax=\|Ax\|_2^2$，所以零空间、秩及正定判据都同样成立。普通转置不能替代 $*$：若 $A=[i]$，则 $A^TA=[-1]$，而 $A^*A=[1]$。

<!-- bilingual-en:start -->
This criterion explains the rank assumption in the [[满列秩投影公式|full-column-rank projection formula]]. The other product $AA^*$ has size $m\times m$ and is positive definite exactly when $A$ has full **row** rank, by applying the same criterion to $A^*$. The sum-of-squares argument for finite sums $\sum_i v_iv_i^*$ is developed separately in [[外积和的半正定性|positive semidefiniteness of outer-product sums]].
<!-- bilingual-en:end -->

这个判据解释了[[满列秩投影公式]]为什么需要秩条件。另一个乘积 $AA^*$ 的尺寸是 $m\times m$；把判据用于 $A^*$，得到它正定当且仅当 $A$ 满**行**秩。有限和 $\sum_i v_iv_i^*$ 的平方和论证则由[[外积和的半正定性]]单独展开。

<!-- visual-lab:links:start -->
## 交互理解

- [[07_Programme/可视化实验室/实验/kernel|核矩阵如何产生非线性拟合]]
<!-- visual-lab:links:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.1sum.pdf#page=2|MIT 18.06SC, Orthogonal Vectors and Subspaces, p. 2]]: equality of nullspaces and ranks, and invertibility exactly under independent columns. The squared-length proof and explicit matrices above check the positive-definiteness formulation.
- [Axler, *Linear Algebra Done Right*, 4th ed., §7A, 7.1 and 7.5–7.9; §7C, 7.34 and 7.38](https://linear.axler.net/LADR4e.pdf): adjoint/conjugate-transpose rules and the nonnegative squared-norm proof for $T^*T$. Axler's “positive” operator means positive semidefinite; strict positivity here is established by the displayed nullspace argument.
<!-- bilingual-en:end -->

- MIT 第 2.1 节摘要第 2 页：核对零空间、秩相同及独立列对应可逆；正文由长度平方核验正定版本和具体例子。
- Axler 第 7A、7C 节上述定义与定理：核对共轭转置及平方范数论证。该书的 positive operator 指半正定算子；本卡的严格正定由零空间论证另行得到。
