---
aliases:
  - Cramer's rule
  - Determinant formula for a linear-system component
student_os: knowledge-atom
atom_id: LA-DET-011
atom_set: determinants
atom_type: theorem
status: source-checked
requires:
  - "[[伴随矩阵求逆]]"
related:
  - "[[可逆性与非零行列式]]"
  - "[[行列式转置不变性]]"
part_of:
  - "[[行列式.canvas]]"
---

<!-- bilingual-en:start -->
# Cramer's rule
<!-- bilingual-en:end -->
*Cramer 法则*

<!-- bilingual-en:start -->
**Cramer's rule** finds one unknown in a square linear system by comparing two [[行列式|determinants]]. Keep the coefficient matrix, except that the column belonging to the chosen unknown is replaced by the right-hand side. Divide this new determinant by the original one. The rule applies when the original determinant is nonzero, so that the system has a unique solution.
<!-- bilingual-en:end -->

**Cramer 法则**通过两个[[行列式]]的比值，求方形线性方程组中的一个未知量：保留系数矩阵，只把所求未知量对应的那一列换成右端向量，再用新行列式除以原行列式。它要求原行列式非零，此时方程组有唯一解。

<!-- bilingual-en:start -->
## Read the replacement in a two-variable system
<!-- bilingual-en:end -->
## 二阶例子

<!-- bilingual-en:start -->
Consider the real equations $2x_1+x_2=5$ and $x_1+3x_2=7$. Collect their coefficients into $A$, their right sides into $b$, and the unknowns into the column $x$:
<!-- bilingual-en:end -->

考虑实数方程 $2x_1+x_2=5$ 与 $x_1+3x_2=7$。把系数放入 $A$，右端放入 $b$，未知量排成列向量 $x$：

$$
A=\begin{pmatrix}2&1\\1&3\end{pmatrix},\qquad
b=\begin{pmatrix}5\\7\end{pmatrix},\qquad
x=\begin{pmatrix}x_1\\x_2\end{pmatrix},\qquad Ax=b.
$$

<!-- bilingual-en:start -->
Column $1$ contains the coefficients of $x_1$, and column $2$ those of $x_2$. The original determinant is $\det A=2\cdot3-1\cdot1=5$. Replace the appropriate column, leaving the other column in place:
<!-- bilingual-en:end -->

第 $1$ 列装着 $x_1$ 的系数，第 $2$ 列装着 $x_2$ 的系数。原行列式为 $\det A=2\cdot3-1\cdot1=5$。只替换相应的一列，另一列保持原位置：

$$
A_1(b)=\begin{pmatrix}5&1\\7&3\end{pmatrix},\qquad
\det A_1(b)=15-7=8,
$$
$$
A_2(b)=\begin{pmatrix}2&5\\1&7\end{pmatrix},\qquad
\det A_2(b)=14-5=9.
$$

<!-- bilingual-en:start -->
The resulting values are $x_1=8/5$ and $x_2=9/5$. Substitution checks both original equations: $2(8/5)+9/5=5$ and $8/5+3(9/5)=7$.
<!-- bilingual-en:end -->

因此 $x_1=8/5$、$x_2=9/5$。代回原方程可核验：$2(8/5)+9/5=5$，$8/5+3(9/5)=7$。

<!-- bilingual-en:start -->
## The general formula
<!-- bilingual-en:end -->
*一般公式*

> [!important] Replace column $j$ to recover component $j$ · 换第 $j$ 列，求第 $j$ 个分量
> <!-- bilingual-en:start -->
> Let $A=[a_1\ \cdots\ a_n]\in\mathbb F^{n\times n}$ and $b\in\mathbb F^n$, with integer $n\ge1$ and a field $\mathbb F$ such as $\mathbb R$ or $\mathbb C$. Here $a_j$ is column $j$, not a single entry. If $\det A\ne0$, then $Ax=b$ has the unique solution
> <!-- bilingual-en:end -->
>
> 设 $A=[a_1\ \cdots\ a_n]\in\mathbb F^{n\times n}$、$b\in\mathbb F^n$，其中整数 $n\ge1$，$\mathbb F$ 是一个域，例如 $\mathbb R$ 或 $\mathbb C$。这里 $a_j$ 是完整的第 $j$ 列，不是单个元素。若 $\det A\ne0$，则 $Ax=b$ 有唯一解，且
>
> $$x_j=\frac{\det A_j(b)}{\det A},\qquad
> A_j(b)=[a_1\ \cdots\ a_{j-1}\ b\ a_{j+1}\ \cdots\ a_n],
> \quad j=1,\ldots,n.$$
>
> <!-- bilingual-en:start -->
> The denominator is the same for every component. At $n=1$, the equation $ax=b$ simply gives $x=b/a$ for $a\ne0$.
> <!-- bilingual-en:end -->
>
> 每个分量的分母相同。$n=1$ 时，方程 $ax=b$ 就给出 $a\ne0$ 时的 $x=b/a$。

<!-- bilingual-en:start -->
## Why the inverse formula produces a replaced column
<!-- bilingual-en:end -->
*为什么逆矩阵公式会产生换列行列式*

<!-- bilingual-en:start -->
The [[伴随矩阵求逆|adjugate inverse formula]] first guarantees the unique solution $x=A^{-1}b=(\det A)^{-1}\operatorname{adj}(A)b$. Let $C_{ij}$ denote the [[余子式与代数余子式|cofactor]] of the entry in row $i$, column $j$ of $A$. The [[伴随矩阵|adjugate]] transposes the cofactor matrix, so its entry in row $j$, column $i$ is $C_{ij}$. The row–column rule for [[矩阵乘法|matrix multiplication]] therefore gives
<!-- bilingual-en:end -->

[[伴随矩阵求逆|伴随求逆公式]]先保证唯一解 $x=A^{-1}b=(\det A)^{-1}\operatorname{adj}(A)b$。用 $C_{ij}$ 表示 $A$ 的第 $i$ 行、第 $j$ 列元素的[[余子式与代数余子式|代数余子式]]。[[伴随矩阵]]把代数余子式矩阵转置，因此其第 $j$ 行、第 $i$ 列元素是 $C_{ij}$。由[[矩阵乘法]]的行列相乘规则，

$$x_j=\frac{1}{\det A}\sum_{i=1}^n C_{ij}b_i.$$

<!-- bilingual-en:start -->
Here $b_i$ is entry $i$ of the right-hand side. Now expand $\det A_j(b)$ along its replaced column $j$ using [[余子式展开|cofactor expansion]]. Each cofactor in that column is still $C_{ij}$ from $A$: computing it deletes column $j$, so every changed entry disappears, and the smaller matrix is unchanged. Its checkerboard multiplier also keeps the same indices. Consequently,
<!-- bilingual-en:end -->

$b_i$ 是右端向量的第 $i$ 个元素。现在沿 $A_j(b)$ 被替换的第 $j$ 列做[[余子式展开]]。这一列的各代数余子式仍是原来的 $C_{ij}$：计算它时会删去第 $j$ 列，所有被替换的元素随之消失，删剩的小矩阵完全不变；棋盘乘数的指标也没变。因此，

$$\det A_j(b)=\sum_{i=1}^n b_iC_{ij}.$$

<!-- bilingual-en:start -->
Scalar multiplication commutes in a field, so this is exactly the numerator in $x_j$. The transposition in the adjugate is what matches row $j$ of the inverse with the cofactors of column $j$ of the original matrix.
<!-- bilingual-en:end -->

域中标量乘法可交换，所以这恰好是 $x_j$ 的分子。伴随矩阵中的转置，使逆矩阵的第 $j$ 行恰好与原矩阵第 $j$ 列的代数余子式相匹配。

<!-- bilingual-en:start -->
## A second proof from column combinations
<!-- bilingual-en:end -->
*从列组合再证明一次*

<!-- bilingual-en:start -->
The [[Ax的列组合解释|column-combination interpretation]] of $Ax=b$ says $b=\sum_{k=1}^n x_ka_k$. A determinant is linear in one column while the others are fixed; this follows from its row definition and [[行列式转置不变性|transpose invariance]]. Insert that combination into column $j$:
<!-- bilingual-en:end -->

$Ax=b$ 的[[Ax的列组合解释|列组合解释]]给出 $b=\sum_{k=1}^n x_ka_k$。行列式在其他列固定时对单列线性，这由逐行定义和[[行列式转置不变性]]得到。把这个组合放入第 $j$ 列：

$$
\det A_j(b)=\sum_{k=1}^n x_k
\det[a_1\ \cdots\ a_{j-1}\ a_k\ a_{j+1}\ \cdots\ a_n].
$$

<!-- bilingual-en:start -->
For $k\ne j$, column $k$ is still present and another copy has been placed in position $j$. The determinant has two equal columns and is zero. Only $k=j$ remains, restoring the original matrix. Thus $\det A_j(b)=x_j\det A$, and division by the nonzero determinant proves the formula. This proof uses the alternating property directly and works over any field, including characteristic two.
<!-- bilingual-en:end -->

当 $k\ne j$ 时，第 $k$ 列仍在原位置，第 $j$ 列又放入它的一个副本，形成两列相同，行列式为零。只剩 $k=j$，此时恢复原矩阵。因此 $\det A_j(b)=x_j\det A$；再除以非零行列式就得到公式。证明直接使用交替性，适用于任意域，包括特征二。

<!-- bilingual-en:start -->
## Singular systems and computational use
<!-- bilingual-en:end -->
## 边界与用途

<!-- bilingual-en:start -->
The last identity has a useful logical boundary: **if a solution exists**, it holds even when $\det A=0$, because no division was used to derive it. Hence, for a singular square matrix, one nonzero replacement determinant proves that no solution exists. The converse fails: all replacement determinants being zero does not establish consistency.
<!-- bilingual-en:end -->

最后那个恒等式有一条有用的逻辑边界：**若解已经存在**，即使 $\det A=0$，它仍然成立，因为推导未做除法。因此，对奇异方阵，只要有一个换列行列式非零，就能断定无解；反过来却不成立，所有换列行列式都为零不能保证有解。

> [!warning] Zero numerators do not settle consistency · 分子全零不能判断是否有解
> <!-- bilingual-en:start -->
> Let $A$ be the $2\times2$ zero matrix and $b=(1,0)^T$, where $T$ denotes [[矩阵转置|transpose]]. Each replaced matrix retains a zero column, so $\det A_1(b)=\det A_2(b)=0$. But $Ax=0$ for every $x$, and it can never equal this nonzero $b$. With the same $A$ and $b=0$, every $x$ is a solution instead. Over $\mathbb R$ or $\mathbb C$ this gives infinitely many solutions; over a finite field there are finitely many.
> <!-- bilingual-en:end -->
>
> 令 $A$ 为 $2\times2$ 零矩阵，$b=(1,0)^T$，其中 $T$ 表示[[矩阵转置|转置]]。每张换列矩阵仍保留一列零，所以 $\det A_1(b)=\det A_2(b)=0$；但任何 $x$ 都满足 $Ax=0$，不可能等于这个非零 $b$。若同一个 $A$ 配上 $b=0$，则每个 $x$ 都是解。在 $\mathbb R$ 或 $\mathbb C$ 上，这给出无穷多解；有限域上则只有有限多个。

<!-- bilingual-en:start -->
Thus an apparent $0/0$ is not a solution or a consistency test. Use [[高斯消元|elimination]] to decide a singular system. Cramer's ratio applies to square matrices with nonzero determinant; it is not a formula for a general rectangular system.
<!-- bilingual-en:end -->

所以，形式上的 $0/0$ 既不是一个解，也不是相容性检验；奇异系统应使用[[高斯消元|消元]]判断。Cramer 比值公式适用于行列式非零的方阵，并非一般矩形系统的求解公式。

<!-- bilingual-en:start -->
The rule is useful for proofs, small symbolic systems, and studying how one component depends on parameters while $\det A$ stays nonzero. Computing many replacement determinants for a large dense numerical system repeats work; [[消元与LU分解|LU]] or [[QR分解|QR]] can solve the system directly. For example, LAPACK's `DGESV` solves real square systems using a pivoted LU factorization. Neither Cramer's rule nor that task requires first writing out every entry of the inverse.
<!-- bilingual-en:end -->

这条规则适合证明、小型符号系统，以及在 $\det A$ 保持非零时研究某个分量如何依赖参数。对大型稠密数值系统，反复计算换列行列式会重复工作；[[消元与LU分解|LU]]或[[QR分解|QR]]可以直接求解。例如 LAPACK 的 `DGESV` 用带换行的 LU 分解解实方阵系统。Cramer 法则和这个求解任务都不要求先写出逆矩阵的全部元素。

> [!question]- Why does only one term survive? · 为什么只剩一项？
> <!-- bilingual-en:start -->
> After replacing column $j$ by $b=\sum_kx_ka_k$, every term with $k\ne j$ contains two copies of column $k$. Alternation makes those terms zero; the $k=j$ term is $x_j\det A$.
> <!-- bilingual-en:end -->
>
> 把第 $j$ 列换为 $b=\sum_kx_ka_k$ 后，每个 $k\ne j$ 的项都含两份第 $k$ 列，交替性使它们为零；$k=j$ 的项是 $x_j\det A$。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.7sum.pdf#page=2|MIT 18.06SC Session 2.7, pp.1–2]] and [[02_Least Squares, Determinants and Eigenvalues#Cramer 法则|the permanent course derivation]]: inverse/cofactor connection, replaced-column determinants, and the nonzero denominator. Both proofs above state the indices and hypotheses explicitly.
- [Mathlib, Matrix.Adjugate](https://leanprover-community.github.io/mathlib4_docs/Mathlib/LinearAlgebra/Matrix/Adjugate.html), `cramer_apply`, `cramer_eq_adjugate_mulVec`, and `mulVec_cramer`: the vector of replacement determinants and its adjugate identity without division. The ratio formula separately requires an invertible determinant.
- [LAPACK, DGESV source](https://netlib.org/lapack/double/dgesv.f), Purpose and `DGETRF`/`DGETRS` calls: solving by factorization. The two-variable example, order-one case, and singular zero-matrix counterexample are checked directly.
<!-- bilingual-en:end -->

- MIT Session 2.7，第1–2页及永久课程推导：核对逆矩阵与代数余子式的联系、换列行列式及非零分母；正文两条证明明确指标与假设。
- Mathlib 所列正式声明：核对换列行列式组成的向量及其不含除法的伴随恒等式；比值公式另需行列式可逆。
- LAPACK `DGESV` 原程序：核对利用分解求解的算法路线；二元例子、一阶情形和奇异零矩阵反例另行直接核算。
