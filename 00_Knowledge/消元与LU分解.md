---
aliases:
  - Elimination and LU factorization
  - LU decomposition with pivoting
  - LU分解
student_os: knowledge-atom
atom_id: LA-SYS-015
atom_set: linear-systems-four-subspaces
atom_type: factorization
status: source-checked
requires:
  - "[[高斯消元]]"
  - "[[初等矩阵]]"
  - "[[行操作的解集不变性]]"
  - "[[主元与自由变量]]"
  - "[[置换矩阵]]"
related:
  - "[[消元计算行列式]]"
  - "[[薄QR最小二乘]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

<!-- bilingual-en:start -->
# Elimination and LU factorization
<!-- bilingual-en:end -->
*消元与 LU 分解*

<!-- bilingual-en:start -->
**LU factorization** saves the work of elimination in two matrices. The upper triangular factor $U$ contains the simplified equations; the lower triangular factor $L$ records how to rebuild the equations. If rows were exchanged, a permutation matrix $P$ also records their order. Once these factors are available, the same coefficient matrix can be used with new right-hand sides without repeating its elimination.
<!-- bilingual-en:end -->

**LU 分解**把消元工作保存在两个矩阵中：上三角因子 $U$ 保存简化后的方程，下三角因子 $L$ 记录怎样重建方程；若交换过行，还要由置换矩阵 $P$ 记录行序。因子得到之后，同一个系数矩阵换用新的右端时，不必重新对它消元。

<!-- bilingual-en:start -->
Here $A$ is an $n\times n$ real or complex matrix, with $n\ge1$. A [[三角矩阵|lower triangular matrix]] has zeros above its diagonal; an upper triangular one has zeros below it. We use **unit lower triangular** $L$, meaning every diagonal entry is $1$. This convention corresponds to eliminating below pivots without separately scaling the pivot rows.
<!-- bilingual-en:end -->

本卡的 $A$ 是 $n\times n$ 实或复矩阵，$n\ge1$。[[三角矩阵|下三角矩阵]]的对角线上方为零，上三角矩阵的对角线下方为零。这里采用**单位下三角**的 $L$，即其对角元都为 $1$；对应做法是消去主元下方的元素，不另行缩放主元行。

<!-- bilingual-en:start -->
## Keep the subtraction multiplier
<!-- bilingual-en:end -->
*保存行减法的乘子*

<!-- bilingual-en:start -->
Take the coefficient matrix from [[高斯消元|Gaussian elimination]]. Subtracting three times row 1 from row 2 gives
<!-- bilingual-en:end -->

取[[高斯消元]]中的系数矩阵，从第二行减去第一行的三倍，得到

$$A=\begin{pmatrix}1&2\\3&7\end{pmatrix},\qquad
E=\begin{pmatrix}1&0\\-3&1\end{pmatrix},\qquad
EA=U=\begin{pmatrix}1&2\\0&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
The [[初等矩阵|elementary matrix]] $E$ performs the subtraction. Its inverse adds three times row 1 back, so
<!-- bilingual-en:end -->

[[初等矩阵]] $E$ 执行减法，其逆把第一行的三倍加回，因此

$$L=E^{-1}=\begin{pmatrix}1&0\\3&1\end{pmatrix},\qquad
A=LU=\begin{pmatrix}1&0\\3&1\end{pmatrix}
\begin{pmatrix}1&2\\0&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
The second row of $LU$ is three copies of $(1,2)$ plus $(0,1)$, namely $(3,7)$. Thus $L$ stores the multiplier $3$, while $U$ stores the resulting equations. The minus sign belongs to the elimination step; the positive multiplier in $L$ reconstructs what was subtracted.
<!-- bilingual-en:end -->

$LU$ 的第二行是三倍 $(1,2)$ 加上 $(0,1)$，即 $(3,7)$。因此 $L$ 保存乘子 $3$，$U$ 保存消元结果。减号属于消元步骤；$L$ 中的正乘子用来重建被减去的部分。

<!-- bilingual-en:start -->
More generally, suppose ordinary column-by-column elimination finishes **without row exchanges** and without scaling pivot rows. If its elementary matrices act in the order $E_1,\ldots,E_s$, then
<!-- bilingual-en:end -->

一般地，假设按列进行的标准消元能够**不换行**地完成，并且不缩放主元行。若初等矩阵依次为 $E_1,\ldots,E_s$，则

$$E_s\cdots E_1A=U,\qquad
A=\underbrace{E_1^{-1}\cdots E_s^{-1}}_{L}U.$$

<!-- bilingual-en:start -->
The [[乘积的逆矩阵|inverse product reverses the actions]]. Every inverse elimination factor is unit lower triangular, and [[下三角乘积封闭性|their product retains this shape]]. In this ordered algorithm, the entry $\ell_{ik}$ of $L$ is the multiplier used to subtract row $k$ from row $i>k$. This is not a rule that one can invert any lower triangular matrix merely by changing signs.
<!-- bilingual-en:end -->

[[乘积的逆矩阵|逆乘积撤销操作的次序相反]]。每个逆消元因子都是单位下三角矩阵，[[下三角乘积封闭性|乘积也保持这种形状]]。在这套有序算法中，$L$ 的元素 $\ell_{ik}$ 就是从第 $i>k$ 行减去第 $k$ 行时使用的乘子。这不是说任意下三角矩阵求逆都只需改变符号。

<!-- bilingual-en:start -->
## A later row swap must move earlier multipliers
<!-- bilingual-en:end -->
*后续换行时，先前的乘子也要跟着换*

<!-- bilingual-en:start -->
A [[主元|pivot]] is the nonzero coefficient used to eliminate entries below it. If a proposed diagonal pivot is zero but a usable entry lies below it, exchange rows. Let the [[置换矩阵|permutation matrix]] $P$ record the accumulated row order. Our convention is
<!-- bilingual-en:end -->

[[主元]]是用来消除下方元素的非零系数。若候选对角主元为零而下方有可用非零元，就交换行。用[[置换矩阵]] $P$ 记录累计行序，本卡采用

> [!summary] Read the factors together · 一起读取三个因子
> $$PA=LU.$$
> <!-- bilingual-en:start -->
> $P$ first reorders the original equations. Multiplying $L$ and $U$ reconstructs that reordered coefficient matrix.
> <!-- bilingual-en:end -->
>
> $P$ 先重排原方程；$L$ 与 $U$ 相乘重建的是这个重排后的系数矩阵。

<!-- bilingual-en:start -->
For a concrete case, start with
<!-- bilingual-en:end -->

例如，从下式开始：

$$A=\begin{pmatrix}1&1&1\\2&2&3\\3&4&5\end{pmatrix}
\longrightarrow
\begin{pmatrix}1&1&1\\0&0&1\\0&1&2\end{pmatrix}.$$

<!-- bilingual-en:start -->
The first-column multipliers are $2$ and $3$. At the next stage, swap rows 2 and 3 to obtain $U$. The multiplier $3$ belongs to the row now in position 2, and $2$ belongs to the row now in position 3. Therefore
<!-- bilingual-en:end -->

第一列的消元乘子分别为 $2$、$3$。下一步交换第二、三行，得到 $U$。原乘子 $3$ 所属的行现在位于第二行，原乘子 $2$ 所属的行现在位于第三行，所以

$$P=\begin{pmatrix}1&0&0\\0&0&1\\0&1&0\end{pmatrix},\qquad
L=\begin{pmatrix}1&0&0\\3&1&0\\2&0&1\end{pmatrix},\qquad
U=\begin{pmatrix}1&1&1\\0&1&2\\0&0&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
Multiplication gives $LU$ with rows $(1,1,1),(3,4,5),(2,2,3)$, exactly $PA$. Leaving $2,3$ in their old positions would instead produce rows $(2,3,4)$ and $(3,3,4)$ below the first row, which is wrong. At elimination stage $k$, a swap among the remaining rows must also swap their already stored entries in columns $1,\ldots,k-1$ of $L$; the unit diagonal stays in place.
<!-- bilingual-en:end -->

相乘后 $LU$ 的各行为 $(1,1,1),(3,4,5),(2,2,3)$，正好是 $PA$。若把 $2,3$ 留在旧位置，后两行会错误地成为 $(2,3,4)$、$(3,3,4)$。因此第 $k$ 步交换尚未处理的行时，还要交换 $L$ 中这些行已经存入第 $1,\ldots,k-1$ 列的元素，单位对角线留在原位。

<!-- bilingual-en:start -->
Why only the stored columns? Before this swap, write the partial reconstruction as $P_{\rm old}A=L_{\rm old}B$, where $B$ is the current working matrix. The unprocessed lower-right block of $L_{\rm old}$ is an identity matrix. For a swap matrix $S$ acting only on remaining rows,
<!-- bilingual-en:end -->

为什么只交换已保存的列？在换行之前，把部分重建关系写成 $P_{\rm old}A=L_{\rm old}B$，其中 $B$ 是当前工作矩阵。$L_{\rm old}$ 尚未处理的右下块仍是单位矩阵。若交换矩阵 $S$ 只作用于剩余行，则

$$SP_{\rm old}A=(SL_{\rm old}S^T)(SB).$$

<!-- bilingual-en:start -->
Here $T$ denotes [[矩阵转置|transpose]], and $S^TS=I$, with $I$ the [[单位矩阵|identity matrix]]. Conjugating $L_{\rm old}$ this way swaps its stored row prefixes while leaving that identity block intact. It preserves the reconstruction equation; subsequent elimination continues from $SB$. This is why a product of raw inverse elimination and swap matrices cannot simply be called a lower triangular $L$ without arranging the permutation separately.
<!-- bilingual-en:end -->

$T$ 表示[[矩阵转置|转置]]，且 $S^TS=I$，$I$ 是[[单位矩阵]]。这样同时在左右作用于 $L_{\rm old}$，会交换已保存的行前缀，却保持右下单位块不变，也保持重建等式；后续从 $SB$ 继续消元。正因如此，不能把原始消元与换行操作的逆直接连乘，就不加整理地称为下三角 $L$；必须单独整理出置换。

<!-- bilingual-en:start -->
## Solve new right-hand sides with the saved factors
<!-- bilingual-en:end -->
*用已有因子求解新的右端*

<!-- bilingual-en:start -->
For $Ax=b$ and $PA=LU$, introduce an intermediate vector $y=Ux$. Solve $Ly=Pb$ from top to bottom (**forward substitution**), then $Ux=y$ from bottom to top (**back-substitution**). In the opening $2\times2$ example, $P=I$ and $b=(5,16)^T$: the first solve gives $y_1=5$, $3y_1+y_2=16$, hence $y=(5,1)^T$; the second gives $x_2=1$, $x_1+2x_2=5$, hence $x=(3,1)^T$.
<!-- bilingual-en:end -->

对 $Ax=b$ 与 $PA=LU$，引入中间向量 $y=Ux$。先从上往下解 $Ly=Pb$，称为**前代**；再从下往上解 $Ux=y$，称为**回代**。开头二维例子中 $P=I$、$b=(5,16)^T$：先得 $y_1=5$、$3y_1+y_2=16$，即 $y=(5,1)^T$；再得 $x_2=1$、$x_1+2x_2=5$，即 $x=(3,1)^T$。

<!-- bilingual-en:start -->
For fixed invertible $A$, its factors can be [[LU分解复用成本|reused for multiple right-hand sides]]. No explicit inverse is required. With row exchanges, the right-hand side must be reordered by the same $P$.
<!-- bilingual-en:end -->

固定可逆 $A$ 后，可以[[LU分解复用成本|对多个右端复用因子]]，不必显式求逆。有换行时，右端必须由同一个 $P$ 重排。

<!-- bilingual-en:start -->
## Check the factorization convention and usable pivots
<!-- bilingual-en:end -->
## 边界

<!-- bilingual-en:start -->
Unpivoted LU is not automatic in a given row order. For $A=\begin{pmatrix}0&1\\1&0\end{pmatrix}$, unit lower triangular $L$ and upper triangular $U$ would force $u_{11}=a_{11}=0$, then $(LU)_{21}=\ell_{21}u_{11}=0\ne a_{21}$. A row swap fixes the obstacle. Conversely, a singular matrix can have LU: $\operatorname{diag}(1,0)=I_2\operatorname{diag}(1,0)$. Here $\operatorname{diag}$ puts the listed entries on the diagonal and zeros elsewhere. Its zero final diagonal entry prevents ordinary division-based back-substitution; consistency and free variables must be checked instead of claiming a unique solution.
<!-- bilingual-en:end -->

给定行序下，无换行 LU 并非自动存在。对 $A=\begin{pmatrix}0&1\\1&0\end{pmatrix}$，若有单位下三角 $L$ 和上三角 $U$，便有 $u_{11}=a_{11}=0$，进而 $(LU)_{21}=\ell_{21}u_{11}=0\ne a_{21}$，产生矛盾；换行能解决这个障碍。反过来，奇异矩阵也可能有 LU：$\operatorname{diag}(1,0)=I_2\operatorname{diag}(1,0)$，其中 $\operatorname{diag}$ 把所列数放在对角线上、其余位置放零。此时最后对角元为零，不能照常除法回代，应检查相容性和自由变量，而非宣称唯一解。

<!-- bilingual-en:start -->
Numerical algorithms may also exchange rows to control multiplier sizes; exact factorization alone is not a [[条件性与算法稳定性|floating-point stability guarantee]]. Software may instead write $A=P_{\rm lib}LU$; then our $P=P_{\rm lib}^{-1}=P_{\rm lib}^T$. Check the stated convention rather than assuming every permutation is its own inverse. Rectangular LU variants use appropriately sized trapezoidal factors and need separate dimension conventions; the square formulas here do not silently specify those variants.
<!-- bilingual-en:end -->

数值算法还可能为了控制乘子大小而主动换行；精确分解本身不是[[条件性与算法稳定性|浮点稳定性保证]]。软件也可能写成 $A=P_{\rm lib}LU$，这时本卡的 $P=P_{\rm lib}^{-1}=P_{\rm lib}^T$。应核对所用约定，不能假设任何置换都等于自己的逆。矩形 LU 变体使用相应尺寸的梯形因子，需要单独说明尺寸约定，本卡方阵公式不暗含所有变体。

> [!question]- What must follow a late row exchange? · 后续换行要带走什么？
> <!-- bilingual-en:start -->
> The already stored multiplier prefixes follow the exchanged rows, while the unit diagonal remains fixed. The $3\times3$ example checks this by multiplying $L$ and $U$ back to $PA$.
> <!-- bilingual-en:end -->
>
> 已保存的乘子前缀随行交换，单位对角线保持原位；三阶例子通过把 $L,U$ 乘回 $PA$ 核验这一点。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf#page=1|MIT 18.06SC Session 1.4, pp.1–3]]: inverse elimination order, unit lower triangular multipliers and row exchanges.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.5sum.pdf#page=1|MIT Session 1.5, p.1]]: the $PA=LU$ convention and $P^{-1}=P^T$.
- [LAPACK DGETRF original source, Purpose and the update of columns 1:J−1](https://www.netlib.org/lapack/double/dgetrf.f): row-pivoted LU, stored multiplier exchanges, rectangular shape conventions and the alternative $A=PLU$ notation. The reconstruction identity and examples above are derived directly.
<!-- bilingual-en:end -->

- MIT Session 1.4，第1–3页：核对逆消元顺序、单位下三角乘子及换行。
- MIT Session 1.5，第1页：核对 $PA=LU$ 约定和 $P^{-1}=P^T$。
- LAPACK DGETRF 原实现的 Purpose 与第1至 J−1列交换段：核对选主元 LU、旧乘子的重排、矩形尺寸及另一种 $A=PLU$ 约定；重建恒等式与算例由正文直接推导。
