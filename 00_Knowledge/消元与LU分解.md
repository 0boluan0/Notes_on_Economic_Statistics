---
aliases:
  - 对方阵不换行的消元给出 A=LU 而换行时由 P 记录置换
  - Elimination and LU factorization
  - LU decomposition with pivoting
student_os: knowledge-atom
atom_id: LA-SYS-015
atom_set: linear-systems-four-subspaces
atom_type: factorization
status: source-checked
mastery_state: unassessed
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

# 对方阵不换行的消元给出 A=LU 而换行时由 P 记录置换
<!-- bilingual-en:start -->
*For a square matrix, elimination without row exchanges gives $A=LU$; row exchanges are recorded by a permutation $P$ in $PA=LU$*
<!-- bilingual-en:end -->

> [!summary] 分解关系
> 对方阵 $A\in\mathbb F^{n\times n}$，若标准高斯消元可以不换行地完成，并且
> $$
> E_k\cdots E_1A=U,
> $$
> 则消元矩阵的逆按相反次序组成下三角矩阵
> $$
> L=E_1^{-1}\cdots E_k^{-1},
> $$
> 从而 $A=LU$。若消元包含行交换，置换应显式记录，常用形式为
> $$
> PA=LU.
> $$
> <!-- bilingual-en:start -->
> For a square matrix on which standard Gaussian elimination can be completed without row exchanges, elimination matrices produce an upper-triangular $U$, and their inverses assemble the lower-triangular $L$ in $A=LU$. With row exchanges, the permutation is recorded as $PA=LU$.
> <!-- bilingual-en:end -->

在消元中不另行缩放主元行时，$L$ 的对角元为 $1$，其下三角条目保存消元乘子；$U$ 保存消元后的主元结构。若 $A$ 非奇异，则 $U$ 的对角主元全非零；分解后，针对多个右端 $b$，可重复用前代与回代解
$$
Ly=Pb,\qquad Ux=y,
$$
而不必重新分解 $A$。若 $A$ 奇异，某些主元为零，$A=LU$ 仍可能存在，但 $Ux=y$ 必须另做相容性与自由变量检查，不能宣称得到唯一解。

## 边界

$A=LU$ 并非对任意给定行序都自动成立；零主元可能迫使换行。实际数值算法通常还会主动选较稳的主元，因此应保留 $P$，不能把一次无换行的课堂例子推广成普遍规则。带换行时，原始消元矩阵的逆不能直接照搬为一个下三角 $L$；标准算法还要把已经保存的乘子随换行同步重排，才能得到 $PA=LU$。不同库也可能把置换写在等式另一侧，使用时先核约定。上述三角形式针对方阵；矩形矩阵的 LU 变体使用梯形因子，必须另行说明尺寸与约定。
<!-- bilingual-en:start -->
An unpivoted $A=LU$ factorization is not automatic for every row order. Numerical algorithms commonly pivot for existence or stability, so the permutation and the library's convention must be checked. If $A$ is singular, an LU factorization may still exist, but the triangular systems require consistency and free-variable checks rather than yielding a unique solution. Rectangular LU variants use trapezoidal factors and require their dimensions and convention to be stated explicitly.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么分解中保存的是消元乘子，而不是完整保存每个初等矩阵？
>
> **答案：** 在不换行消元中，这些乘子按位置填入单位下三角矩阵 $L$，与 $U$ 一起已能重现三角求解过程；若发生换行，还必须同时记录 $P$，并重排 $L$ 中此前已经保存的乘子。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.4sum.pdf|MIT 18.06SC Session 1.4 summary]]：核对消元矩阵逆、$L$ 中的乘子、$A=LU$ 与行交换。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.5sum.pdf|MIT 18.06SC Session 1.5 summary]]：核对置换矩阵与 $PA=LU$ 的约定。
