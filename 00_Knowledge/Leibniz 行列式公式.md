---
aliases:
  - "Leibniz 公式的每一项从每行每列各取一个元素并带排列符号"
  - Leibniz determinant formula
  - Permutation formula for determinants
student_os: knowledge-atom
atom_id: LA-DET-008
atom_set: determinants
atom_type: formula
status: source-checked
mastery_state: unassessed
requires:
  - "[[行列式]]"
part_of:
  - "[[行列式.canvas]]"
---

# Leibniz 公式的每一项从每行每列各取一个元素并带排列符号
<!-- bilingual-en:start -->
*Each term in the Leibniz formula selects one entry from every row and every column and carries a permutation sign*
<!-- bilingual-en:end -->

> [!summary] 核心公式
> 对 $A=(a_{ij})\in\mathbb F^{n\times n}$，
> $$
> \det A=
> \sum_{\sigma\in S_n}\operatorname{sgn}(\sigma)
> \prod_{i=1}^n a_{i,\sigma(i)}.
> $$
> 每个排列 $\sigma$ 指定第 $i$ 行从第 $\sigma(i)$ 列取数，因此每一项恰好使用每行、每列各一次；偶排列取正号，奇排列取负号。
> <!-- bilingual-en:start -->
> For $A=(a_{ij})\in\mathbb F^{n\times n}$, the Leibniz formula sums $\operatorname{sgn}(\sigma)\prod_i a_{i,\sigma(i)}$ over all permutations. A permutation selects exactly one entry from every row and every column; even permutations contribute positively and odd permutations negatively.
> <!-- bilingual-en:end -->

逐行线性把 determinant 展开成标准基行的组合。若两行选到同一列，对应矩阵有重复标准基行，项为零；只有列指标构成排列时才留下。排列的符号正是把这些标准基行恢复到自然顺序所需交换次数的奇偶性。
<!-- bilingual-en:start -->
Separate row linearity expands the determinant into matrices built from standard-basis rows. Selecting the same column in two rows creates repeated basis rows and a zero term. Only column choices forming a permutation survive, with sign determined by the parity of the exchanges needed to restore natural order.
<!-- bilingual-en:end -->

## 何时用它

- 它揭示 determinant 为何与排列、符号和每行每列各取一次相连。
- 对置换矩阵，只有一个乘积非零，所以 determinant 只能是 $\pm1$。
- 一般有 $n!$ 项，不适合稠密大矩阵的数值计算；消元约需 $O(n^3)$ 次算术操作。
<!-- bilingual-en:start -->
- The formula exposes the connection among determinants, permutations, signs, and one selection per row and column.
- A permutation matrix has exactly one nonzero product, so its determinant is $\pm1$.
- The general formula contains $n!$ terms and is unsuitable for dense numerical computation; elimination uses roughly $O(n^3)$ arithmetic operations.
<!-- bilingual-en:end -->

## 易错边界

三阶矩阵常见的“斜线记忆法”只是 Leibniz 公式在 $3\times3$ 的特例，不能直接延伸到四阶。只检查“每行取一个”也不够；若重复列，该乘积不是合法项。
<!-- bilingual-en:start -->
The familiar diagonal mnemonic for a $3\times3$ determinant is only a special case and does not extend directly to order four. Selecting one entry per row is insufficient unless every column is also selected exactly once.
<!-- bilingual-en:end -->

> [!question]- 自检
> $4\times4$ 的 Leibniz 公式有多少项？每项含多少个矩阵元素？
>
> **答案：** 有 $4!=24$ 项；每项是四个元素的乘积，并且四行四列各出现一次。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf|MIT 18.06SC Session 2.6 summary]]：核对排列公式、符号和项数。
- [[01_Math/02_linear algebra/02_Least Squares, Determinants and Eigenvalues.md#Lecture：排列大公式|课程排列大公式]]：核对每行每列各取一次及计算边界。
<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.6sum.pdf|MIT 18.06SC Session 2.6 summary]] was checked for the permutation formula, signs, and number of terms.
- [[01_Math/02_linear algebra/02_Least Squares, Determinants and Eigenvalues.md#Lecture：排列大公式|The course Leibniz-formula section]] was checked for the one-per-row-and-column condition and computational boundary.
<!-- bilingual-en:end -->
