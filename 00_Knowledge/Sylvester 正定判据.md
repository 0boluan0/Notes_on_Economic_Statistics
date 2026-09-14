---
aliases:
  - "实对称或复 Hermitian 矩阵正定当且仅当全部顺序主子式严格为正"
  - "Sylvester 判据用全部顺序主子式严格为正判断正定"
  - Sylvester's criterion
  - Leading principal minor test for positive definiteness
student_os: knowledge-atom
atom_id: LA-SPD-014
atom_set: symmetric-positive-definite
atom_type: equivalence
status: source-checked
mastery_state: unassessed
requires:
  - "[[LDL正定判别]]"
  - "[[正定矩阵]]"
  - "[[Hermitian 矩阵]]"
related:
  - "[[半正定主子式判别]]"
  - "[[行列式.canvas]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

# 实对称或复 Hermitian 矩阵正定当且仅当全部顺序主子式严格为正
<!-- bilingual-en:start -->
*Sylvester's criterion tests positive definiteness by strict positivity of every leading principal minor*
<!-- bilingual-en:end -->

> [!summary] Sylvester 判据
> 对实对称 $A\in\mathbb R^{n\times n}$，令
> $$\Delta_k=\det A_{1:k,1:k},\qquad k=1,\ldots,n.$$
> 则
> $$A\succ0\iff \Delta_1>0,\ldots,\Delta_n>0.$$
> 复 Hermitian 矩阵也有相同判据。
> <!-- bilingual-en:start -->
> A real symmetric matrix is positive definite if and only if all leading principal minors $\Delta_k=\det A_{1:k,1:k}$ are strictly positive. The same criterion holds for Hermitian matrices.
> <!-- bilingual-en:end -->

在 $A=LDL^T$ 中，单位下三角 $L$ 的前 $k$ 阶块行列式为 $1$，所以
$$
\Delta_k=d_1\cdots d_k,
\qquad d_k=\frac{\Delta_k}{\Delta_{k-1}},\quad \Delta_0=1.
$$
于是全部 $\Delta_k>0$ 与全部对称消元主元 $d_k>0$ 等价。
<!-- bilingual-en:start -->
For $A=LDL^T$, the leading block of the unit lower-triangular $L$ has determinant one, so $\Delta_k=d_1\cdots d_k$ and $d_k=\Delta_k/\Delta_{k-1}$. Strict positivity of all leading minors is therefore equivalent to positivity of all symmetric-elimination pivots.
<!-- bilingual-en:end -->

复 Hermitian 情形把 $LDL^T$ 换成 $LDL^*$，其中 $D$ 的对角元为实数；同一个主元乘积论证仍然成立。
<!-- bilingual-en:start -->
For a complex Hermitian matrix, replace $LDL^T$ by $LDL^*$; the diagonal pivots in $D$ are real, and the same leading-minor product argument applies.
<!-- bilingual-en:end -->

必须同时保留“实对称/Hermitian”“全部 $k=1,\ldots,n$”和“严格大于零”。只看最终的行列式不够；只看对角元也只是检查了一部分 $1\times1$ 主子式。把 $>0$ 机械改为 $\ge0$ 也不会得到半正定的充分判据。
<!-- bilingual-en:start -->
The symmetric or Hermitian assumption, every order $k$, and strict positivity are all essential. The final determinant alone is insufficient, as are the diagonal entries. Replacing $>0$ by $\ge0$ does not produce a sufficient semidefinite test.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对 $\begin{bmatrix}a&b\\b&c\end{bmatrix}$，Sylvester 判据是什么？
>
> **答案：** $a>0$ 且 $ac-b^2>0$。
>
> <!-- bilingual-en:start -->
> **Question:** What does Sylvester's criterion require for $\begin{bmatrix}a&b\\b&c\end{bmatrix}$?
>
> **Answer:** It requires $a>0$ and $ac-b^2>0$.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf|MIT 18.06SC Session 3.3 summary]]：核对二阶、三阶顺序主子式判据。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#主元判据为何成立|课程主元—主子式推导]]：核对 $\Delta_k=d_1\cdots d_k$ 与完整前提。
- [Brown University, *Linear Algebra Done Wrong*, §7.4](https://www.math.brown.edu/streil/papers/LADW/HTML_2026_04-30/Ch7.html)：核对 Hermitian 版 Sylvester 判据与非严格号不能平移到 PSD 的边界。
<!-- bilingual-en:start -->
- The MIT Session 3.3 summary was checked for the leading-minor tests in two and three dimensions.
- The course derivation was checked for $\Delta_k=d_1\cdots d_k$ and all assumptions.
- Brown University's linear-algebra text was checked for the Hermitian version of Sylvester's criterion and the failure of a nonstrict leading-minor analogue for PSD matrices.
<!-- bilingual-en:end -->
