---
aliases:
  - "方阵在底层域上存在普通 Jordan 标准形当且仅当其特征多项式在该域上分裂"
  - Jordan canonical form field condition
  - Jordan 标准形的域条件
student_os: knowledge-atom
atom_id: LA-EIG-014
atom_set: eigenvalues-linear-dynamics
atom_type: boundary
status: source-checked
mastery_state: unassessed
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

# 方阵在底层域上存在普通 Jordan 标准形当且仅当其特征多项式在该域上分裂
<!-- bilingual-en:start -->
*A square matrix has an ordinary Jordan form over the underlying field if and only if its characteristic polynomial splits over that field*
<!-- bilingual-en:end -->

> [!summary] 核心边界
> 对 $A\in\mathbb F^{n\times n}$，存在可逆 $S\in\mathbb F^{n\times n}$ 使 $S^{-1}AS$ 是 Jordan 块的直和，当且仅当特征多项式 $p_A(t)$ 在 $\mathbb F$ 上分解为一次因子的乘积。
> 在 $\mathbb C$ 上每个方阵都满足这个分裂条件；在 $\mathbb R$ 上则不一定。
> <!-- bilingual-en:start -->
> Jordan form over $\mathbb F$ exists when the characteristic polynomial splits into linear factors over $\mathbb F$.
> <!-- bilingual-en:end -->

必要性很直接：若 $A$ 已经相似于 Jordan 块的直和，那么相似保持特征多项式，而每个 Jordan 块的特征多项式都是 $(t-\lambda)^r$，所以 $p_A(t)$ 必在当前域上分裂。反过来，特征多项式一旦在当前域上分裂，广义特征空间分解和 Jordan 链便能给出所需的 Jordan 基。这两条方向合在一起，才是精确的域条件。

实数域的最小反例是平面旋转
$$R=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.$$
它的特征多项式是 $t^2+1$，在 $\mathbb R$ 上没有一次因子，因此不存在对角元为实特征值的普通实 Jordan 标准形。把标量域扩张到 $\mathbb C$ 后，特征值变成 $i,-i$，并可得到 $\operatorname{diag}(i,-i)$。

若坚持留在实数域，可以使用承载共轭复根的实 $2\times2$ 对角块；例如实 Schur 形或相应的实块标准形。它们不能与“全部由普通 Jordan 块组成”的 Jordan 标准形混为一谈。问题不在矩阵本身，而在允许使用的标量域。
<!-- bilingual-en:start -->
A real matrix may require real $2\times2$ blocks for a conjugate pair even though it has an ordinary Jordan form after extending scalars to $\mathbb C$.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> “每个实方阵都有只含实特征值的实 Jordan 标准形”错在哪里？
>
> **答案：** 实特征多项式可能含不可约二次因子；普通 Jordan 块要求其特征值属于当前底层域。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|MIT 18.06SC Session 3.4 summary]]：核对 Jordan 块直和的课程范围。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf|MIT Lecture 28 transcript]]：核对 Jordan 形式与特征值所在域。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#Session 3.4 Similar matrices and Jordan form|课程 Session 3.4]]：核对底层域与实 $2\times2$ 块边界。
