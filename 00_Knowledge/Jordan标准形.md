---
aliases:
  - "Jordan 标准形是由 Jordan 块组成并与原矩阵相似的块对角矩阵"
  - Jordan canonical form
  - Jordan 标准形
  - Jordan 形
student_os: knowledge-atom
atom_id: LA-EIG-033
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[Jordan块]]"
  - "[[相似矩阵]]"
related:
  - "[[Jordan链]]"
  - "[[Jordan标准形的域条件]]"
  - "[[Jordan形唯一性]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# Jordan 标准形是由 Jordan 块组成并与原矩阵相似的块对角矩阵
<!-- bilingual-en:start -->
*A Jordan canonical form is a block-diagonal matrix of Jordan blocks similar to the original matrix*
<!-- bilingual-en:end -->

> [!summary] 核心定义
> 若存在可逆矩阵 $S$ 使
> $$S^{-1}AS=J=\operatorname{diag}\bigl(J_{r_1}(\lambda_1),\ldots,J_{r_q}(\lambda_q)\bigr),$$
> 其中每个 $J_{r_i}(\lambda_i)$ 都是 Jordan 块，就称 $J$ 是 $A$ 的 Jordan 标准形。
> <!-- bilingual-en:start -->
> Jordan form records a linear operator as independent chains, one block for each chain.
> <!-- bilingual-en:end -->

$S$ 的列可以按[[Jordan链]]分组。每条链生成一个块：链长决定块大小，所属特征值决定块的对角元。于是 Jordan 标准形不只列出特征值，还记录普通特征向量不足时，广义特征方向怎样逐层连接。

例如
$$J=\operatorname{diag}\left(
\begin{bmatrix}3&1\\0&3\end{bmatrix},[1]
\right)$$
有两个特征值 $3$ 与 $1$；特征值 $3$ 对应一条长度二的链，特征值 $1$ 对应一条长度一的链。它与 $\operatorname{diag}(3,3,1)$ 的特征值列表相同，但块结构不同；由[[Jordan形唯一性]]可知二者不相似。

Jordan 标准形的定义只说明这种块对角表示是什么。并非每个域上的方阵都能在原域中写成这种形式；存在性需要另查[[Jordan标准形的域条件]]，唯一性则由[[Jordan形唯一性]]承担。
<!-- bilingual-en:start -->
The definition, the field condition for existence, and uniqueness of the block data are separate claims.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 一个 Jordan 标准形有三个块，是否说明它一定有三个不同特征值？
>
> **答案：** 不一定。同一特征值可以对应多条 Jordan 链，因此可以出现多个具有相同对角元的块。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|MIT 18.06SC Session 3.4 summary]]：核对 Jordan 标准形的块对角结构。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf|MIT Lecture 28 transcript]]：核对相似变换与 Jordan 块表示。
