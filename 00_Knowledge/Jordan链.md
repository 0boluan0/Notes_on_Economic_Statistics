---
aliases:
  - "Jordan 链是一列在 A−λI 作用下逐级送向同一特征向量的非零向量"
  - Jordan chain
  - Jordan 链
student_os: knowledge-atom
atom_id: LA-EIG-012
atom_set: eigenvalues-linear-dynamics
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[广义特征向量]]"
related:
  - "[[代数重数与几何重数]]"
  - "[[Jordan块]]"
  - "[[Jordan标准形]]"
leads_to:
  - "[[Jordan链向量线性无关]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

# Jordan 链是一列在 A−λI 作用下逐级送向同一特征向量的非零向量
<!-- bilingual-en:start -->
*A Jordan chain is a sequence of nonzero vectors mapped step by step toward one eigenvector by $A-\lambda I$*
<!-- bilingual-en:end -->

> [!summary] 核心定义
> 对特征值 $\lambda$，若非零向量 $v_1,\ldots,v_r$ 满足
> $$(A-\lambda I)v_1=0,
> \qquad
> (A-\lambda I)v_{j+1}=v_j\quad(1\le j<r),$$
> 就称它们组成一条属于 $\lambda$、长度为 $r$ 的 Jordan 链。
> <!-- bilingual-en:start -->
> The first vector is an eigenvector; each later vector maps to the preceding one under $A-\lambda I$.
> <!-- bilingual-en:end -->

$v_1$ 是普通特征向量；对 $j\ge2$，$v_j$ 是阶为 $j$ 的真广义特征向量，因为
$$
(A-\lambda I)^jv_j=0,
\qquad
(A-\lambda I)^{j-1}v_j=v_1\ne0.
$$
同一条链中的向量由[[Jordan链向量线性无关|链递推关系保证线性无关]]。因此在这组链向量形成的链基中，
$$Av_1=\lambda v_1,
\qquad
Av_{j+1}=\lambda v_{j+1}+v_j,$$
所以 $A$ 在这条链上的矩阵表示正是一个[[Jordan块]]。

例如对
$$J=\begin{bmatrix}2&1\\0&2\end{bmatrix},$$
标准基 $e_1,e_2$ 满足 $(J-2I)e_1=0$、$(J-2I)e_2=e_1$，因此形成长度为二的 Jordan 链。矩阵只有一个普通特征方向，但这条链完整描述了二维广义特征空间。

一条链本身不保证补齐整个空间。只有在[[Jordan标准形的域条件|特征多项式于底层域上分裂]]时，才能选择足够多条链，把它们合在一起组成全空间的一组 Jordan 基。
<!-- bilingual-en:start -->
A single chain describes one block. A full Jordan basis is assembled from enough chains under the appropriate splitting condition.
<!-- bilingual-en:end -->

> [!question]- 最小自检
> 若 $(A-\lambda I)v_3=v_2$、$(A-\lambda I)v_2=v_1$ 且 $(A-\lambda I)v_1=0$，$v_3$ 的阶是多少？
>
> **答案：** 阶为 $3$，因为三次作用后归零，而两次作用后仍得到非零的 $v_1$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf|MIT 18.06SC Session 3.4 summary]]：核对缺失特征方向与 Jordan 链。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S04_Lecture_Lecture_28_Similar_Matrices_and_Jordan_Form.pdf|MIT Lecture 28 transcript]]：核对一条链对应一个 Jordan 块。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.4.4 Jordan链与Jordan标准形|课程 3.4.4]]：核对链方程与链基作用。
