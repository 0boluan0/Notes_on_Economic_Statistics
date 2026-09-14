---
aliases:
  - "SVD 的正奇异方向与左右零空间补基给出矩阵四个基本子空间的标准正交基"
  - Four fundamental subspaces from the SVD
student_os: knowledge-atom
atom_id: LA-SVD-022
atom_set: singular-value-decomposition-low-rank
atom_type: classification
status: source-checked
mastery_state: unassessed
requires:
  - "[[奇异向量]]"
  - "[[四个基本子空间]]"
  - "[[奇异值与秩]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[SVD零块补基不唯一]]"
related:
  - "[[紧SVD]]"
---

# SVD 的正奇异方向与左右零空间补基给出矩阵四个基本子空间的标准正交基
<!-- bilingual-en:start -->
*The positive singular directions and the left and right nullspace completions in an SVD provide orthonormal bases for the four fundamental subspaces*
<!-- bilingual-en:end -->

> [!summary] 四组方向分别属于哪里
> 设 $A\in\mathbb F^{m\times n}$ 的秩为 $r$，完整 SVD 为 $A=U\Sigma V^*$。把右奇异向量记为 $v_i$，左奇异向量记为 $u_i$，则
> $$
> \begin{aligned}
> \mathcal R(A^*)&=\operatorname{span}(v_1,\ldots,v_r),
> &\mathcal N(A)&=\operatorname{span}(v_{r+1},\ldots,v_n),\\
> \mathcal R(A)&=\operatorname{span}(u_1,\ldots,u_r),
> &\mathcal N(A^*)&=\operatorname{span}(u_{r+1},\ldots,u_m).
> \end{aligned}
> $$
> <!-- bilingual-en:start -->
> Positive right and left singular vectors span the row and column spaces; the remaining columns complete orthonormal bases of the right and left nullspaces.
> <!-- bilingual-en:end -->

对 $i\le r$，$Av_i=\sigma_i u_i$ 且 $\sigma_i>0$，所以 $A$ 把行空间中的标准正交基一一送到列空间中的标准正交基，只改变各方向的尺度。对 $i>r$，$Av_i=0$，所以其余右奇异向量正好张成零空间。把同一论证用于 $A^*$，便得到左零空间。

这四组基同时给出正交分解
$$
\mathbb F^n=\mathcal R(A^*)\oplus\mathcal N(A),
\qquad
\mathbb F^m=\mathcal R(A)\oplus\mathcal N(A^*).
$$
因此 SVD 不只是一个矩阵乘法公式，它还把输入空间和输出空间各自分成“会产生非零输出的方向”与“被压到零的方向”。

> [!question]- 自检
> 若 $A$ 是 $7\times4$、秩为 $2$，完整 SVD 中哪几列张成 $\mathcal N(A)$ 与 $\mathcal N(A^*)$？
>
> **答案：** $v_3,v_4$ 张成 $\mathcal N(A)$；$u_3,\ldots,u_7$ 张成 $\mathcal N(A^*)$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session 3.5 summary]]：核对 SVD 与四个基本子空间的对应。
- [[四个基本子空间]]：核对行空间、列空间、零空间与左零空间的维数及正交补关系。
