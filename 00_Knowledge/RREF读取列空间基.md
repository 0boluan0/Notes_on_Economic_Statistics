---
aliases:
  - RREF 的主元位置必须返回原矩阵对应列才能给出原矩阵列空间的一组基
  - Reading a column-space basis from RREF
student_os: knowledge-atom
atom_id: LA-SYS-025
atom_set: linear-systems-four-subspaces
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[行最简形]]"
  - "[[基]]"
  - "[[主元与自由变量]]"
  - "[[列空间]]"
  - "[[行操作与四个子空间]]"
related:
  - "[[RREF读取子空间基]]"
  - "[[四个基本子空间]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

# RREF 的主元位置必须返回原矩阵对应列才能给出原矩阵列空间的一组基
<!-- bilingual-en:start -->
*Pivot locations in RREF must be used to select the corresponding original columns for a basis of the original column space*
<!-- bilingual-en:end -->

> [!summary] 方法
> 设 $A=[a_1\ \cdots\ a_n]$，其 RREF 为 $R$。若 $R$ 的主元位于第 $j_1,\ldots,j_r$ 列，那么
> $$
> \{a_{j_1},\ldots,a_{j_r}\}
> $$
> 是 $C(A)$ 的一组基。$R$ 只负责告诉你**列号**；真正写进答案的是 $A$ 的原列。
> <!-- bilingual-en:start -->
> Use RREF to locate pivot-column indices, then take those indexed columns from the original matrix $A$.
> <!-- bilingual-en:end -->

非主元列在消元关系中由主元列线性表示；可逆行操作不会改变这些列之间的线性关系。因此原矩阵的主元列既线性无关，又能张成全部原列。

不能把 $R$ 的主元列抄作 $C(A)$ 的基。若
$$
A=\begin{bmatrix}1&0&1\\0&1&1\\1&1&2\end{bmatrix},
\qquad
R=\begin{bmatrix}1&0&1\\0&1&1\\0&0&0\end{bmatrix},
$$
主元都在前两列，但 $C(A)=\operatorname{span}\{(1,0,1)^T,(0,1,1)^T\}$，而 $R$ 的前两列张成的是另一个平面。主元位置相同，不等于列向量相同。

> [!question]- 自检
> RREF 的第 $2$、$5$ 列是主元列。求 $C(A)$ 的基时应写哪两列？
>
> **答案：** 写原矩阵 $A$ 的第 $2$、$5$ 列；RREF 只提供位置。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.9sum.pdf|MIT 18.06SC Session 1.9 summary]]：核对由 RREF 主元位置返回原矩阵主元列的规则。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf|MIT 18.06SC Session 1.10 summary]]：核对行操作保持列依赖关系但一般改变列空间的边界。
