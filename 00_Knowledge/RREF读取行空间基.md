---
aliases:
  - RREF 的非零行构成原矩阵行空间的一组基
  - Reading a row-space basis from RREF
student_os: knowledge-atom
atom_id: LA-SYS-026
atom_set: linear-systems-four-subspaces
atom_type: method
status: source-checked
mastery_state: unassessed
requires:
  - "[[行最简形]]"
  - "[[基]]"
  - "[[行空间]]"
related:
  - "[[RREF读取子空间基]]"
  - "[[四个基本子空间]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

# RREF 的非零行构成原矩阵行空间的一组基
<!-- bilingual-en:start -->
*The nonzero rows of RREF form a basis of the original matrix's row space*
<!-- bilingual-en:end -->

> [!summary] 方法
> 对 $A\in\mathbb F^{m\times n}$，若 $R$ 是 $A$ 的 RREF，那么把 $R$ 的所有非零行按行向量取出，就得到 $A$ 的行空间的一组基。
> <!-- bilingual-en:start -->
> The nonzero rows of the RREF of $A$ form a basis for the row space of $A$.
> <!-- bilingual-en:end -->

初等行操作只把现有各行作可逆线性重组，所以 $A$ 与 $R$ 的行空间完全相同。RREF 的非零行又必然线性无关：在每个主元列中，只有对应行的主元 $1$ 非零；若这些行的线性组合为零，逐个查看主元列就会迫使所有系数都为零。因此它们既张成原行空间，又没有冗余。

这里可以直接取 $R$ 的行，正是因为目标是行空间。不要把这条规则误搬到列空间：行操作一般会把列空间整体变换为另一个子空间。

复矩阵若把行空间记作 $C(A^*)$，则应把 $R$ 的每条非零行取共轭转置后作为列向量；不能直接把实数情形的 $A^T$ 记号搬过去。

> [!question]- 自检
> 若 $R$ 有三行，其中最后一行全为零，为什么只取前两行？
>
> **答案：** 零行不可能属于任何基；前两条非零行已经线性无关并张成与 $A$ 相同的行空间。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.10sum.pdf|MIT 18.06SC Session 1.10 summary]]：核对 $A$ 与其 RREF 具有相同行空间，以及 RREF 非零行给出行空间基。
