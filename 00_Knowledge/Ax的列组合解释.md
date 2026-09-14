---
aliases:
  - Ax 是 A 的列向量按 x 的坐标形成的线性组合
  - Ax as a linear combination of columns
  - 列组合解释
student_os: knowledge-atom
atom_id: LA-SYS-001
atom_set: linear-systems-four-subspaces
atom_type: identity
status: source-checked
mastery_state: unassessed
related:
  - "[[列空间]]"
  - "[[表示矩阵的列]]"
  - "[[张成]]"
leads_to:
  - "[[方程组解分类]]"
part_of:
  - "[[线性方程组与四个基本子空间.canvas]]"
---

# Ax 是 A 的列向量按 x 的坐标形成的线性组合
<!-- bilingual-en:start -->
*$Ax$ is the linear combination of the columns of $A$ weighted by the coordinates of $x$*
<!-- bilingual-en:end -->

> [!summary] 核心恒等式
> 若 $A=[a_1\ \cdots\ a_n]\in\mathbb F^{m\times n}$ 且 $x=(x_1,\ldots,x_n)^T$，则
> $$
> Ax=x_1a_1+\cdots+x_na_n.
> $$
> 因而 $A$ 的[[列空间]]正是所有可能输出：
> $$
> C(A)=\{Ax:x\in\mathbb F^n\}.
> $$
> <!-- bilingual-en:start -->
> If $A=[a_1\ \cdots\ a_n]$ and $x=(x_j)$, then $Ax=\sum_jx_ja_j$. Hence the column space is exactly the image $C(A)=\{Ax:x\in\mathbb F^n\}$.
> <!-- bilingual-en:end -->

这个读法把矩阵乘法直接翻译成可达性问题：$Ax=b$ 有解，意味着可以用 $A$ 的列向量线性组合出 $b$。坐标向量 $x$ 记录组合系数，而不是输出空间中的另一个几何点。
<!-- bilingual-en:start -->
This reading turns matrix multiplication into a reachability question: solving $Ax=b$ means expressing $b$ as a linear combination of the columns. The vector $x$ records the coefficients of that combination.
<!-- bilingual-en:end -->

## 边界

$C(A)\subseteq\mathbb F^m$，因为每一列有 $m$ 个分量；$x\in\mathbb F^n$，因为组合中有 $n$ 个系数。只有方阵时两边环境空间维数才相同，但即使维数相同，也不能把输入坐标和输出向量的角色混为一谈。

> [!question]- 自检
> 若 $b$ 不在 $C(A)$ 中，改变 $x$ 的取值能否让 $Ax=b$ 成立？
>
> **答案：** 不能。$C(A)$ 已经包含全部可能的 $Ax$。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.1sum.pdf|MIT 18.06SC Session 1.1 summary]]：核对 $Ax$ 的列线性组合解释及其与 $Ax=b$ 的关系。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses1.6sum.pdf|MIT 18.06SC Session 1.6 summary]]：核对列空间定义与 $b\in C(A)$ 的可解性解释。
