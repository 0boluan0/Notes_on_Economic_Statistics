---
aliases:
  - "Gram-Schmidt 正交化逐步去除已有方向的投影并归一化从而保持张成空间不变"
  - Gram-Schmidt orthogonalization
  - Classical Gram-Schmidt
  - 经典 Gram-Schmidt 正交化
student_os: knowledge-atom
atom_id: LA-PROJ-015
atom_set: orthogonal-projection-least-squares
atom_type: procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[标准正交组]]"
  - "[[标准正交基投影公式]]"
leads_to:
  - "[[薄QR分解]]"
  - "[[Gram-Schmidt数值失稳]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---

# Gram-Schmidt 正交化逐步去除已有方向的投影并归一化从而保持张成空间不变
<!-- bilingual-en:start -->
*Gram-Schmidt orthogonalization removes projections onto previously constructed directions and normalizes the remainders while preserving the span*
<!-- bilingual-en:end -->

> [!summary] 计算步骤
> 设 $A=[a_1\ \cdots\ a_n]\in\mathbb R^{m\times n}$ 的列线性无关。第 $k$ 步先减去 $a_k$ 在已得标准正交向量上的投影，再归一化：
> $$
> r_{ik}=q_i^Ta_k\quad(i<k),\qquad
> v_k=a_k-\sum_{i=1}^{k-1}q_ir_{ik},
> $$
> $$
> r_{kk}=\|v_k\|_2,\qquad q_k=\frac{v_k}{r_{kk}}.
> $$
> 得到的 $q_1,\ldots,q_n$ 两两正交且长度为 $1$，并与原列向量张成同一空间。
> <!-- bilingual-en:start -->
> Let $A=[a_1\ \cdots\ a_n]\in\mathbb R^{m\times n}$ have linearly independent columns. At step $k$, subtract the projections of $a_k$ onto the previously constructed orthonormal vectors, then normalize the remainder. The resulting vectors $q_1,\ldots,q_n$ are orthonormal and span the same space as the original columns.
> <!-- bilingual-en:end -->

## 为什么张成空间不变
<!-- bilingual-en:start -->
*Why the span is preserved*
<!-- bilingual-en:end -->

$v_k$ 由 $a_k$ 减去 $q_1,\ldots,q_{k-1}$ 的线性组合得到，因而
$$
v_k\in\operatorname{span}(a_1,\ldots,a_k).
$$
反过来，由
$$
a_k=v_k+\sum_{i=1}^{k-1}q_ir_{ik}
$$
可知 $a_k$ 也在 $q_1,\ldots,q_k$ 的张成空间中。归纳即得
$$
\operatorname{span}(q_1,\ldots,q_k)
=\operatorname{span}(a_1,\ldots,a_k).
$$
列线性无关保证 $v_k\ne0$，所以每次归一化都有定义。
<!-- bilingual-en:start -->
The remainder $v_k$ is obtained from $a_k$ by subtracting a linear combination of earlier $q_i$, so it lies in the span of the first $k$ original columns. Conversely, the displayed reconstruction of $a_k$ places it in the span of $q_1,\ldots,q_k$. Induction gives equality of the two prefix spans. Linear independence ensures that no remainder is zero, so every normalization is defined.
<!-- bilingual-en:end -->

## 与 QR 分解的关系
<!-- bilingual-en:start -->
*Connection with QR factorization*
<!-- bilingual-en:end -->

把系数 $r_{ik}$ 排成上三角矩阵 $R$，并令 $Q=[q_1\ \cdots\ q_n]$，就得到
$$
A=QR,\qquad Q^TQ=I_n.
$$
这说明 Gram-Schmidt 是构造 [[薄QR分解]] 的一种精确算术过程；它在浮点运算中的可靠性是另一个问题。
<!-- bilingual-en:start -->
Collecting the coefficients $r_{ik}$ into an upper-triangular matrix $R$ and setting $Q=[q_1\ \cdots\ q_n]$ gives $A=QR$ with $Q^TQ=I_n$. Gram-Schmidt is therefore one exact-arithmetic construction of a [[薄QR分解|thin QR factorization]]; its floating-point reliability is a separate question.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 若第 $k$ 步得到 $v_k=0$，这对原列向量意味着什么？
> <!-- bilingual-en:start -->
> If step $k$ produces $v_k=0$, what does that imply about the original columns?
> <!-- bilingual-en:end -->
>
> **答案：** $a_k$ 已经落在 $a_1,\ldots,a_{k-1}$ 的张成空间中，因此前 $k$ 列线性相关。
> <!-- bilingual-en:start -->
> **Answer:** The column $a_k$ already lies in the span of $a_1,\ldots,a_{k-1}$, so the first $k$ columns are linearly dependent.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|MIT 18.06SC Session 2.4 summary]]：核验逐步减投影、归一化、保持张成空间以及 $A=QR$。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 2.4 summary supports the projection-subtraction steps, normalization, preservation of the span, and the construction $A=QR$.
<!-- bilingual-en:end -->
