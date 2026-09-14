---
aliases:
  - "任意矩阵的 $A^+A$ 是其行空间正交投影"
  - "任意矩阵的 A+A 是其行空间正交投影"
  - A plus A row-space projector
  - Pseudoinverse row-space projection identity
student_os: knowledge-atom
atom_id: LA-PROJ-014
atom_set: orthogonal-projection-least-squares
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[正交投影矩阵判别]]"
  - "[[Moore–Penrose 伪逆]]"
related:
  - "[[AA+列空间投影]]"
leads_to:
  - "[[最小范数最小二乘解]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
  - "[[广义逆与最小范数解.canvas]]"
---

# 任意矩阵的 $A^+A$ 是其行空间正交投影
<!-- bilingual-en:start -->
*For every matrix $A$, $A^+A$ is the orthogonal projector onto its row space*
<!-- bilingual-en:end -->

> [!summary] 核心恒等式
> 对任意实矩阵 $A\in\mathbb R^{m\times n}$ 及其 Moore–Penrose 伪逆 $A^+$，
> $$
> A^+A=P_{C(A^T)}=P_{N(A)^\perp}.
> $$
> 它把输入向量保留在行空间中的分量，并去掉不会改变 $Ax$ 的零空间分量。
> <!-- bilingual-en:start -->
> For every real matrix $A\in\mathbb R^{m\times n}$ and its Moore–Penrose pseudoinverse $A^+$, $A^+A=P_{C(A^T)}=P_{N(A)^\perp}$. It retains the row-space component of an input vector and removes the null-space component that cannot affect $Ax$.
> <!-- bilingual-en:end -->

## SVD 直接显出行空间投影
<!-- bilingual-en:start -->
*The SVD exposes the row-space projector directly*
<!-- bilingual-en:end -->

取紧 SVD
$$
A=U_r\Sigma_rV_r^T,\qquad A^+=V_r\Sigma_r^{-1}U_r^T.
$$
$V_r$ 的列标准正交并张成 $C(A^T)$，因此
$$
A^+A=V_rV_r^T.
$$
这正是标准正交基投影公式，所以 $A^+A$ 对称且幂等，值域为 $C(A^T)$，零空间为 $N(A)$。复矩阵情形把转置换为共轭转置，目标空间相应为 $C(A^*)$。
<!-- bilingual-en:start -->
For a compact SVD $A=U_r\Sigma_rV_r^T$ with $A^+=V_r\Sigma_r^{-1}U_r^T$, the orthonormal columns of $V_r$ span $C(A^T)$. Hence $A^+A=V_rV_r^T$, the orthonormal-basis projector onto the row space. It is symmetric and idempotent, with range $C(A^T)$ and kernel $N(A)$. Over the complex numbers, transpose becomes conjugate transpose and the target space is $C(A^*)$.
<!-- bilingual-en:end -->

## 最小例子与边界
<!-- bilingual-en:start -->
*Minimal example and boundary*
<!-- bilingual-en:end -->

令 $A=\begin{bmatrix}1&0\end{bmatrix}$，则 $A^+=\begin{bmatrix}1\\0\end{bmatrix}$，所以
$$
A^+A=
\begin{bmatrix}
1&0\\
0&0
\end{bmatrix}.
$$
它把 $(x_1,x_2)^T$ 变为 $(x_1,0)^T$：第一项是行空间分量，第二项属于 $N(A)$，不会影响 $Ax$。只有当 $A$ 满列秩、行空间等于整个输入空间时，才有 $A^+A=I_n$。
<!-- bilingual-en:start -->
For $A=[1\ 0]$, one has $A^+=(1,0)^T$ and $A^+A=\operatorname{diag}(1,0)$. It sends $(x_1,x_2)^T$ to $(x_1,0)^T$: the first component lies in the row space, while the discarded second component lies in $N(A)$ and cannot affect $Ax$. The identity $A^+A=I_n$ holds only when $A$ has full column rank and its row space fills the input space.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 为什么向 $x$ 加上 $z\in N(A)$ 不会改变 $A^+Ax$？
> <!-- bilingual-en:start -->
> Why does adding $z\in N(A)$ to $x$ leave $A^+Ax$ unchanged?
> <!-- bilingual-en:end -->
>
> **答案：** $A^+Az=0$；等价地，$A^+A$ 会删除零空间分量，只保留行空间分量。
> <!-- bilingual-en:start -->
> **Answer:** $A^+Az=0$; equivalently, $A^+A$ removes the null-space component and retains only the row-space component.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|MIT 18.06SC Session 3.8 summary]]：支持用紧 SVD 构造 $A^+$，以及 $A^+A$ 投影到行空间并去掉零空间分量。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.8.3 SVD 定义伪逆|课程 3.8.3]]：支持 $A^+A=V_rV_r^T$、其值域和零空间。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 3.8 summary supports the compact-SVD construction of $A^+$ and the fact that $A^+A$ projects onto the row space while removing the null-space component.
- Course Section 3.8.3 supports $A^+A=V_rV_r^T$ and its range and kernel.
<!-- bilingual-en:end -->
