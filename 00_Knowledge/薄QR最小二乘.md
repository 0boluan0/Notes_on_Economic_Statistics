---
aliases:
  - "满列秩薄 QR 把最小二乘化为上三角方程"
  - Thin QR least-squares solve
  - Reduced QR least squares
  - QR 最小二乘
student_os: knowledge-atom
atom_id: LA-PROJ-010
atom_set: orthogonal-projection-least-squares
atom_type: procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[薄QR分解]]"
  - "[[线性最小二乘]]"
related:
  - "[[Gram-Schmidt数值失稳]]"
  - "[[正规方程条件数平方]]"
contrasts_with:
  - "[[秩亏最小二乘算法]]"
part_of:
  - "[[正交投影与最小二乘.canvas]]"
---

# 满列秩薄 QR 把最小二乘化为上三角方程
<!-- bilingual-en:start -->
*A full-rank thin QR factorisation reduces least squares to a triangular system*
<!-- bilingual-en:end -->

> [!summary] 求解规则
> 若 $A\in\mathbb R^{m\times n}$ 满列秩且 $m\ge n$，取薄 QR 分解
> $$
> A=QR,\qquad Q^TQ=I_n,\qquad R\in\mathbb R^{n\times n}
> $$
> 其中 $R$ 上三角可逆。则唯一最小二乘解由
> $$
> R\hat x=Q^Tb
> $$
> 给出；计算时解上三角系统，不显式形成 $R^{-1}$。
> <!-- bilingual-en:start -->
> If $A\in\mathbb R^{m\times n}$ has full column rank and $m\ge n$, let $A=QR$ be a thin QR factorisation with $Q^TQ=I_n$ and nonsingular upper-triangular $R$. The unique least-squares solution satisfies $R\hat x=Q^Tb$. Compute it by triangular solution rather than forming $R^{-1}$.
> <!-- bilingual-en:end -->

## 为什么三角方程就是最小二乘
<!-- bilingual-en:start -->
*Why the triangular system solves least squares*
<!-- bilingual-en:end -->

把 $Q$ 补成方阵正交基 $[Q\ Q_\perp]$。正交变换保持二范数，因此
$$
\begin{aligned}
\|Ax-b\|_2^2
&=\left\|\begin{bmatrix}Q^T\\Q_\perp^T\end{bmatrix}(QRx-b)\right\|_2^2\\
&=\|Rx-Q^Tb\|_2^2+\|Q_\perp^Tb\|_2^2.
\end{aligned}
$$
第二项与 $x$ 无关；令第一项为零便得到 $R\hat x=Q^Tb$。相应拟合值是 $QQ^Tb$，残差向量与范数分别是
$$
r=b-QQ^Tb=Q_\perp Q_\perp^Tb\in C(A)^\perp,
\qquad \|r\|_2=\|Q_\perp^Tb\|_2.
$$
<!-- bilingual-en:start -->
Extend $Q$ to the square orthogonal matrix $[Q\ Q_\perp]$. Norm preservation splits the objective into $\|Rx-Q^Tb\|_2^2+\|Q_\perp^Tb\|_2^2$. The second term is independent of $x$, so the minimum occurs at $R\hat x=Q^Tb$. The fitted vector is $QQ^Tb$, while $r=b-QQ^Tb=Q_\perp Q_\perp^Tb\in C(A)^\perp$ and $\|r\|_2=\|Q_\perp^Tb\|_2$.
<!-- bilingual-en:end -->

## 使用边界
<!-- bilingual-en:start -->
*Use boundary*
<!-- bilingual-en:end -->

这条程序依赖 $R$ 可逆，也就是 $A$ 满列秩。若 $A$ 精确秩亏，无主元薄 QR 的可逆三角求解失效，应改用 [[秩亏最小二乘算法]]。若 $A$ 只是接近秩亏但仍满列秩，Householder 薄 QR 在代数上仍适用，但问题病态时系数的前向误差可能很大。是否把小方向视为零，必须另行声明数值秩阈值。
<!-- bilingual-en:start -->
This procedure requires nonsingular $R$, equivalently full column rank of $A$. Under exact rank deficiency, the unpivoted thin-QR triangular solve fails; use [[秩亏最小二乘算法|a rank-deficient least-squares method]] instead. If $A$ is merely nearly rank deficient but still has full column rank, Householder thin QR remains algebraically applicable, although ill-conditioning can cause large forward errors in the coefficients. Treating small directions as zero requires a separately stated numerical-rank threshold.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 为什么 QR 最小二乘只需 $Q^Tb$，而不必构造完整的 $m\times m$ 正交矩阵？
> <!-- bilingual-en:start -->
> Why does a QR least-squares solve need only $Q^Tb$ rather than a full $m\times m$ orthogonal matrix?
> <!-- bilingual-en:end -->
>
> **答案：** 系数只由 $b$ 在 $C(A)$ 的坐标决定；正交补分量只决定无法消除的残差，不影响 $\hat x$。
> <!-- bilingual-en:start -->
> **Answer:** The coefficients depend only on the coordinates of $b$ in $C(A)$; the orthogonal-complement component determines the irreducible residual but not $\hat x$.
> <!-- bilingual-en:end -->

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.4sum.pdf|MIT 18.06SC Session 2.4 summary]]：支持 $A=QR$、$Q^TQ=I$ 与上三角 $R$。
- [[01_Math/02_linear algebra/02_Least Squares, Determinants and Eigenvalues.md#从 Gram–Schmidt 到 QR 分解|课程：从 Gram–Schmidt 到 QR]]：支持薄 QR 尺寸、$R\hat x=Q^Tb$ 与满列秩边界。
- [LAPACK Users’ Guide: Linear Least Squares Problems](https://www.netlib.org/lapack/lug/node27.html)：支持 full-rank least squares 使用 QR/LQ，而秩亏问题转向完整正交分解或 SVD。
<!-- bilingual-en:start -->
- The MIT 18.06SC Session 2.4 summary supports $A=QR$, $Q^TQ=I$, and upper-triangular $R$.
- The course section from Gram-Schmidt to QR supports the thin dimensions, $R\hat x=Q^Tb$, and the full-column-rank boundary.
- The LAPACK Users’ Guide supports QR/LQ for full-rank least squares and complete orthogonal or SVD methods for rank-deficient problems.
<!-- bilingual-en:end -->
