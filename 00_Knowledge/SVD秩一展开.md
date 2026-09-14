---
aliases:
  - "紧 SVD 可以把矩阵展开为在 Frobenius 内积下彼此正交的奇异值加权秩一外积之和"
  - 奇异值分解的秩一展开
  - Rank-one expansion of the SVD
student_os: knowledge-atom
atom_id: LA-SVD-023
atom_set: singular-value-decomposition-low-rank
atom_type: identity
status: source-checked
mastery_state: unassessed
requires:
  - "[[紧SVD]]"
  - "[[秩一矩阵外积刻画]]"
  - "[[奇异向量]]"
part_of:
  - "[[奇异值分解与低秩近似.canvas]]"
leads_to:
  - "[[截断SVD]]"
  - "[[奇异向量共同相位]]"
related:
  - "[[矩阵乘积外积展开]]"
---

# 紧 SVD 可以把矩阵展开为在 Frobenius 内积下彼此正交的奇异值加权秩一外积之和
<!-- bilingual-en:start -->
*The compact SVD expands a matrix as a sum of singular-value-weighted rank-one outer products that are mutually orthogonal in the Frobenius inner product*
<!-- bilingual-en:end -->

> [!summary] 核心恒等式
> 若 $A$ 的秩为 $r$，紧 SVD 为 $A=U_r\Sigma_rV_r^*$，则
> $$
> A=\sum_{i=1}^{r}\sigma_i u_iv_i^*.
> $$
> 每个 $u_iv_i^*$ 都是秩一矩阵：它读取输入在 $v_i$ 方向上的坐标，再沿 $u_i$ 方向输出；$\sigma_i$ 决定这一分量的尺度。
> <!-- bilingual-en:start -->
> Each rank-one term reads the input coordinate along $v_i$ and outputs it along $u_i$, scaled by $\sigma_i$.
> <!-- bilingual-en:end -->

这个展开只是矩阵乘法的逐列—逐行形式：
$$
U_r\Sigma_rV_r^*
=\begin{bmatrix}u_1&\cdots&u_r\end{bmatrix}
\begin{bmatrix}\sigma_1&&\\&\ddots&\\&&\sigma_r\end{bmatrix}
\begin{bmatrix}v_1^*\\\vdots\\v_r^*\end{bmatrix}.
$$
由于两组奇异向量都标准正交，不同秩一分量在 Frobenius 内积下也彼此正交。

例如
$$
A=\begin{bmatrix}3&0\\0&1\end{bmatrix}
=3e_1e_1^T+e_2e_2^T.
$$
第一项描述强度为 $3$ 的第一方向，第二项描述强度为 $1$ 的第二方向。[[截断SVD]] 正是从这个和式中保留前若干项。

> [!question]- 自检
> 为什么 $u_iv_i^*$ 把任意输入 $x$ 送到 $u_i$ 方向？
>
> **答案：** $(u_iv_i^*)x=u_i(v_i^*x)$；括号内是一个标量，所以输出必为 $u_i$ 的倍数。

## 来源与核验

- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.5sum.pdf|MIT 18.06SC Session 3.5 summary]]：核对 SVD 的秩一展开及输入—输出解释。
- [[矩阵乘积外积展开]]：核对三因子乘积展开为列—行外积之和的代数形式。
