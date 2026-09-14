---
aliases:
  - "矩阵的 Moore–Penrose 伪逆由紧 SVD 中的正奇异值取倒数得到"
  - SVD formula for the Moore-Penrose pseudoinverse
student_os: knowledge-atom
atom_id: LA-PINV-016
atom_set: pseudoinverse-one-sided-inverses
atom_type: formula
status: source-checked
mastery_state: unassessed
requires:
  - "[[Moore–Penrose 伪逆]]"
  - "[[紧SVD]]"
related:
  - "[[满列秩伪逆公式]]"
  - "[[满行秩伪逆公式]]"
  - "[[AA+列空间投影]]"
  - "[[A+A行空间投影]]"
leads_to:
  - "[[伪逆存在唯一性]]"
  - "[[小奇异值放大噪声]]"
  - "[[截断SVD反演]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
---

# 矩阵的 Moore–Penrose 伪逆由紧 SVD 中的正奇异值取倒数得到
<!-- bilingual-en:start -->
*The Moore–Penrose pseudoinverse is obtained by reciprocating the positive singular values in a compact SVD*
<!-- bilingual-en:end -->

> [!summary] 公式
> 若秩为 $r$ 的矩阵 $A\in\mathbb F^{m\times n}$ 具有紧 SVD
> $$
> A=U_r\Sigma_rV_r^*,
> $$
> 则
> $$
> A^+=V_r\Sigma_r^{-1}U_r^*.
> $$
> 也就是说，每个正奇异值 $\sigma_i$ 被换成 $1/\sigma_i$；零奇异方向不进入紧 SVD，也不会被伪逆恢复。
> <!-- bilingual-en:start -->
> If a rank-$r$ matrix $A\in\mathbb F^{m\times n}$ has compact SVD $A=U_r\Sigma_rV_r^*$, then $A^+=V_r\Sigma_r^{-1}U_r^*$. Each positive singular value $\sigma_i$ is replaced by $1/\sigma_i$; zero singular directions are absent from the compact SVD and are not recovered by the pseudoinverse.
> <!-- bilingual-en:end -->

## 方向上的作用
<!-- bilingual-en:start -->
*Action on singular directions*
<!-- bilingual-en:end -->

对任意 $b$，
$$
A^+b=\sum_{i=1}^{r}\frac{u_i^*b}{\sigma_i}v_i.
$$
$b$ 在 $u_i$ 方向上的分量被映到相应的 $v_i$ 方向并除以 $\sigma_i$；与 $C(A)$ 正交的分量被消去。由此立即得到
$$
AA^+=U_rU_r^*,
\qquad
A^+A=V_rV_r^*.
$$
<!-- bilingual-en:start -->
For any $b$, $A^+b=\sum_{i=1}^{r}(u_i^*b/\sigma_i)v_i$. The component of $b$ along $u_i$ is transferred to the matching $v_i$ direction and divided by $\sigma_i$; components orthogonal to $C(A)$ are removed. Consequently, $AA^+=U_rU_r^*$ and $A^+A=V_rV_r^*$.
<!-- bilingual-en:end -->

## 精确公式与数值阈值的边界
<!-- bilingual-en:start -->
*Boundary between the exact formula and numerical thresholding*
<!-- bilingual-en:end -->

这是原矩阵精确伪逆的代数公式，所有正奇异值都要取倒数。若数值算法按容差把一个正但很小的奇异值当作零，得到的是所采用数值秩下的解；若主动截掉该方向以抑制噪声，则进入 [[截断SVD反演]]，已经改变了反演算子。不要把阈值化结果无条件称为原矩阵的精确 $A^+$。
<!-- bilingual-en:start -->
This is the algebraic formula for the exact pseudoinverse of the original matrix, so every positive singular value is reciprocated. A numerical algorithm may treat a small positive value as zero under a declared tolerance, producing the solution for the adopted numerical rank. Deliberately removing that direction to control noise gives [[截断SVD反演|truncated SVD inversion]] and changes the inverse operator. A thresholded result should not be called the exact $A^+$ of the original matrix without qualification.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

若
$$
A=\begin{bmatrix}3&0\\0&0\end{bmatrix},
$$
则唯一的正奇异值是 $3$，因此
$$
A^+=\begin{bmatrix}1/3&0\\0&0\end{bmatrix}.
$$
伪逆撤销第一个方向的三倍伸缩，同时把无法由 $A$ 产生的第二个输出方向送到零。
<!-- bilingual-en:start -->
For $A=\operatorname{diag}(3,0)$, the only positive singular value is $3$, so $A^+=\operatorname{diag}(1/3,0)$. The pseudoinverse reverses the threefold scaling in the first direction and sends the unattainable second output direction to zero.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 在精确伪逆公式中，零奇异方向应当取一个很大的倒数，还是保持为零？
> <!-- bilingual-en:start -->
> In the exact pseudoinverse formula, should a zero singular direction receive a very large reciprocal or remain zero?
> <!-- bilingual-en:end -->
>
> **答案：** 保持为零。伪逆只对正奇异值取倒数，零方向没有可逆信息。
> <!-- bilingual-en:start -->
> **Answer:** It remains zero. The pseudoinverse reciprocates only positive singular values; a zero direction contains no invertible information.
> <!-- bilingual-en:end -->

## 来源与核验

- R. Penrose, [“A generalized inverse for matrices”](https://doi.org/10.1017/S0305004100030401), *Proceedings of the Cambridge Philosophical Society* 51 (1955), 406–413：核验任意形状实复矩阵的规范伪逆及其唯一性背景。
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|MIT 18.06SC Session 3.8 summary]]：核验紧 SVD 伪逆公式、左右投影及最小范数解释。
- [LAPACK Users' Guide: Singular Value Decomposition](https://www.netlib.org/lapack/lug/node32.html)：核对矩形 SVD 的尺寸与正奇异块。
<!-- bilingual-en:start -->
- Penrose's 1955 paper supports the canonical pseudoinverse for arbitrary real or complex matrix shapes and the uniqueness background.
- The MIT 18.06SC Session 3.8 summary supports the compact-SVD formula, the two orthogonal projectors, and the minimum-norm interpretation.
- The LAPACK Users' Guide supports the dimensions of a rectangular SVD and its positive singular block.
<!-- bilingual-en:end -->
