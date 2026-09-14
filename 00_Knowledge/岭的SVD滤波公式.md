---
aliases:
  - "岭正则化把伪逆的 1/σᵢ 滤波因子替换为 σᵢ/(σᵢ²+λ)"
  - SVD filter formula for ridge regularization
  - Ridge spectral filter factors
student_os: knowledge-atom
atom_id: LA-PINV-017
atom_set: pseudoinverse-one-sided-inverses
atom_type: formula
status: source-checked
mastery_state: unassessed
requires:
  - "[[岭正则化]]"
  - "[[奇异值分解]]"
  - "[[小奇异值放大噪声]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
leads_to:
  - "[[截断SVD与岭正则化]]"
related:
  - "[[数值秩]]"
---

# 岭正则化把伪逆的 1/σᵢ 滤波因子替换为 σᵢ/(σᵢ²+λ)
<!-- bilingual-en:start -->
*Ridge regularization replaces the pseudoinverse filter 1/σᵢ by σᵢ/(σᵢ²+λ)*
<!-- bilingual-en:end -->

> [!summary] 滤波公式
> 若 $A=U_r\Sigma_rV_r^*$ 且岭目标使用惩罚 $\lambda\|x\|_2^2$、$\lambda>0$，则
> $$
> \hat x_\lambda
> =\sum_{i=1}^{r}
> \frac{\sigma_i}{\sigma_i^2+\lambda}
> (u_i^*b)v_i.
> $$
> 因而精确伪逆的 $1/\sigma_i$ 被连续滤波因子 $\sigma_i/(\sigma_i^2+\lambda)$ 替代。
> <!-- bilingual-en:start -->
> With penalty $\lambda\|x\|_2^2$, ridge replaces the exact-pseudoinverse factor $1/\sigma_i$ by the continuous filter $\sigma_i/(\sigma_i^2+\lambda)$ in every positive singular direction.
> <!-- bilingual-en:end -->

由正规方程
$$
(A^*A+\lambda I)\hat x_\lambda=A^*b
$$
代入 SVD，便有
$$
A^*A+\lambda I
=V_r(\Sigma_r^2+\lambda I)V_r^*+\lambda P_{N(A)},
$$
从而正奇异方向得到摘要中的系数，零奇异方向仍取零。相对于 $1/\sigma_i$，比值
$$
\frac{\sigma_i/(\sigma_i^2+\lambda)}{1/\sigma_i}
=\frac{\sigma_i^2}{\sigma_i^2+\lambda}
$$
随 $\sigma_i$ 变小而下降，所以小奇异方向收缩更强。
<!-- bilingual-en:start -->
Substituting the SVD into the ridge normal equations gives the stated coefficient in each positive singular direction and zero in the null directions. Relative to exact inversion, the retained fraction is $\sigma_i^2/(\sigma_i^2+\lambda)$, so smaller singular directions are shrunk more strongly.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

对 $A=\operatorname{diag}(1,10^{-4})$ 和 $\lambda=10^{-4}$，两个岭滤波因子分别为
$$
\frac{1}{1+10^{-4}},
\qquad
\frac{10^{-4}}{10^{-8}+10^{-4}}\approx1.
$$
第二个因子远小于精确伪逆的 $10^4$，所以噪声放大被抑制，同时该方向的信号也被强烈收缩。
<!-- bilingual-en:start -->
For $A=\operatorname{diag}(1,10^{-4})$ and $\lambda=10^{-4}$, the second ridge factor is about $1$, far below the exact-pseudoinverse factor $10^4$. Noise amplification is reduced, but signal in that direction is also heavily shrunk.
<!-- bilingual-en:end -->

## 参数化边界
<!-- bilingual-en:start -->
*Parameterisation boundary*
<!-- bilingual-en:end -->

若文献把目标写成 $\|Ax-b\|_2^2+\lambda^2\|x\|_2^2$，分母相应为 $\sigma_i^2+\lambda^2$。这是参数记号改变，不是另一种滤波器。无论哪种记号，$\lambda>0$ 都改变了反演算子；它不是同一个精确伪逆的纯数值实现。
<!-- bilingual-en:start -->
If a source writes the penalty as $\lambda^2\|x\|_2^2$, the denominator becomes $\sigma_i^2+\lambda^2$. This is a change of parameterisation, not a different filter. In either convention a positive regularisation parameter changes the inverse operator rather than merely implementing the same exact pseudoinverse.
<!-- bilingual-en:end -->

> [!question]- 自检
> 当 $\sigma_i$ 很小时，岭滤波因子为什么不会像 $1/\sigma_i$ 那样发散？
>
> **答案：** 分母含有固定的正项 $\lambda$；当 $\sigma_i\to0$ 时，$\sigma_i/(\sigma_i^2+\lambda)\to0$。
>
> <!-- bilingual-en:start -->
> **Question:** Why does the ridge filter not diverge like $1/\sigma_i$ when $\sigma_i$ is small?
>
> **Answer:** The denominator contains the fixed positive term $\lambda$, so $\sigma_i/(\sigma_i^2+\lambda)\to0$ as $\sigma_i\to0$.
> <!-- bilingual-en:end -->

## 来源与核验

- P. C. Hansen, [Intro to Inverse Problems, Chapter 4](https://www2.imm.dtu.dk/~pcha/DIP/chap4.pdf#page=16)：核对 Tikhonov 的奇异方向滤波展开、连续收缩及参数化差异。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.8.8 边界、反例与易错点|课程 3.8.8]]：核对正则化改变原精确反演算子的边界。
<!-- bilingual-en:start -->
- Hansen, Chapter 4, supports the filtered SVD expansion of Tikhonov regularisation, continuous shrinkage, and the parameterisation convention.
- Course Section 3.8.8 supports the boundary that regularisation changes the exact inverse operator.
<!-- bilingual-en:end -->
