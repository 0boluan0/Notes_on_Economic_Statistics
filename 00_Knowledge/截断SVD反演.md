---
aliases:
  - "截断 SVD 反演只对高于阈值的奇异值取倒数"
  - Truncated SVD inversion
  - TSVD inverse solution
student_os: knowledge-atom
atom_id: LA-PINV-012
atom_set: pseudoinverse-one-sided-inverses
atom_type: procedure
status: source-checked
mastery_state: unassessed
requires:
  - "[[伪逆SVD公式]]"
  - "[[数值秩]]"
  - "[[小奇异值放大噪声]]"
related:
  - "[[截断SVD]]"
  - "[[秩亏最小二乘算法]]"
leads_to:
  - "[[截断SVD与岭正则化]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
---

# 截断 SVD 反演只对高于阈值的奇异值取倒数
<!-- bilingual-en:start -->
*Truncated SVD inversion reciprocates only singular values above a threshold*
<!-- bilingual-en:end -->

> [!summary] 计算规则
> 若
> $$
> A=\sum_{i=1}^{r}\sigma_i u_iv_i^*,
> \qquad \sigma_1\ge\cdots\ge\sigma_r>0,
> $$
> 给定阈值 $\tau\ge0$ 后，截断 SVD 反演定义
> $$
> \hat x_\tau
> =\sum_{\sigma_i>\tau}\frac{u_i^*b}{\sigma_i}v_i.
> $$
> 它完全反演阈值上的方向，并把阈值下的方向直接设为零。
> <!-- bilingual-en:start -->
> If $A=\sum_{i=1}^{r}\sigma_i u_iv_i^*$, truncated SVD inversion at threshold $\tau\ge0$ defines $\hat x_\tau=\sum_{\sigma_i>\tau}(u_i^*b/\sigma_i)v_i$. It fully inverts directions above the threshold and sets directions below it to zero.
> <!-- bilingual-en:end -->

## 它与精确伪逆的边界
<!-- bilingual-en:start -->
*Boundary with exact pseudoinversion*
<!-- bilingual-en:end -->

若 $\tau<\sigma_r$，没有正奇异值被截掉，则 $\hat x_\tau=A^+b$。一旦某个正奇异值被设为零，所得算子就不再是原矩阵的精确 Moore–Penrose 伪逆；它用偏差换取对噪声的稳定性。阈值同时决定采用的 [[数值秩]]，因而必须记录尺度与容差口径。
<!-- bilingual-en:start -->
If $\tau<\sigma_r$, no positive singular value is removed and $\hat x_\tau=A^+b$. Once a positive singular direction is set to zero, the resulting operator is no longer the exact Moore–Penrose pseudoinverse of the original matrix; it trades bias for reduced noise amplification. The threshold also fixes the adopted [[数值秩|numerical rank]], so its scale and tolerance convention must be reported.
<!-- bilingual-en:end -->

## 不要与低秩矩阵近似混同
<!-- bilingual-en:start -->
*Do not confuse inversion with low-rank matrix approximation*
<!-- bilingual-en:end -->

[[截断SVD]] 中的 $A_k$ 直接近似矩阵 $A$；这里的 $\hat x_\tau$ 则修改从观测 $b$ 恢复系数 $x$ 的反演算子。两者都使用奇异方向的硬截断，但作用的对象不同。
<!-- bilingual-en:start -->
The matrix $A_k$ in a [[截断SVD|rank-$k$ truncated SVD]] approximates $A$ itself. The vector $\hat x_\tau$ here changes the inverse operator that recovers $x$ from observations $b$. Both use hard truncation in singular directions, but they act on different objects.
<!-- bilingual-en:end -->

## 最小例子
<!-- bilingual-en:start -->
*Minimal example*
<!-- bilingual-en:end -->

若 $A=\operatorname{diag}(1,10^{-4})$ 且 $\tau=10^{-3}$，则截断反演保留第一个方向，把第二个方向设为零。它避免了 $10^4$ 倍的误差放大，也同时放弃恢复该方向中的信号。
<!-- bilingual-en:start -->
For $A=\operatorname{diag}(1,10^{-4})$ and $\tau=10^{-3}$, TSVD retains the first direction and sets the second to zero. It avoids a $10^4$ amplification of error, but also gives up recovery of signal in that direction.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 若阈值没有截掉任何正奇异值，截断 SVD 反演是否已经改变了原精确伪逆？
> <!-- bilingual-en:start -->
> If the threshold removes no positive singular value, has TSVD inversion changed the exact pseudoinverse?
> <!-- bilingual-en:end -->
>
> **答案：** 没有；此时求和包含全部正奇异方向，所以 $\hat x_\tau=A^+b$。
> <!-- bilingual-en:start -->
> **Answer:** No. Every positive singular direction remains in the sum, so $\hat x_\tau=A^+b$.
> <!-- bilingual-en:end -->

## 来源与核验

- P. C. Hansen, [*Intro to Inverse Problems*, Chapter 4](https://www2.imm.dtu.dk/~pcha/DIP/chap4.pdf#page=16)：核验截断 SVD 的滤波因子展开、噪声抑制与偏差边界。
- [LAPACK Users’ Guide: Linear Least Squares Problems](https://www.netlib.org/lapack/lug/node27.html)：核验 `RCOND`、有效秩与 SVD 最小范数求解的关系。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.8.8 边界、反例与易错点|课程 3.8.8]]：核对小奇异值与阈值化反演的使用边界。
<!-- bilingual-en:start -->
- Hansen, *Intro to Inverse Problems*, Chapter 4, supports the filter-factor expansion, noise suppression, and bias boundary of truncated SVD inversion.
- The LAPACK Users’ Guide supports the relation among `RCOND`, effective rank, and SVD-based minimum-norm solving.
- Course Section 3.8.8 checks the practical boundary between small singular values and thresholded inversion.
<!-- bilingual-en:end -->
