---
aliases:
  - "截断 SVD 反演按硬阈值删除奇异方向而岭正则化连续收缩每个正奇异方向"
  - Truncated SVD inversion versus ridge regularization
  - TSVD versus Tikhonov regularization
student_os: knowledge-atom
atom_id: LA-PINV-008
atom_set: pseudoinverse-one-sided-inverses
atom_type: method-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[截断SVD反演]]"
  - "[[岭正则化]]"
  - "[[岭的SVD滤波公式]]"
part_of:
  - "[[广义逆与最小范数解.canvas]]"
related:
  - "[[小奇异值放大噪声]]"
  - "[[数值秩]]"
  - "[[正则化判别]]"
---

# 截断 SVD 反演按硬阈值删除奇异方向而岭正则化连续收缩每个正奇异方向
<!-- bilingual-en:start -->
*Truncated SVD inversion removes singular directions with a hard threshold, whereas ridge regularization continuously shrinks every positive singular direction*
<!-- bilingual-en:end -->

> [!summary] 核心区别
> [[截断SVD反演]]在阈值处作离散选择：阈值上的方向使用精确倒数，阈值下的方向直接归零。[[岭的SVD滤波公式|岭正则化]]不作这种保留或删除的二分，而是用连续滤波因子收缩每个正奇异方向。前者通过阈值选定一个离散有效秩，后者通常仍保留全部正奇异方向。
> <!-- bilingual-en:start -->
> [[截断SVD反演|Truncated SVD inversion]] makes a discrete choice at a threshold: directions above it use the exact reciprocal, while directions below it are set to zero. [[岭的SVD滤波公式|Ridge regularization]] does not make this keep-or-delete decision; it continuously shrinks every positive singular direction. The former selects a discrete effective rank through the threshold, whereas the latter normally retains all positive singular directions.
> <!-- bilingual-en:end -->

## 滤波规则的直接对照
<!-- bilingual-en:start -->
*Direct comparison of the filter rules*
<!-- bilingual-en:end -->

若 $A=U_r\Sigma_rV_r^*$，两种方法都可写成沿奇异方向处理 $u_i^*b$，但使用不同的滤波系数：
$$
g_i^{\mathrm{TSVD}}(\tau)
=
\begin{cases}
1/\sigma_i,&\sigma_i>\tau,\\
0,&\sigma_i\le\tau,
\end{cases}
\qquad
g_i^{\mathrm{ridge}}(\lambda)
=\frac{\sigma_i}{\sigma_i^2+\lambda},
\quad \lambda>0.
$$
因此 TSVD 在阈值处不连续：方向要么完整反演，要么完全删除。岭的系数随 $\sigma_i$ 连续变化；奇异值越小，相对于精确倒数 $1/\sigma_i$ 的收缩越强。
<!-- bilingual-en:start -->
For $A=U_r\Sigma_rV_r^*$, both methods act on $u_i^*b$ along the singular directions, but they use different filter coefficients: TSVD uses $1/\sigma_i$ above the threshold and zero at or below it, whereas ridge uses $\sigma_i/(\sigma_i^2+\lambda)$ for $\lambda>0$. TSVD is discontinuous at the threshold: a direction is either inverted fully or removed completely. The ridge coefficient varies continuously with $\sigma_i$, and smaller singular values are shrunk more strongly relative to the exact reciprocal $1/\sigma_i$.
<!-- bilingual-en:end -->

## 两个参数回答不同问题
<!-- bilingual-en:start -->
*The two parameters answer different questions*
<!-- bilingual-en:end -->

TSVD 的 $\tau$ 回答“哪些方向要当作不可分辨并删除”，因而直接确定 [[数值秩]]。岭的 $\lambda$ 回答“每个方向应收缩多少”，并不产生一个离散的数值秩。两者都可抑制 [[小奇异值放大噪声]]，但 $\tau$ 与 $\lambda$ 不是可以直接互换的同一种参数。
<!-- bilingual-en:start -->
The TSVD threshold $\tau$ answers which directions should be treated as unresolvable and removed, so it directly determines a [[数值秩|numerical rank]]. The ridge parameter $\lambda$ answers how strongly each direction should be shrunk and does not itself produce a discrete numerical rank. Both can reduce [[小奇异值放大噪声|noise amplification by small singular values]], but $\tau$ and $\lambda$ are not interchangeable versions of the same parameter.
<!-- bilingual-en:end -->

## 与精确伪逆的共同边界
<!-- bilingual-en:start -->
*Their shared boundary with the exact pseudoinverse*
<!-- bilingual-en:end -->

若 TSVD 没有截掉任何正奇异值，它与原矩阵的精确伪逆相同；一旦删去一个正奇异方向，算子就已改变。对非零矩阵采用 $\lambda>0$ 时，岭算子会收缩每个正奇异方向，因此也不同于原精确伪逆。二者都是以改变反演算子换取稳定性，而不是“更准确地计算同一个 $A^+$”。在明确的随机噪声模型下，这种改变可表现为偏差—方差取舍；选择参数仍需依据噪声尺度、恢复目标和验证设计。
<!-- bilingual-en:start -->
If TSVD removes no positive singular value, it equals the exact pseudoinverse of the original matrix; once it removes one, the operator has changed. For a nonzero matrix, ridge with $\lambda>0$ shrinks every positive singular direction and therefore also differs from the exact pseudoinverse. Both methods trade a change in the inverse operator for greater stability rather than computing the same $A^+$ more accurately. Under an explicit stochastic noise model, this change can appear as a bias-variance trade-off; parameter choice must still reflect the noise scale, recovery objective, and validation design.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若一个小但正的奇异方向仍需保留一部分信息，TSVD 与岭会怎样处理它？
> <!-- bilingual-en:start -->
> If a small but positive singular direction still contains information worth retaining, how do TSVD and ridge treat it differently?
> <!-- bilingual-en:end -->
>
> **答案：** TSVD 只能按阈值把该方向完整反演或完全删除；岭则保留该方向但连续收缩其系数。两者都可能更稳定，却施加了不同的结构选择。
> <!-- bilingual-en:start -->
> **Answer:** TSVD either inverts the direction fully or removes it completely according to the threshold. Ridge retains the direction but continuously shrinks its coefficient. Both may improve stability, but they impose different structural choices.
> <!-- bilingual-en:end -->

## 来源与核验

- P. C. Hansen, [*Intro to Inverse Problems*, Chapter 4: Regularization Methods](https://www2.imm.dtu.dk/~pcha/DIP/chap4.pdf#page=16)：核验截断 SVD 与 Tikhonov 的滤波因子、硬截断与连续收缩的区别，以及稳定性取舍；该讲义使用 $\lambda^2$ 参数化。
- [LAPACK `DGELSD`](https://www.netlib.org/lapack/explore-html/d9/d67/group__gelsd_ga0bee7e1b9e7e43f59ecf2419b2759c42.html)：核验按 `RCOND` 将小奇异值判零并据此形成有效秩的 TSVD/SVD 求解口径。
- [[01_Math/02_linear algebra/03_Positive Definite Matrices and Applications.md#3.8.8 边界、反例与易错点|课程 3.8.8]]：核对“正则化改变原精确反演算子而不是更准确计算同一伪逆”的课程边界。
<!-- bilingual-en:start -->
- Hansen, *Intro to Inverse Problems*, Chapter 4, supports the filter factors for truncated SVD and Tikhonov regularization, the distinction between hard truncation and continuous shrinkage, and the associated stability trade-off; the notes use a $\lambda^2$ parameterization.
- LAPACK `DGELSD` verifies the SVD-based convention that uses `RCOND` to classify small singular values as zero and thereby select an effective rank.
- Course Section 3.8.8 checks the boundary that regularization changes the original inverse operator rather than computing the same pseudoinverse more accurately.
<!-- bilingual-en:end -->
