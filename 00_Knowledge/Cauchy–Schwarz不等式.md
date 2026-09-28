---
student_os: knowledge-atom
atom_id: OPT-TOOLS-036
aliases:
  - "实向量内积的绝对值不超过两向量 Euclidean 长度的乘积"
  - "Cauchy–Schwarz inequality"
  - "柯西施瓦茨不等式"
status: source-checked
---

# 实向量内积的绝对值不超过两向量 Euclidean 长度的乘积

<!-- bilingual-en:start -->
*The absolute Euclidean inner product is at most the product of the vector lengths*
<!-- bilingual-en:end -->

对 $u,v\in\mathbb R^n$，Cauchy–Schwarz 不等式为
$$
|u^Tv|\le\|u\|_2\|v\|_2.
$$
若 $v\ne0$，把 $\lambda=(u^Tv)/\|v\|_2^2$ 代入非负平方 $\|u-\lambda v\|_2^2\ge0$，展开得 $\|u\|_2^2-(u^Tv)^2/\|v\|_2^2\ge0$，移项、取平方根即得。$v=0$ 时两边都为零。等号当且仅当两向量线性相关，包括其中一个为零的情况。

<!-- bilingual-en:start -->
The displayed bound applies to any real Euclidean vectors. For a nonzero second vector, minimizing the squared length of the first vector minus a scalar multiple of the second gives the inequality. A zero second vector is immediate. Equality holds exactly when the pair is linearly dependent, including the zero-vector cases.
<!-- bilingual-en:end -->

例如 $u=(1,2),v=(2,-1)$ 时内积为零，两者正交；$v=3u$ 时两者同向，达到上界。在二次型估计中取 $u=h,v=Bh$，再用[[谱范数|诱导二范数]]的 $\|Bh\|_2\le\|B\|_2\|h\|_2$，得到 $|h^TBh|\le\|B\|_2\|h\|_2^2$，可控制[[多元Taylor近似]]的余项。

<!-- bilingual-en:start -->
Orthogonal vectors have zero inner product, while proportional vectors attain the bound. Applying it to a displacement and its matrix image, then using the [[谱范数|induced Euclidean norm]], bounds a quadratic form by the matrix norm times the squared displacement length. This controls the remainder in [[多元Taylor近似|multivariable Taylor approximation]].
<!-- bilingual-en:end -->

## 来源与核验

- [Boyd–Vandenberghe, Convex Optimization, p.74 / PDF p.88](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=88)：明确写出平方形式并用于 Hessian 定号；本卡的配方证明、等号条件和两个例子直接展开核验。
- [[01_SOFP Lecture 1 - 二次型、Taylor 展开与凹凸性#5.4 为什么沿直线可以在参数一处取值|SOFP Lecture 1 的余项推导]]：说明这一不等式如何把矩阵连续性转为统一误差界。

<!-- bilingual-en:start -->
The textbook states the squared inequality in its Hessian examples. The completed-square proof, equality cases, and examples are checked directly. The linked course passage uses the inequality to convert Hessian continuity into a uniform remainder bound.
<!-- bilingual-en:end -->
