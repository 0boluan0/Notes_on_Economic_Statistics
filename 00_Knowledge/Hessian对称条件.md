---
aliases:
  - '若二阶偏导在一点的某个邻域连续，则混合偏导可交换且 Hessian 对称；仅在该点存在二阶偏导并不足够'
  - Continuous second partial derivatives in a neighbourhood make the mixed partials interchangeable and the Hessian symmetric, whereas pointwise existence alone is insufficient
student_os: knowledge-atom
atom_id: CALC-MV-016
atom_set: multivariable-differentiation
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Hessian矩阵]]"
related:
  - "[[实对称矩阵]]"
  - "[[二次型对称化]]"
leads_to:
  - "[[多元Taylor近似]]"
part_of:
  - "[[多元微分.canvas|多元微分]]"
---

# 若二阶偏导在一点的某个邻域连续，则混合偏导可交换且 Hessian 对称；仅在该点存在二阶偏导并不足够
<!-- bilingual-en:start -->
*Continuous second partial derivatives in a neighbourhood make the mixed partials interchangeable and the Hessian symmetric, whereas pointwise existence alone is insufficient*
<!-- bilingual-en:end -->

> [!summary] 对称性是一条带正则条件的定理
> 若实值函数 $f$ 的二阶偏导在 $a$ 的某个邻域存在且连续，则 Young/Clairaut 定理给出
> $$
> \frac{\partial^2f}{\partial x_i\partial x_j}(a)
> =
> \frac{\partial^2f}{\partial x_j\partial x_i}(a).
> $$
> 因而 [[Hessian矩阵|Hessian]] 在 $a$ 是 [[实对称矩阵]]。对称性不是 Hessian 记号自动附带的性质，而是这些正则条件推出的结论。
>
> <!-- bilingual-en:start -->
> If the second partial derivatives exist and are continuous in a neighbourhood of $a$, Young's or Clairaut's theorem makes the mixed partials equal at $a$, so the Hessian is symmetric. Symmetry is a theorem under regularity, not part of the notation by definition.
> <!-- bilingual-en:end -->

## 为什么“在一点存在”不够
<!-- bilingual-en:start -->
*Why existence at one point is insufficient*
<!-- bilingual-en:end -->

令
$$
f(x,y)=
\begin{cases}
\dfrac{xy(x^2-y^2)}{x^2+y^2},&(x,y)\ne(0,0),\\[4pt]
0,&(x,y)=(0,0).
\end{cases}
$$
沿坐标轴先求一次偏导，可得
$$
f_x(0,y)=-y,
\qquad
f_y(x,0)=x.
$$
于是原点的两个混合偏导都存在，却满足
$$
f_{xy}(0,0)=-1,
\qquad
f_{yx}(0,0)=1.
$$
因此，仅知道一点处二阶偏导存在，不能交换求导次序，也不能无条件把该点的 Hessian 当成对称矩阵。

<!-- bilingual-en:start -->
For the displayed function, differentiating first along one coordinate axis gives $f_x(0,y)=-y$ and $f_y(x,0)=x$. Both mixed partial derivatives therefore exist at the origin, but they are $-1$ and $1$. Pointwise existence alone does not permit the order of differentiation to be exchanged.
<!-- bilingual-en:end -->

## 它和二次型对称化不是同一件事
<!-- bilingual-en:start -->
*This is different from symmetrising a quadratic form*
<!-- bilingual-en:end -->

任意方阵 $A$ 的二次型都满足
$$
h^TAh=h^T\frac{A+A^T}{2}h,
$$
因为反对称部分对二次型没有贡献。这是 [[二次型对称化]] 的代数恒等式；它不能证明原矩阵 $A$ 本身对称，也不能替代混合偏导可交换所需的分析条件。

<!-- bilingual-en:start -->
Every quadratic form depends only on the symmetric part of its matrix, but that algebraic identity does not prove that the original Hessian entries are symmetric. Equality of mixed partial derivatives still requires an analytical regularity theorem.
<!-- bilingual-en:end -->

> [!question]- 自检
> 已知 $f_{xy}(a)$ 与 $f_{yx}(a)$ 都存在，能否直接断言二者相等？还需要什么类型的信息？
>
> <!-- bilingual-en:start -->
> If $f_{xy}(a)$ and $f_{yx}(a)$ both exist, may we immediately conclude that they are equal? What kind of additional information is needed?
> <!-- bilingual-en:end -->
>
> **答案：** 不能。一个常用的充分条件是二阶偏导在 $a$ 的某个邻域存在且连续；仅有一点处存在会被上面的反例推翻。
>
> <!-- bilingual-en:start -->
> **Answer:** No. A standard sufficient condition is that the second partial derivatives exist and are continuous in a neighbourhood of $a$. The counterexample above shows that pointwise existence alone is insufficient.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- LSE EC400，*Revision Maths Notes 7: Working with Multivariate Calculus*（课程讲义） §7.7：直接核对 Young 定理、混合偏导可交换的连续性条件与 Hessian 对称结论。
- [MIT 18.02SC, Partial Derivatives](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/unit-2-derivatives-of-multivariable-functions/part-a-functions-of-two-variables-tangent-approximation-and-optimization/session-27-partial-derivatives/)：交叉核对混合偏导相等需要连续性假设，而不是只靠记号。

<!-- bilingual-en:start -->
- EC400 Revision Maths Notes 7, §7.7, was checked directly for Young's theorem, the continuity condition for interchanging mixed partials, and the resulting Hessian symmetry.
- MIT 18.02SC was used to cross-check that equality of mixed partial derivatives depends on a continuity assumption rather than notation alone.
<!-- bilingual-en:end -->
