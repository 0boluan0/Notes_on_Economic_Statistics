---
aliases:
- Slater and LICQ
- Slater 与 LICQ
- Slater versus LICQ
student_os: knowledge-atom
atom_id: OPT-MV-023
atom_set: multivariable-optimization
atom_type: concept-distinction
status: source-checked
requires:
- '[[Slater条件]]'
- '[[LICQ条件]]'
related:
- '[[KKT必要性]]'
part_of:
- '[[多元优化.canvas|多元优化]]'
---

<!-- bilingual-en:start -->
# Slater and LICQ
<!-- bilingual-en:end -->

*Slater 与 LICQ*

<!-- bilingual-en:start -->
Slater and LICQ both help justify KKT multipliers, but they inspect different things. [[Slater条件|Slater]] asks whether one appropriately strict feasible point exists somewhere in a convex problem. [[LICQ条件|LICQ]] asks whether the relevant constraint gradients at a specified feasible point are linearly independent. To compare them, specify both the problem class and the point: in the smooth open-domain convex setting with affine equalities, LICQ at one feasible point implies a Slater witness; the converse fails.
<!-- bilingual-en:end -->

Slater 和 LICQ 都能帮助保证 KKT 乘子存在，但检查对象不同。[[Slater条件|Slater]] 问某个凸问题中是否存在一个适当的严格可行点；[[LICQ条件|LICQ]] 问指定可行点的相关约束梯度是否线性无关。比较它们时必须说清问题范围与检查点：在共同开域上光滑、等式仿射的凸问题中，某个可行点满足 LICQ 能推出 Slater 见证点存在；反过来不成立。

<!-- bilingual-en:start -->
Slater searches a convex feasibility problem for a strictly feasible point, which need not be optimal. LICQ tests linear independence of equality and active-inequality gradients at a specified feasible point. One is global strict feasibility; the other is local first-order geometry and depends on the constraint representation.
<!-- bilingual-en:end -->

[[Slater条件]] 在整个凸可行性问题中寻找一个严格可行点，该点不必最优；它是全局的可行性条件。[[LICQ条件]] 在指定可行点检查等式梯度与活动不等式梯度是否线性无关；它是局部且依赖约束表示的一阶几何条件。

<!-- bilingual-en:start -->
An inequality $g_i(x)\le0$ is active when $g_i(x)=0$; its [[全微分#实值函数、梯度与全微分|gradient]] describes first-order change near that point. [[线性无关|Linear independence]] means that no nontrivial coefficients combine the retained gradients to zero. LICQ retains every equality gradient and every active inequality gradient, including separate indices for repeated constraints. Slater's witness must satisfy every equality exactly and the inequalities as required by the stated basic or refined version.
<!-- bilingual-en:end -->

不等式 $g_i(x)\le0$ 在 $g_i(x)=0$ 时称为活动；它的[[全微分#实值函数、梯度与全微分|梯度]]描述该点附近的一阶变化。[[线性无关]]要求这些梯度只有使用全零系数才能组合成零向量。LICQ 保留全部等式梯度及全部活动不等式梯度，重复约束也按不同索引保留。Slater 见证点则必须精确满足全部等式，并按明确采用的基本或精细版本满足各不等式。

<!-- bilingual-en:start -->
## Slater can hold while LICQ fails at a boundary point
<!-- bilingual-en:end -->

## 一个条件成立而另一个失败

<!-- bilingual-en:start -->
The duplicate constraints $x\le1$ and $2x\le2$ have a strictly feasible point, so Slater holds. At the boundary point one, both active gradients are dependent, so LICQ fails. Duplicating the constraint leaves the feasible set and Slater unchanged but destroys LICQ.
<!-- bilingual-en:end -->

约束
$$
x\le1,
\qquad
2x\le2
$$

在 $x=0$ 有严格可行点，所以 Slater 成立。在边界点 $x=1$，两条约束都活动，而梯度 $1$ 与 $2$ 线性相关，所以 LICQ 失败。重复第二条约束没有改变可行集或 Slater，却破坏了 LICQ；这已经否定两者等价。

<!-- bilingual-en:start -->
Explicitly, set $g_1(x)=x-1$ and $g_2(x)=2x-2$, with common domain $\mathbb R$. At the witness zero, their values are $-1$ and $-2$. At one, the two active gradients are $1$ and $2$, and $2\cdot1-1\cdot2=0$ is a nontrivial dependence. At zero neither inequality is active, and with no equalities the empty gradient family satisfies LICQ vacuously. Thus Slater is a property of the whole feasibility problem, while the LICQ verdict can change with the specified point.
<!-- bilingual-en:end -->

明确写成 $g_1(x)=x-1$、$g_2(x)=2x-2$，共同域为 $\mathbb R$。在见证点零，两者分别等于 $-1$、$-2$；在一点，两条活动梯度为 $1$、$2$，且 $2\cdot1-1\cdot2=0$ 是非平凡的相关关系。相反，在零点没有活动不等式，又没有等式，空梯度族自动线性无关。因此 Slater 是整个可行性问题的性质，LICQ 的结论则可以随指定点改变。

<!-- bilingual-en:start -->
## When LICQ does imply a strict Slater point
<!-- bilingual-en:end -->

## 在什么范围内 LICQ 能推出 Slater

<!-- bilingual-en:start -->
Assume a common open convex domain $U\subseteq\mathbb R^n$, finitely many continuously differentiable convex inequalities $g_i\le0$, affine equalities $Ax=b$, and a concave objective on $U$. Suppose LICQ holds at a feasible $x^*\in U$. We can construct a nearby point satisfying **all** inequalities strictly. This proves basic Slater, and therefore also its refined version, in this setting.
<!-- bilingual-en:end -->

假设共同定义域 $U\subseteq\mathbb R^n$ 为开凸集，不等式 $g_i\le0$ 有限且连续可微、凸，等式为 $Ax=b$，目标在 $U$ 上凹。设可行点 $x^*\in U$ 满足 LICQ。可以构造一个附近点，使**所有**不等式严格成立。这在上述范围内证明基本 Slater 成立，因而也满足精细版本。

<!-- bilingual-en:start -->
Let $I=\{i:g_i(x^*)=0\}$ be the active indices. Stack the rows of $A$ and the row gradients $\nabla g_i(x^*)^T$ for $i\in I$ into a matrix $M$. LICQ says these rows are independent, so $M$ has full row rank. A [[右逆与满行秩|full-row-rank matrix can solve a linear system for every right-hand side]]. We can therefore choose a direction $d$ solving the following system: no change in equalities and first-order decrease of size one in every active inequality.
<!-- bilingual-en:end -->

令 $I=\{i:g_i(x^*)=0\}$ 为活动索引。把 $A$ 的各行及每个 $i\in I$ 的行梯度 $\nabla g_i(x^*)^T$ 叠成矩阵 $M$。LICQ 表明这些行无关，所以 $M$ 满行秩；[[右逆与满行秩|满行秩保证任意右端项的线性方程都有解]]。因此可以选取方向 $d$ 满足下面的方程，让等式不变，并让每条活动不等式的一阶变化都为负一。

$$
Ad=0,\qquad\nabla g_i(x^*)^Td=-1\quad(i\in I).
$$

<!-- bilingual-en:start -->
Move to $x_\varepsilon=x^*+\varepsilon d$ with a sufficiently small $\varepsilon>0$. Openness keeps the new point in $U$, and affinity keeps $Ax_\varepsilon=b$ exactly, not just to first order. Differentiability gives the active constraints the expansion below. Here $o(\varepsilon)/\varepsilon\to0$, so the negative linear term wins for sufficiently small positive $\varepsilon$.
<!-- bilingual-en:end -->

取足够小的 $\varepsilon>0$，移动到 $x_\varepsilon=x^*+\varepsilon d$。开域保证新点仍在 $U$ 内，仿射性保证 $Ax_\varepsilon=b$ 精确成立，而不只是满足一阶近似。对活动约束，可微性给出下式；其中 $o(\varepsilon)/\varepsilon\to0$，所以充分小的正步长下，负的一阶项决定符号。

$$
g_i(x_\varepsilon)=g_i(x^*)+\varepsilon\nabla g_i(x^*)^Td+o(\varepsilon)=-\varepsilon+o(\varepsilon)<0\quad(i\in I).
$$

<!-- bilingual-en:start -->
Every inactive inequality already has $g_i(x^*)<0$ and remains negative nearby by continuity. There are only finitely many constraints, so choose a single small step satisfying all these neighborhood requirements. Then $x_\varepsilon\in U=\operatorname{ri}U$ is a Slater witness. If no inequality is active, $x^*$ itself already is a witness. The linear-algebra and small-step argument establishes this implication; convexity places it in the standard Slater problem class.
<!-- bilingual-en:end -->

每条非活动不等式原本已有 $g_i(x^*)<0$，由连续性，附近仍为负。约束只有有限条，因而能选取同一个充分小步长，同时满足所有邻域要求。这样 $x_\varepsilon\in U=\operatorname{ri}U$ 就是 Slater 见证点。若没有任何活动不等式，$x^*$ 本身已经是见证点。这个推论由线性代数和小步长论证建立；凸性确保它处于标准 Slater 问题范围。

> [!important] The implication has a stated scope · 推论有明确范围
> <!-- bilingual-en:start -->
> Within the common-open-domain, smooth convex setting with affine equalities, LICQ at a feasible point implies a strict Slater point somewhere in the problem. Slater does not imply LICQ at an arbitrary specified feasible point, as the duplicate-constraint example shows. “Not equivalent” does not mean “neither can ever imply the other.”
> <!-- bilingual-en:end -->
>
> 在共同开域、光滑凸不等式、仿射等式的范围内，可行点的 LICQ 能推出问题中存在严格 Slater 点；但 Slater 不能推出任意指定可行点都满足 LICQ，重复约束例子已经说明。不能把“不等价”读成“两个方向都绝不可能成立”。

<!-- bilingual-en:start -->
## Nonlinear equalities belong to a different setting
<!-- bilingual-en:end -->

## 非线性等式属于另一个适用范围

<!-- bilingual-en:start -->
LICQ also applies to local nonconvex problems where the standard Slater strong-duality theorem is inapplicable. The nonlinear equality $x^2+y^2=1$ is regular at $(1,0)$, yet it is not an affine-equality convex program and LICQ there is not a version of Slater.
<!-- bilingual-en:end -->

LICQ 还可以用于非凸局部问题，而标准 Slater 强对偶定理并不适用。例如非线性等式 $x^2+y^2=1$ 在 $(1,0)$ 的梯度非零，局部等式资格成立；但这不是含仿射等式的凸规划，不能把它称为 Slater 的另一种写法。

<!-- bilingual-en:start -->
For $h(x,y)=x^2+y^2-1$, the gradient at $(1,0)$ is $(2,0)$; a one-element family with a nonzero vector is independent. The circle, however, is not convex: the segment between $(1,0)$ and $(-1,0)$ passes through the infeasible origin. This example illustrates LICQ's wider applicability. It is **not** an example refuting the LICQ-to-Slater implication within the open-domain, affine-equality convex setting proved above. For nonlinear equalities, a tangent direction need not keep a straight step exactly feasible.
<!-- bilingual-en:end -->

对 $h(x,y)=x^2+y^2-1$，在 $(1,0)$ 的梯度为 $(2,0)$；仅含一个非零向量的族线性无关。但圆周不是凸集：$(1,0)$ 和 $(-1,0)$ 之间的线段经过不可行的原点。这个例子说明 LICQ 的适用范围更广，**不能**用来反驳刚证明的共同开域、仿射等式凸问题内的 LICQ 到 Slater 推论。对非线性等式，切向方向也未必保证沿直线迈步后仍精确可行。

<!-- bilingual-en:start -->
## Both routes can support KKT necessity for different reasons
<!-- bilingual-en:end -->

## 两条路线都可能给 KKT 必要性

<!-- bilingual-en:start -->
LICQ supports KKT necessity through local normal geometry, while Slater supports it in convex problems through strong duality and multiplier existence. Sharing a downstream conclusion does not make the assumptions synonymous.
<!-- bilingual-en:end -->

LICQ 通过候选点的局部法向几何支持 [[KKT必要性]]；Slater 通过 [[Slater强对偶|强对偶与乘子存在]] 支持凸问题的 KKT 必要方向。相同的下游结论不把上游假设变成同义词。

<!-- bilingual-en:start -->
For the LICQ necessity theorem, retain continuous differentiability near the point and local optimality; convexity is not required. For the Slater route, retain the convex setup and finite optimal value; to obtain ordinary gradient KKT at an actual optimum, also account for differentiability, the domain, and primal attainment as in [[Slater强对偶|the Slater duality theorem]]. Neither qualification alone asserts that a point is optimal.
<!-- bilingual-en:end -->

LICQ 必要性定理还要求该点附近连续可微以及该点局部最优，不要求凸性。Slater 路线则保留凸问题范围与有限最优值；要在实际最优点得到通常的梯度 KKT，还须按 [[Slater强对偶]] 说明可微性、定义域及原问题最优值取得的条件。两种资格本身都不宣称某个点最优。

> [!question]- What does duplication change? · 自检
> <!-- bilingual-en:start -->
> Why can duplicating an active constraint destroy LICQ without changing Slater's condition?
>
> **Answer:** Duplication creates linearly dependent active gradients, but it leaves the feasible set and every strictly feasible point unchanged.
> <!-- bilingual-en:end -->
>
> 为什么复制一条活动约束可能破坏 LICQ，却不改变 Slater？
>
> **答案：** 复制会制造线性相关的活动梯度；但可行集和严格可行点没有改变。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- Boyd and Vandenberghe, *Convex Optimization*, §5.2.3 was checked for Slater's strict-feasibility definition.
  [Book text](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), printed pp. 226–227 / PDF pp. 240–241: convex-problem scope, common-domain relative interior, and the refined affine-inequality version. These assumptions specify the shared problem class in the LICQ-to-Slater argument above.
<!-- bilingual-en:end -->

- [Boyd and Vandenberghe, *Convex Optimization*, §5.2.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：核对 Slater 的凸问题严格可行性定义。
  精确位置：书页 226–227／PDF 页 240–241，核对凸问题范围、共同域相对内部及仿射不等式精细版本；这些假设明确了正文 LICQ 到 Slater 推论所比较的共同问题范围。

<!-- bilingual-en:start -->
- MIT 6.7220 Lecture 7 was checked for LICQ as a local active-gradient condition.
  [Lecture PDF](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/mit6_7220_s25_lec07.pdf), pp. 2–5: equality and active-inequality indices, the C¹ LICQ necessity theorem and footnote 3 retaining repetitions, followed by the convex-inequality/affine-equality Slater route.
<!-- bilingual-en:end -->

- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：核对 LICQ 的局部活动梯度定义。
  精确位置：PDF 页 2–5，核对等式及活动不等式索引、C¹ 条件下的 LICQ 必要性定理及保留重复梯度的脚注 3，以及其后的凸不等式／仿射等式 Slater 路线。

<!-- bilingual-en:start -->
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/02_SOFP/Lectures/EC400 Lecture Notes SOFP.pdf|LSE EC400 SOFP notes]], PDF p. 34: course context for comparing the qualifications.
  The source's assertion that the qualifications are equivalent is corrected by the duplicate-constraint example and scoped implication above. The full-row-rank direction construction and circle example are derived explicitly here, not presented as verbatim statements from these sources.
<!-- bilingual-en:end -->

- LSE EC400 *SOFP Lecture Notes*，PDF 页 34：课程中比较两种约束资格的背景。
  讲义将两种资格等同的表述，由正文重复约束反例和有范围的正向推论校正。满行秩方向构造与圆周例子均在正文直接推导，不冒充上述来源的逐字陈述。

