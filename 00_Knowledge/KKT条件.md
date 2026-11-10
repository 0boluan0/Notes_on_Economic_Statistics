---
aliases:
- KKT conditions
- Karush–Kuhn–Tucker conditions
- Karush-Kuhn-Tucker conditions
student_os: knowledge-atom
atom_id: OPT-MV-005
atom_set: multivariable-optimization
atom_type: definition
status: source-checked
requires:
- '[[全微分]]'
- '[[Jacobian矩阵]]'
- '[[Lagrange必要条件]]'
related:
- '[[LICQ条件]]'
- '[[Slater条件]]'
leads_to:
- '[[互补松弛]]'
- '[[KKT必要性]]'
- '[[KKT充分性]]'
part_of:
- '[[多元优化.canvas|多元优化]]'
---

<!-- bilingual-en:start -->
# KKT conditions
<!-- bilingual-en:end -->

*KKT 条件*

<!-- bilingual-en:start -->
The **Karush–Kuhn–Tucker (KKT) conditions** test a proposed input together with coefficients attached to its constraints. At a constrained optimum, the objective may still increase in a direction that the constraints forbid. KKT asks whether the objective's first-order change can be balanced by the constraint gradients, while the input remains feasible and the inequality coefficients have the right signs. The balance equation and the feasibility checks form one system.
<!-- bilingual-en:end -->

**Karush–Kuhn–Tucker（KKT）条件**一起检查候选输入以及配给各条约束的系数。约束最优点处，目标仍可能沿某个被约束禁止的方向上升。KKT 要求目标的一阶变化能由约束梯度平衡，同时输入确实可行、不等式系数的符号正确。平衡方程与可行性检查共同组成一个系统。

<!-- bilingual-en:start -->
Here the variables form a vector $x\in U$, where $U\subseteq\mathbb R^n$ is open and $n\ge1$. The real-valued objective $f$ is to be maximized. There are finitely many inequality functions $g_1,\ldots,g_m$ and equality functions $h_1,\ldots,h_p$, with $m,p\ge0$. An inequality permits $g_i(x)\le0$; an equality requires $h_j(x)=0$. The feasible set consists of the inputs satisfying all these requirements. For the differentiable KKT system below, these functions are [[全微分|differentiable]] at the candidate point.
<!-- bilingual-en:end -->

输入是向量 $x\in U$，其中 $U\subseteq\mathbb R^n$ 为开集、$n\ge1$；实值目标函数 $f$ 要被最大化。不等式函数有有限个 $g_1,\ldots,g_m$，等式函数有有限个 $h_1,\ldots,h_p$，允许 $m,p\ge0$。不等式允许 $g_i(x)\le0$，等式要求 $h_j(x)=0$；同时满足全部要求的输入构成可行集。下面的可微 KKT 系统要求这些函数在候选点[[全微分|可微]]。

<!-- bilingual-en:start -->
Write $g=(g_1,\ldots,g_m)^T$ and $h=(h_1,\ldots,h_p)^T$ as columns; the superscript $T$ denotes [[矩阵转置|transpose]]. The multiplier vectors $\lambda\in\mathbb R^m$ and $\nu\in\mathbb R^p$ supply one real coefficient per inequality and equality. They are auxiliary unknowns in the test, not extra choices in the original objective. The **Lagrangian** $\mathcal L$ combines the objective and these weighted constraints; for example, $\lambda^Tg=\sum_i\lambda_i g_i$. Fixing the following signs is essential.
<!-- bilingual-en:end -->

把函数值排成列 $g=(g_1,\ldots,g_m)^T$、$h=(h_1,\ldots,h_p)^T$，其中上标 $T$ 表示[[矩阵转置|转置]]。乘子向量 $\lambda\in\mathbb R^m$、$\nu\in\mathbb R^p$ 分别为每条不等式、每条等式提供一个实系数；它们是检查系统中的辅助未知量，不是原目标里新增加的选择。**Lagrangian（拉格朗日函数）** $\mathcal L$ 把目标和加权约束组合起来，例如 $\lambda^Tg=\sum_i\lambda_i g_i$。必须先固定下面的符号约定。

> [!summary] The joint system · 条件的定义
> <!-- bilingual-en:start -->
> For the stated maximisation problem and sign convention, the KKT conditions are the following four classes imposed jointly on a candidate triple. These equations define the system. [[KKT必要性|KKT necessity]] states when a local optimum must enter it, while [[KKT充分性|KKT sufficiency]] states when the system certifies global optimality.
> <!-- bilingual-en:end -->
>
> 考虑
> $$
> \max_x f(x)\quad\text{s.t.}\quad h(x)=0,\qquad g(x)\le0,
> $$
> 并统一取
> $$
> \mathcal L(x,\lambda,\nu)=f(x)-\lambda^Tg(x)-\nu^Th(x).
> $$
> 对一个候选三元组 $(x^*,\lambda^*,\nu^*)$，Karush–Kuhn–Tucker（KKT）条件就是下面四类等式与不等式同时成立。它先定义一个候选系统；局部最优何时必须进入这个系统由 [[KKT必要性]] 负责，系统何时足以证明全局最优由 [[KKT充分性]] 负责。

<!-- bilingual-en:start -->
## The four classes of conditions
<!-- bilingual-en:end -->

## 四类条件

<!-- bilingual-en:start -->
**Primal feasibility:** $h(x^*)=0$ and $g(x^*)\le0$. The candidate must actually belong to the feasible set.
<!-- bilingual-en:end -->

**原问题可行性：**
$$
h(x^*)=0,
\qquad g(x^*)\le0.
$$
候选点必须真的在可行域内。

<!-- bilingual-en:start -->
**Dual feasibility:** $\lambda^*\ge0$. This sign belongs to the chosen convention: maximisation, $g\le0$, and $\mathcal L=f-\lambda^Tg-\nu^Th$. Equality multipliers $\nu$ are unrestricted in sign.
<!-- bilingual-en:end -->

**对偶可行性：**
$$
\lambda^*\ge0.
$$
这个符号与“最大化、$g\le0$、$\mathcal L=f-\lambda^Tg-\nu^Th$”的统一约定配套。等式乘子 $\nu$ 没有符号限制。

<!-- bilingual-en:start -->
**Stationarity:** $\nabla f(x^*)-Dg(x^*)^T\lambda^*-Dh(x^*)^T\nu^*=0$. The objective's first-order improvement is balanced by normals to the active constraints.
<!-- bilingual-en:end -->

**驻点：**
$$
\nabla f(x^*)-Dg(x^*)^T\lambda^*-Dh(x^*)^T\nu^*=0.
$$
目标的一阶改善方向被活动约束的法向量抵消。

<!-- bilingual-en:start -->
The [[全微分#实值函数、梯度与全微分|gradient]] $\nabla f$ is the column of first partial derivatives. Here $Dg$ and $Dh$ are [[Jacobian矩阵|Jacobian matrices]], with one row per constraint, of sizes $m\times n$ and $p\times n$. Thus the stationarity equation says $\nabla f(x^*)=\sum_i\lambda_i^*\nabla g_i(x^*)+\sum_j\nu_j^*\nabla h_j(x^*)$. It is the derivative with respect to $x$, holding the multipliers fixed. The constraint gradients are the vectors used in this balance; the definition does not assume that they are nonzero or independent.
<!-- bilingual-en:end -->

[[全微分#实值函数、梯度与全微分|梯度]] $\nabla f$ 是一阶偏导组成的列。这里 $Dg,Dh$ 是[[Jacobian矩阵]]，每行对应一条约束，尺寸分别为 $m\times n$、$p\times n$。因此驻点方程就是说 $\nabla f(x^*)=\sum_i\lambda_i^*\nabla g_i(x^*)+\sum_j\nu_j^*\nabla h_j(x^*)$；求导时改变的是 $x$，乘子保持固定。约束梯度是参与这种平衡的向量；定义本身并不假定它们非零或线性无关。

<!-- bilingual-en:start -->
**Complementary slackness:** $\lambda_i^*g_i(x^*)=0$ for every inequality. [[互补松弛|The complementary-slackness explanation]] shows how to read this definition and which converses fail.
<!-- bilingual-en:end -->

**互补松弛：**
$$
\lambda_i^*g_i(x^*)=0\qquad(i=1,\ldots,m).
$$
每条不等式的乘子与约束值乘积必须为零。怎样读取这项定义，以及哪些反向推论不成立，由 [[互补松弛]] 展开。

<!-- bilingual-en:start -->
An inequality is **active** or **binding** at a feasible $x^*$ when $g_i(x^*)=0$, and **slack** when $g_i(x^*)<0$. Its status is determined by the constraint value. Complementarity forces every slack inequality's coefficient to be zero, so only active inequalities can contribute to stationarity. An active inequality can still have a zero coefficient. All equality constraints remain in the system, including those with $\nu_j^*=0$.
<!-- bilingual-en:end -->

在可行点 $x^*$，若 $g_i(x^*)=0$，称该不等式**活动**或**绑定**；若 $g_i(x^*)<0$，称它**松弛**。活动状态由约束值决定。互补松弛使每条松弛不等式的系数都为零，因此只有活动不等式可能参与驻点平衡；活动不等式的系数仍可为零。所有等式约束都留在系统中，包括那些 $\nu_j^*=0$ 的等式。

<!-- bilingual-en:start -->
## Substitute a complete candidate
<!-- bilingual-en:end -->

## 把完整候选代回系统

<!-- bilingual-en:start -->
Consider two inputs $x,y$ and the following problem. The inequalities restrict $x$ to $[0,1]$, while the equality fixes $y$ at zero.
<!-- bilingual-en:end -->

考虑两个输入 $x,y$ 以及下面的问题。不等式把 $x$ 限制在 $[0,1]$，等式把 $y$ 固定为零。

$$
\begin{gathered}
\max_{(x,y)\in\mathbb R^2} f(x,y)=-(x-2)^2-(y-1)^2,\\
g_1(x,y)=x-1\le0,\qquad g_2(x,y)=-x\le0,\\
h(x,y)=y=0.
\end{gathered}
$$

<!-- bilingual-en:start -->
At $(x^*,y^*)=(1,0)$, choose $\lambda^*=(2,0)^T$ and $\nu^*=2$. Feasibility gives $g_1=0$, $g_2=-1$, and $h=0$; both inequality coefficients are nonnegative. The gradients and products check the remaining conditions directly:
<!-- bilingual-en:end -->

在 $(x^*,y^*)=(1,0)$，取 $\lambda^*=(2,0)^T$、$\nu^*=2$。可行性检查给出 $g_1=0$、$g_2=-1$、$h=0$；两个不等式系数均非负。其余条件可以直接代入梯度和乘积检查：

$$
\begin{gathered}
\nabla f(1,0)=\begin{pmatrix}2\\2\end{pmatrix},\quad
\nabla g_1=\begin{pmatrix}1\\0\end{pmatrix},\\
\nabla g_2=\begin{pmatrix}-1\\0\end{pmatrix},\quad
\nabla h=\begin{pmatrix}0\\1\end{pmatrix},\\
\begin{pmatrix}2\\2\end{pmatrix}
-2\begin{pmatrix}1\\0\end{pmatrix}
-0\begin{pmatrix}-1\\0\end{pmatrix}
-2\begin{pmatrix}0\\1\end{pmatrix}
=\begin{pmatrix}0\\0\end{pmatrix},\\
\lambda_1^*g_1(1,0)=2(0)=0,\qquad
\lambda_2^*g_2(1,0)=0(-1)=0.
\end{gathered}
$$

<!-- bilingual-en:start -->
The active upper bound supplies the horizontal coefficient $2$; the slack lower bound has coefficient zero. The equality coefficient supplies the vertical balance. We can also verify optimality without relying on the system: every feasible input has $0\le x\le1$ and $y=0$, so $(x-2)^2\ge1$ and $f(x,0)\le-2=f(1,0)$. Equality requires $x=1$.
<!-- bilingual-en:end -->

活动的上界提供水平系数 $2$，松弛的下界系数为零；等式乘子提供竖直方向的平衡。还可以不依赖方程组而直接核实最优性：每个可行输入都满足 $0\le x\le1$、$y=0$，因此 $(x-2)^2\ge1$，从而 $f(x,0)\le-2=f(1,0)$，等号要求 $x=1$。

<!-- bilingual-en:start -->
By contrast, $(0,0)$ with $\lambda=(0,-4)^T$ and $\nu=2$ satisfies stationarity and both complementary products, and the input is feasible. It fails the multiplier-sign condition. Replacing the equality $h=y$ by the equivalent equality $\widetilde h=-y$ would change the correct equality coefficient to $-2$; no sign restriction is imposed on that coefficient.
<!-- bilingual-en:end -->

相反，在 $(0,0)$ 取 $\lambda=(0,-4)^T$、$\nu=2$，驻点方程和两个互补乘积都成立，输入也可行，却违反了乘子符号要求。若把等式 $h=y$ 改写成等价的 $\widetilde h=-y$，正确等式乘子会变成 $-2$；该系数本来就没有符号限制。

<!-- bilingual-en:start -->
## Order of use and logical boundary
<!-- bilingual-en:end -->

## 使用顺序与逻辑边界

<!-- bilingual-en:start -->
State the objective direction, inequality direction, and Lagrangian sign convention before checking all four classes. Treating every inequality as an equality discards interior solutions. Solving stationarity without checking $g\le0$ and $\lambda\ge0$ leaves infeasible or sign-inconsistent false candidates.
<!-- bilingual-en:end -->

先写清目标方向、约束方向和 Lagrangian 号约定，再逐条检查四类条件。把所有不等式预先当成等式会丢掉内部解；只解驻点方程而不检查 $g\le0$ 与 $\lambda\ge0$ 会留下不可行或符号错误的伪解。

<!-- bilingual-en:start -->
Solving the KKT system produces a candidate triple. [[KKT必要性|Necessity]] needs differentiability and a suitable constraint qualification; [[KKT充分性|sufficiency]] needs convex structure. Those theorem assumptions are not part of the definition itself.
<!-- bilingual-en:end -->

解出 KKT 系统只得到候选三元组。[[KKT必要性]] 需要可微性与适当约束资格；[[KKT充分性]] 需要凹目标、凸不等式与仿射等式等结构。不要把这些定理的假设塞回 KKT 的定义，也不要只解驻点方程就忽略另外三类条件。

<!-- bilingual-en:start -->
More precisely, $C^1$ data near a local maximizer and [[LICQ条件|linearly independent active inequality gradients together with all equality gradients]] are one sufficient route to necessary KKT conditions. LICQ is a qualification for that theorem, not a fifth KKT equation. For global sufficiency, a differentiable [[凹凸性|concave]] objective, convex inequality functions, and affine equalities on a common open [[凸集|convex domain]] make every feasible KKT triple certify a global maximum. This direction does not require Slater's condition once the triple is supplied. The linked necessity and sufficiency atoms give the respective proofs and boundaries.
<!-- bilingual-en:end -->

更准确地说，局部最大点附近的数据为 $C^1$，再加上[[LICQ条件|活动不等式梯度与全部等式梯度线性无关]]，是保证必要 KKT 条件的一条充分路线。LICQ 是该定理的资格条件，不是第五条 KKT 方程。对于全局充分性，在共同的开[[凸集|凸定义域]]上，可微的[[凹凸性|凹]]目标、凸不等式函数和仿射等式，使每个可行 KKT 三元组都能认证全局最大；已经给定这个三元组时，此方向不要求 Slater 条件。相应必要性和充分性卡片分别展开证明与边界。

<!-- bilingual-en:start -->
With no inequalities, omit $\lambda$ and complementary slackness, obtaining the equality-constrained [[Lagrange必要条件|Lagrange system]]. With no constraints at all, stationarity becomes $\nabla f(x^*)=0$. Bounds such as $x_i\ge0$ are constraints too: writing them as $-x_i\le0$ ensures that their coefficients are included. The differentiable system stated here does not by itself supply a rule for discrete choices or nonsmooth objectives.
<!-- bilingual-en:end -->

没有不等式时，省去 $\lambda$ 和互补松弛，就得到等式约束的 [[Lagrange必要条件|Lagrange 系统]]；完全没有约束时，驻点退化为 $\nabla f(x^*)=0$。$x_i\ge0$ 这样的变量界限也是约束，写成 $-x_i\le0$ 才能把相应系数列入系统。这里给出的可微系统本身不直接处理离散选择或不可微目标。

> [!question]- Check your understanding · 自检
> <!-- bilingual-en:start -->
> After solving stationarity, why must the candidate and multipliers still be substituted into the other three classes of conditions?
> <!-- bilingual-en:end -->
>
> 已经解出驻点方程后，为什么仍必须把 $x^*$ 和乘子代回另外三类条件？
>
> <!-- bilingual-en:start -->
> **Answer:** Stationarity alone does not guarantee primal feasibility, correct multiplier signs, or a zero multiplier on a slack constraint. KKT obtains its force from the joint system.
> <!-- bilingual-en:end -->
>
> **答案：** 驻点方程本身不保证原问题可行、乘子符号正确或松弛约束的乘子为零；KKT 的信息来自四类条件的联立。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- The EC400 SOFP lecture notes were checked for the course convention: maximisation, $g\le0$, multiplier signs, and the four KKT classes.
- MIT 6.7220 Lecture 7 was used to cross-check the standard four-part KKT system; the necessity theorem is stated in [[KKT必要性|KKT necessity]].
- Boyd and Vandenberghe, *Convex Optimization*, §5.5 was checked for the KKT system and the logical boundary between necessity and convex sufficiency.
<!-- bilingual-en:end -->

- LSE EC400，*SOFP Lecture Notes*（课程讲义）：核对课程采用的最大化问题、$g\le0$ 方向、乘子符号与四类条件。
- [MIT 6.7220, Lecture 7: Lagrange Multipliers and KKT Conditions](https://ocw.mit.edu/courses/6-7220j-nonlinear-optimization-spring-2025/resources/mit6_7220_s25_lec07_pdf/)：交叉核对 KKT 四类条件的标准系统；必要性定理由 [[KKT必要性]] 独立承担。
- [Boyd and Vandenberghe, *Convex Optimization*, §5.5](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：核对 KKT 系统及其与凸充分性的逻辑分界。

<!-- bilingual-en:start -->
For precise locations, the [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/02_SOFP/Lectures/EC400 Lecture Notes SOFP.pdf#page=29|SOFP notes]] use the maximization convention on PDF pp.29–30. MIT Lecture 7, PDF pp.2–4, gives the system and a separate LICQ necessity theorem; pp.5–6 give convex sufficiency. Boyd and Vandenberghe, printed pp.242–244 / PDF pp.256–258, separates complementary slackness, the four KKT conditions, and convex sufficiency. Both external texts use minimization: apply them to $F=-f$, so $F+\lambda^Tg+\nu^Th=-\mathcal L$. This verifies the signs above without importing different conventions. The complete example is checked by substitution and direct comparison.
<!-- bilingual-en:end -->

精确定位：[[01_Math/08_EC400_MathsCamp/2026_Course_Materials/02_SOFP/Lectures/EC400 Lecture Notes SOFP.pdf#page=29|SOFP PDF 第29–30页]]采用最大化约定；MIT Lecture 7 PDF 第2–4页给出系统和独立的 LICQ 必要性定理，第5–6页给出凸充分性；Boyd 与 Vandenberghe 印本第242–244页／PDF 第256–258页分开讨论互补松弛、四类 KKT 条件和凸充分性。后两来源使用最小化，把它们应用于 $F=-f$ 后有 $F+\lambda^Tg+\nu^Th=-\mathcal L$，即可核对本文符号而不混用约定。完整算例通过代入与直接比较核验。

