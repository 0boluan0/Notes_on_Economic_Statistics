---
aliases:
- Slater's condition
- Slater condition
- Slater 约束资格
student_os: knowledge-atom
atom_id: OPT-MV-008
atom_set: multivariable-optimization
atom_type: definition
status: source-checked
related:
- '[[凸优化全局性]]'
- '[[LICQ条件]]'
leads_to:
- '[[Slater强对偶]]'
- '[[Slater不等于LICQ]]'
part_of:
- '[[多元优化.canvas|多元优化]]'
---

<!-- bilingual-en:start -->
# Slater's condition
<!-- bilingual-en:end -->

*Slater 条件*

<!-- bilingual-en:start -->
Slater's condition asks whether the constraints leave room somewhere: can one point satisfy all required inequalities with a genuine margin while still obeying every equality exactly? Such a point is a **strictly feasible point**. It is a witness that the restrictions are not squeezed shut, not a proposed optimum. The same point must pass every test simultaneously.
<!-- bilingual-en:end -->

Slater 条件问的是约束是否在某处留有余地：能不能找到**同一个点**，让需要严格成立的所有不等式都留出余量，同时精确满足每个等式？这个点叫**严格可行点**，它证明限制没有被完全挤紧，不是在猜最优解。不能为每条约束分别找一个不同的点来替代它。

<!-- bilingual-en:start -->
The standard setting is a convex optimization problem. We use maximization, so the objective $f$ is [[凹凸性|concave]], each inequality function $g_i$ is convex, and equality constraints are affine. A [[凸集|convex set]] contains the whole segment between any two of its points. These hypotheses make the feasible region convex and are the setting in which Slater gives the duality conclusion explained in [[Slater强对偶|strong duality under Slater]].
<!-- bilingual-en:end -->

标准背景是凸优化问题。本卡采用最大化记号，因此目标 $f$ 是[[凹凸性|凹函数]]，不等式函数 $g_i$ 是凸函数，等式约束是仿射的。[[凸集]]包含任意两点之间的整条线段；上述条件使可行区域保持凸性，也给出 [[Slater强对偶]] 所用的适用范围。

<!-- bilingual-en:start -->
## First identify the common domain
<!-- bilingual-en:end -->

## 先确定共同定义域

<!-- bilingual-en:start -->
Let $x\in\mathbb R^n$ be the choice vector, $g_1,\ldots,g_m$ the finitely many inequality functions, and $A\in\mathbb R^{p\times n}$ and $b\in\mathbb R^p$ the equality data. The **common domain** $D$ consists of points where the objective and every inequality function are finite. Assume $D$ is nonempty and convex; the functions are concave or convex on their domains as stated above. First define $D$, then impose the displayed constraints to obtain the feasible set $F$.
<!-- bilingual-en:end -->

设 $x\in\mathbb R^n$ 是选择向量，$g_1,\ldots,g_m$ 是有限个不等式函数，$A\in\mathbb R^{p\times n}$、$b\in\mathbb R^p$ 是等式的数据。**共同定义域** $D$ 包含目标和所有不等式函数都取有限值的点；假设它非空且凸，各函数在其定义域上具有前述凹凸性。先确定 $D$，再施加写出的约束，才得到可行集 $F$。

$$
D=\operatorname{dom}f\cap\bigcap_{i=1}^m\operatorname{dom}g_i,\qquad F=\{x\in D:g_i(x)\le0\ (i=1,\ldots,m),\ Ax=b\}.
$$

<!-- bilingual-en:start -->
The [[相对内部|relative interior]] of $D$ means its interior measured inside its own [[仿射包|affine hull]], the smallest affine set containing it. For example, a line segment in the plane contains no two-dimensional open disk; nevertheless, it has interior points when we move along its line. For $D=\{(t,0):0\le t\le1\}$, the affine hull is the entire horizontal axis and $\operatorname{ri}D=\{(t,0):0<t<1\}$. In Slater's test, take the relative interior of the **common domain**, not of the already constrained set $F$. If $D$ is open in $\mathbb R^n$, then $\operatorname{ri}D=D$.
<!-- bilingual-en:end -->

$D$ 的[[相对内部]]是沿它自身的[[仿射包]]来衡量的内部；仿射包是包含它的最小仿射集合。例如，平面中的线段容不下任何二维小圆盘，但沿着它所在的直线，仍有内部点。对 $D=\{(t,0):0\le t\le1\}$，仿射包是整个水平轴，$\operatorname{ri}D=\{(t,0):0<t<1\}$。Slater 检查的是**共同定义域**的相对内部，不是已经加完约束的可行集 $F$ 的相对内部。若 $D$ 在 $\mathbb R^n$ 中是开集，则 $\operatorname{ri}D=D$。

> [!important] The basic version · 基本版本
> <!-- bilingual-en:start -->
> Consider maximising a concave $f$ subject to convex inequalities $g_i(x)\le0$ and affine equalities $Ax=b$. Slater's condition holds if some $\bar x$ in the relative interior of the domain satisfies the applicable inequalities strictly and all equalities exactly. Its strong-duality consequence is stated in [[Slater强对偶|Slater strong duality]].
>
> Here the basic version requires **every** inequality to be strict at one $\bar x\in\operatorname{ri}D$.
> <!-- bilingual-en:end -->
>
> 考虑凹最大化问题
> $$
> \max_x f(x)\quad\text{s.t.}\quad g_i(x)\le0,\qquad Ax=b,
> $$
> 其中 $f$ 凹、每个 $g_i$ 凸，等式约束为仿射。若存在定义域相对内部中的 $\bar x$ 满足
> $$
> g_i(\bar x)<0\quad(\forall i),
> \qquad A\bar x=b,
> $$
> 就称满足 Slater 条件。这里定义的是严格可行性；由它导出强对偶和 KKT 必要性的条件见 [[Slater强对偶]]。

<!-- bilingual-en:start -->
For example, impose $x_1^2+x_2^2\le1$ and $x_1+x_2=1$, with all functions defined on $\mathbb R^2$. The point $(1/2,1/2)$ satisfies the equality and gives $g=1/4+1/4-1=-1/2<0$. It is a Slater witness. The point $(1,0)$ satisfies the same equality but gives $g=0$, so it is feasible without being a strict witness. No objective needs to be optimized to make this check.
<!-- bilingual-en:end -->

例如，约束是 $x_1^2+x_2^2\le1$ 与 $x_1+x_2=1$，各函数定义在 $\mathbb R^2$ 上。点 $(1/2,1/2)$ 满足等式，且 $g=1/4+1/4-1=-1/2<0$，所以它是 Slater 见证点。点 $(1,0)$ 满足同一等式，却有 $g=0$，只是可行点，不是严格可行见证。作这项检查无需先优化目标。

<!-- bilingual-en:start -->
## Why equality constraints must be affine
<!-- bilingual-en:end -->

## 为什么等式必须是仿射的

<!-- bilingual-en:start -->
Standard convex optimisation requires a convex feasible set. A convex inequality $g_i\le0$ gives a convex sublevel set, and $Ax=b$ gives an affine set. In contrast, setting a general convex function equal to zero need not give a convex set: $x^2-1=0$ has feasible set $\{-1,1\}$. Hence “convex equality functions” is not the convex-programming condition; equalities must be affine.
<!-- bilingual-en:end -->

标准凸优化要求可行集保持凸。$g_i\le0$ 在 $g_i$ 凸时给凸次水平集；$Ax=b$ 给仿射集。一般的“凸函数等于零”却不一定给凸集，例如 $h(x)=x^2-1=0$ 的可行集是 $\{-1,1\}$。因此不能把“等式函数也凸”当作凸规划条件。

<!-- bilingual-en:start -->
## The refined version: affine inequalities may be tight
<!-- bilingual-en:end -->

## 精细版本：仿射不等式允许取等号

<!-- bilingual-en:start -->
For affine inequalities, Slater can be weakened using relative interior; in practice one often requires strict feasibility only for the nonaffine convex inequalities. Whichever version is used must be stated explicitly. Slater must not be assumed when no appropriate strictly feasible point exists.
<!-- bilingual-en:end -->

只含仿射不等式时，Slater 可作相对内部的弱化版本；实务上常只要求非仿射凸不等式严格成立。无论采用哪个版本，都应把版本写清楚，不能在没有严格可行点时默认为 Slater 已成立。

<!-- bilingual-en:start -->
More precisely, let $I_{\rm aff}$ index the affine inequalities. The **refined Slater condition** still asks for one point in $\operatorname{ri}D$ satisfying every equality and every inequality. Only the demand for a strict inequality is waived for indices in $I_{\rm aff}$; those constraints still have to hold weakly. Thus a refined witness need not be strictly feasible for every displayed inequality.
<!-- bilingual-en:end -->

准确地说，令 $I_{\rm aff}$ 表示仿射不等式的索引。**精细 Slater 条件**仍要求同一个点位于 $\operatorname{ri}D$，并满足每条等式及每条不等式；它仅免除仿射不等式的严格性要求，这些约束依然必须以 $\le0$ 成立。因此，精细版本的见证点不一定严格满足所有写出的不等式。

$$
\bar x\in\operatorname{ri}D,\quad A\bar x=b,\quad g_i(\bar x)\le0\ (i\in I_{\rm aff}),\quad g_i(\bar x)<0\ (i\notin I_{\rm aff}).
$$

> [!example] A witness for the refined version · 精细版本的见证点
> <!-- bilingual-en:start -->
> Take $D=\mathbb R$ and constraints $x\le0$, $-x\le0$, and $x^2-1\le0$. The first two force $x=0$ and can never both be strict. At zero, however, both affine inequalities hold weakly and the nonaffine one gives $-1<0$. The refined condition holds even though the basic all-strict version fails.
> <!-- bilingual-en:end -->
>
> 取 $D=\mathbb R$，约束为 $x\le0$、$-x\le0$ 和 $x^2-1\le0$。前两条迫使 $x=0$，不可能同时严格成立。但在零点，两条仿射不等式以等号成立，非仿射不等式给出 $-1<0$。因此精细版本成立，而全部严格的基本版本失败。

<!-- bilingual-en:start -->
Do not drop the domain test just because every displayed constraint is affine. If the objective is $\sqrt{x}$, its domain is $D=[0,\infty)$; adding $x\le0$ leaves only zero feasible. Zero is outside $\operatorname{ri}D=(0,\infty)$, so this is not a refined Slater witness. When all inequalities are affine **and the common domain is open**, any feasible point is in $\operatorname{ri}D$, and the refined test reduces to feasibility.
<!-- bilingual-en:end -->

不能因为写出的约束全是仿射，就省去定义域检查。例如目标为 $\sqrt{x}$，定义域 $D=[0,\infty)$；再加 $x\le0$，唯一可行点是零，但零不在 $\operatorname{ri}D=(0,\infty)$ 中，不能充当精细 Slater 见证点。只有当不等式全为仿射，且**共同定义域为开集**时，任一可行点自动落在 $\operatorname{ri}D$，精细检查才退化为普通可行性。

<!-- bilingual-en:start -->
## How to verify a proposed strictly feasible point
<!-- bilingual-en:end -->

## 怎样验证一个候选严格可行点

<!-- bilingual-en:start -->
For a proposed $\bar x$, check membership in the relative interior of the common domain, strict satisfaction of each applicable nonaffine convex inequality, and exact satisfaction of every affine equality. The point need not be optimal and no multipliers are required to identify it.
<!-- bilingual-en:end -->

给定 $\bar x$ 后，逐条检查：它位于目标与约束函数共同定义域的相对内部；每个适用的非仿射凸不等式严格小于零；每条仿射等式精确成立。$\bar x$ 不必是最优点，也无需先求出乘子。

<!-- bilingual-en:start -->
Complete this checklist by checking the affine inequalities too: strict in the basic version, weak in the refined version. A failed proposed witness says only that this point did not pass; to establish failure of the condition, show that no appropriate witness exists.
<!-- bilingual-en:end -->

最后还要补全仿射不等式的检查：基本版本要求严格成立，精细版本允许等号。某个候选点不通过，只能说明这个点不合格；要证明整个条件失败，须说明任何合适的见证点都不存在。

<!-- bilingual-en:start -->
## Boundary with neighbouring results
<!-- bilingual-en:end -->

## 与相邻结论的边界

<!-- bilingual-en:start -->
Slater's condition itself tests strict feasibility. [[Slater强对偶|The strong-duality result]] states the additional assumptions under which it guarantees strong duality and multiplier existence, while [[Slater不等于LICQ|the Slater-versus-LICQ distinction]] compares the two qualifications' different objects of inspection.
<!-- bilingual-en:end -->

Slater 条件本身只判断严格可行性。[[Slater强对偶]] 给出它保证强对偶与乘子存在时还需要的假设；[[Slater不等于LICQ]] 则比较它与 [[LICQ条件]] 的不同检查对象。

> [!question]- Check a candidate · 自检
> <!-- bilingual-en:start -->
> For $x^2-1\le0$, can zero serve as a Slater point? What about the boundary point one?
>
> **Answer:** Zero is strictly feasible because $-1<0$. The boundary point one satisfies equality and is not a strict Slater point.
> <!-- bilingual-en:end -->
>
> 对 $x^2-1\le0$，$x=0$ 能否作为 Slater 点？边界点 $x=1$ 呢？
>
> **答案：** $x=0$ 严格满足 $-1<0$，可以；$x=1$ 只取等号，不是严格可行点。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Boyd and Vandenberghe, *Convex Optimization*, §2.1.3, p. 23, and §5.2.3, pp. 226–227](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf): relative interior, the common domain, basic Slater, and its refined affine-inequality version. The max-concave convention here is obtained by negating the book's convex minimization objective.
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/02_SOFP/Lectures/EC400 Lecture Notes SOFP.pdf|LSE EC400 SOFP notes]], PDF p. 34: course notation and strict feasibility were checked against Boyd. The notes' claim of equivalence with LICQ and their convex-equality wording are not used; equality constraints in this theorem are affine.
<!-- bilingual-en:end -->

- Boyd 与 Vandenberghe，§2.1.3（书页 23）与 §5.2.3（书页 226–227，PDF 页 240–241）：核对相对内部、共同域、基本版本及仿射不等式的精细版本；本卡将书中凸最小化的目标取负，改写为凹最大化。
- LSE EC400 *SOFP Lecture Notes*，PDF 页 34：核对课程记号与严格可行性，并以 Boyd 校正范围。讲义将 Slater 与 LICQ 等同、把一般凸等式纳入凸规划的说法未予采用；这里要求仿射等式。

