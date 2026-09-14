---
aliases:
  - 'Slater 条件要求凸问题存在一个满足仿射等式并严格满足适用凸不等式的相对内部可行点'
  - Slater's condition requires a relative-interior feasible point satisfying affine equalities and the applicable convex inequalities strictly
student_os: knowledge-atom
atom_id: OPT-MV-008
atom_set: multivariable-optimization
atom_type: definition
status: source-checked
mastery_state: unassessed
related:
  - "[[凸优化全局性]]"
  - "[[LICQ条件]]"
leads_to:
  - "[[Slater强对偶]]"
  - "[[Slater不等于LICQ]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# Slater 条件要求凸问题存在一个满足仿射等式并严格满足适用凸不等式的相对内部可行点
<!-- bilingual-en:start -->
*Slater's condition requires a relative-interior feasible point satisfying affine equalities and the applicable convex inequalities strictly*
<!-- bilingual-en:end -->

> [!summary] 它是什么
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
>
> <!-- bilingual-en:start -->
> Consider maximising a concave $f$ subject to convex inequalities $g_i(x)\le0$ and affine equalities $Ax=b$. Slater's condition holds if some $\bar x$ in the relative interior of the domain satisfies the applicable inequalities strictly and all equalities exactly. Its strong-duality consequence is stated in [[Slater强对偶|Slater strong duality]].
> <!-- bilingual-en:end -->

## 为什么等式必须是仿射的
<!-- bilingual-en:start -->
*Why equality constraints must be affine*
<!-- bilingual-en:end -->

标准凸优化要求可行集保持凸。$g_i\le0$ 在 $g_i$ 凸时给凸次水平集；$Ax=b$ 给仿射集。一般的“凸函数等于零”却不一定给凸集，例如 $h(x)=x^2-1=0$ 的可行集是 $\{-1,1\}$。因此不能把“等式函数也凸”当作凸规划条件。

<!-- bilingual-en:start -->
Standard convex optimisation requires a convex feasible set. A convex inequality $g_i\le0$ gives a convex sublevel set, and $Ax=b$ gives an affine set. In contrast, setting a general convex function equal to zero need not give a convex set: $x^2-1=0$ has feasible set $\{-1,1\}$. Hence “convex equality functions” is not the convex-programming condition; equalities must be affine.
<!-- bilingual-en:end -->

只含仿射不等式时，Slater 可作相对内部的弱化版本；实务上常只要求非仿射凸不等式严格成立。无论采用哪个版本，都应把版本写清楚，不能在没有严格可行点时默认为 Slater 已成立。

<!-- bilingual-en:start -->
For affine inequalities, Slater can be weakened using relative interior; in practice one often requires strict feasibility only for the nonaffine convex inequalities. Whichever version is used must be stated explicitly. Slater must not be assumed when no appropriate strictly feasible point exists.
<!-- bilingual-en:end -->

## 怎样验证一个候选严格可行点
<!-- bilingual-en:start -->
*How to verify a proposed strictly feasible point*
<!-- bilingual-en:end -->

给定 $\bar x$ 后，逐条检查：它位于目标与约束函数共同定义域的相对内部；每个适用的非仿射凸不等式严格小于零；每条仿射等式精确成立。$\bar x$ 不必是最优点，也无需先求出乘子。

<!-- bilingual-en:start -->
For a proposed $\bar x$, check membership in the relative interior of the common domain, strict satisfaction of each applicable nonaffine convex inequality, and exact satisfaction of every affine equality. The point need not be optimal and no multipliers are required to identify it.
<!-- bilingual-en:end -->

## 与相邻结论的边界
<!-- bilingual-en:start -->
*Boundary with neighbouring results*
<!-- bilingual-en:end -->

Slater 条件本身只判断严格可行性。[[Slater强对偶]] 给出它保证强对偶与乘子存在时还需要的假设；[[Slater不等于LICQ]] 则比较它与 [[LICQ条件]] 的不同检查对象。

<!-- bilingual-en:start -->
Slater's condition itself tests strict feasibility. [[Slater强对偶|The strong-duality result]] states the additional assumptions under which it guarantees strong duality and multiplier existence, while [[Slater不等于LICQ|the Slater-versus-LICQ distinction]] compares the two qualifications' different objects of inspection.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对 $x^2-1\le0$，$x=0$ 能否作为 Slater 点？边界点 $x=1$ 呢？
>
> <!-- bilingual-en:start -->
> For $x^2-1\le0$, can zero serve as a Slater point? What about the boundary point one?
> <!-- bilingual-en:end -->
>
> **答案：** $x=0$ 严格满足 $-1<0$，可以；$x=1$ 只取等号，不是严格可行点。
>
> <!-- bilingual-en:start -->
> **Answer:** Zero is strictly feasible because $-1<0$. The boundary point one satisfies equality and is not a strict Slater point.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Boyd and Vandenberghe, *Convex Optimization*, §5.2.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：直接核对 Slater 的相对内部、严格凸不等式与仿射等式定义。
- LSE EC400，*SOFP Lecture Notes*（课程讲义）：核对课程中的严格可行点与仿射等式范围。

<!-- bilingual-en:start -->
- Boyd and Vandenberghe, *Convex Optimization*, §5.2.3 was checked for Slater's relative-interior strict-feasibility definition and affine equalities.
- The EC400 SOFP notes were checked for the course scope of strict feasibility and affine equalities.
<!-- bilingual-en:end -->
