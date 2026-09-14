---
aliases:
  - '凹目标、凸不等式和仿射等式构成的最大化问题中，任何满足 KKT 的可行点都是全局最优，且这一充分性不需要 Slater 条件'
  - In a maximization problem with a concave objective, convex inequalities, and affine equalities, every KKT point is globally optimal, and this sufficiency does not require Slater
student_os: knowledge-atom
atom_id: OPT-MV-009
atom_set: multivariable-optimization
atom_type: global-optimality-theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[KKT条件]]"
  - "[[凸优化全局性]]"
related:
  - "[[Slater强对偶]]"
  - "[[半正定矩阵]]"
  - "[[严格凹不保证乘子唯一]]"
  - "[[KKT必要性]]"
  - "[[值函数]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# 凹目标、凸不等式和仿射等式构成的最大化问题中，任何满足 KKT 的可行点都是全局最优，且这一充分性不需要 Slater 条件
<!-- bilingual-en:start -->
*In a maximization problem with a concave objective, convex inequalities, and affine equalities, every KKT point is globally optimal, and this sufficiency does not require Slater*
<!-- bilingual-en:end -->

> [!summary] KKT 在凸结构下成为全局证书
> 考虑
> $$
> \max_x f(x)\quad\text{s.t.}\quad g_i(x)\le0,\qquad Ax=b,
> $$
> 其中 $f$ 可微且凹，每个 $g_i$ 可微且凸。若可行点 $x^*$ 与某些 $\lambda^*\ge0,\nu^*$ 满足全部 KKT 条件，则 $x^*$ 是全局最大点。这个方向不要求先验证 Slater。
>
> <!-- bilingual-en:start -->
> Consider maximising a differentiable concave $f$ subject to differentiable convex inequalities $g_i(x)\le0$ and affine equalities $Ax=b$. If a feasible $x^*$ and some $\lambda^*\ge0,\nu^*$ satisfy all KKT conditions, then $x^*$ is a global maximizer. This direction does not require Slater.
> <!-- bilingual-en:end -->

## 为什么 KKT 足以证明全局最优
<!-- bilingual-en:start -->
*Why KKT certifies global optimality*
<!-- bilingual-en:end -->

在统一约定
$$
\mathcal L(x,\lambda,\nu)=f(x)-\lambda^Tg(x)-\nu^T(Ax-b)
$$
下，固定 $\lambda^*\ge0$ 时，$f$ 凹且 $-\lambda_i^*g_i$ 凹，所以 $\mathcal L(\cdot,\lambda^*,\nu^*)$ 凹。驻点条件使 $x^*$ 成为这个凹函数的全局最大点。

<!-- bilingual-en:start -->
Under the convention $\mathcal L=f-\lambda^Tg-\nu^T(Ax-b)$, fixing $\lambda^*\ge0$ makes the Lagrangian concave in $x$: $f$ is concave and every $-\lambda_i^*g_i$ is concave. Stationarity therefore makes $x^*$ a global maximizer of this concave Lagrangian.
<!-- bilingual-en:end -->

对任意其他可行点 $x$，有 $g(x)\le0$、$Ax=b$，所以
$$
\mathcal L(x,\lambda^*,\nu^*)
=f(x)-{\lambda^*}^Tg(x)
\ge f(x).
$$
而在 $x^*$，互补松弛与等式可行性给
$$
\mathcal L(x^*,\lambda^*,\nu^*)=f(x^*).
$$
结合 Lagrangian 的全局最大性便得 $f(x^*)\ge f(x)$。

<!-- bilingual-en:start -->
For any other feasible $x$, $g(x)\le0$ and $Ax=b$, so $\mathcal L(x,\lambda^*,\nu^*)=f(x)-{\lambda^*}^Tg(x)\ge f(x)$. At $x^*$, complementary slackness and equality feasibility give $\mathcal L(x^*,\lambda^*,\nu^*)=f(x^*)$. Global maximality of the Lagrangian then yields $f(x^*)\ge f(x)$.
<!-- bilingual-en:end -->

## 为什么这里不需要 Slater
<!-- bilingual-en:start -->
*Why Slater is not required here*
<!-- bilingual-en:end -->

上面的证明从一个已经存在的 KKT 三元组出发，没有使用严格可行点。因此 KKT 到全局最优的充分方向不依赖 Slater。最优点何时一定拥有 KKT 乘子，见 [[Slater强对偶]]。

<!-- bilingual-en:start -->
The proof starts from an existing KKT triple and never uses strict feasibility. [[Slater强对偶|Slater strong duality]] supplies the reverse direction from an optimum to the existence of KKT multipliers.
<!-- bilingual-en:end -->

$$
\text{凸结构 + KKT}\Rightarrow\text{全局最优},
$$

<!-- bilingual-en:start -->
Convex structure plus KKT implies global optimality.
<!-- bilingual-en:end -->

## 本定理不负责的唯一性问题
<!-- bilingual-en:start -->
*Uniqueness questions outside this theorem*
<!-- bilingual-en:end -->

这条定理只把已经找到的 KKT 候选认证为全局最优，不证明最优点存在、选择唯一或乘子唯一。严格凹的取得性边界见 [[严格凹不保证解存在]]；即使选择唯一，乘子仍可能不唯一，见 [[严格凹不保证乘子唯一]]。

<!-- bilingual-en:start -->
This theorem certifies an existing KKT candidate as globally optimal. It does not establish attainment, uniqueness of the primal choice, or uniqueness of multipliers. Those boundaries are explained by [[严格凹不保证解存在|strict concavity without attainment]] and [[严格凹不保证乘子唯一|strict concavity without unique multipliers]].
<!-- bilingual-en:end -->

最小化版本把方向翻转：目标凸、不等式凸、等式仿射，满足相应号约定的 KKT 点为全局最小。

<!-- bilingual-en:start -->
For minimisation, reverse the objective direction: a convex objective with convex inequalities and affine equalities makes a KKT point globally minimal under the corresponding sign convention.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么“凸问题 + Slater”常被写成 KKT 的充要条件，却不能说 KKT 的充分性依赖 Slater？
>
> <!-- bilingual-en:start -->
> Why is “convex problem plus Slater” often used to state KKT as necessary and sufficient, even though KKT sufficiency does not depend on Slater?
> <!-- bilingual-en:end -->
>
> **答案：** 凸性给 KKT $\Rightarrow$ 最优；Slater 只补上最优 $\Rightarrow$ 存在 KKT 乘子的反向，二者合起来才是充要。
>
> <!-- bilingual-en:start -->
> **Answer:** Convexity gives KKT $\Rightarrow$ optimality. Slater adds only the reverse implication, optimality $\Rightarrow$ existence of KKT multipliers. Together they form an equivalence.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Boyd and Vandenberghe, *Convex Optimization*, §5.5.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：直接核对凸问题中 KKT 本身的充分性。
- [Stanford EE364A, Lecture 9 transcript](https://see.stanford.edu/materials/lsocoee364a/transcripts/ConvexOptimizationI-Lecture09.html)：交叉核对 KKT、强对偶与 Slater 的分工。
- LSE EC400，*SOFP Lecture Notes*（课程讲义）：核对课程中的凹目标、凸不等式与 Lagrangian 证明；仿射等式假设及充分、必要方向的区分由上述两个外部来源交叉核对。

<!-- bilingual-en:start -->
- Boyd and Vandenberghe, *Convex Optimization*, §5.5.3 was checked directly for KKT sufficiency in a convex problem.
- The Stanford EE364A Lecture 9 transcript was used to cross-check the roles of KKT, strong duality, and Slater.
- The EC400 SOFP lecture notes were checked for the course proof using a concave objective, convex inequalities, and the Lagrangian. Equality constraints were corrected to affine ones, and the sufficient and necessary directions were separated.
<!-- bilingual-en:end -->
