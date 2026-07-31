---
aliases:
  - "Multivariable Optimization"
  - "Constrained Optimization"
  - "Lagrange Multipliers"
  - "KKT Conditions"
  - "约束优化"
status: source-checked
---

# 多元优化：无约束、Lagrange、KKT 与包络
<!-- bilingual-en:start -->
*Multivariable optimisation: unconstrained problems, Lagrange multipliers, KKT conditions, and the envelope theorem*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 在多个选择变量、等式或不等式约束下，找出候选最优解并判断它是局部还是全局解。
> **具体锚点：** 消费者在预算线上选两种商品；内点最优时，效用等高线与预算线相切，但角点解不满足同样的切线条件。
> **核心难点：** 一阶条件只在正则内点上给候选；KKT 需同时检查可行性、乘子符号、驻点和互补松弛。
> **为什么重要：** 微观经济、资源分配、估计、机器学习与政策比较静态都是约束优化。
> **继续：** 先定义可行域与边界，再根据约束类型选无约束、Lagrange 或 KKT，最后用凸性/凹性或值比较确认。
> <!-- bilingual-en:start -->
> **What it solves:** Multivariable optimisation finds and verifies optima with several choices and equality or inequality constraints.
> **Concrete anchor:** A consumer chooses two goods on a budget line. At a regular interior optimum, the utility contour is tangent to the budget line, whereas a corner solution need not satisfy the same tangency equation.
> **Central difficulty:** First-order conditions generate candidates only at regular points. KKT requires feasibility, multiplier signs, stationarity, and complementary slackness together.
> **Why it matters:** Resource allocation, estimation, machine learning, and comparative statics are constrained optimisation problems.
> **Continue with:** Define the feasible set first, choose the condition set that matches the constraints, and verify local versus global optimality.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[01_Math/08_MathsCamp-EC400/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf]]：支持 EC400 课程范围、记号与例题。
> <!-- bilingual-en:start -->
> - [[01_Math/08_MathsCamp-EC400/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf|EC400 revision mathematics notes]] were checked for unconstrained first- and second-order conditions, equality constraints, Lagrange multipliers, inequality constraints, KKT conditions, and comparative statics.
> <!-- bilingual-en:end -->

## 无约束优化
<!-- bilingual-en:start -->
*Unconstrained optimisation*
<!-- bilingual-en:end -->

内部可微最优满足 $\nabla f=0$。先检查定义域、边界和不可微点，再分类候选。严格凸目标的任何局部最小都是唯一全局最小；没有凸性时初值和局部结构重要。
<!-- bilingual-en:start -->
An interior differentiable optimum satisfies $\nabla f=0$. Check the domain, boundary, and nondifferentiable points before classifying candidates. For a strictly convex objective, every local minimum is the unique global minimum; without convexity, starting values and local structure matter.
<!-- bilingual-en:end -->

## 等式约束与 Lagrange 乘子
<!-- bilingual-en:start -->
*Equality constraints and Lagrange multipliers*
<!-- bilingual-en:end -->

约束 $g(x)=c$ 下，内点且约束梯度非零时 $\nabla f=\lambda\nabla g$。几何上目标等高面与可行面相切；$\lambda$ 在正则条件下是放宽约束一单位的最优值边际变化。多个约束使用多个乘子。
<!-- bilingual-en:start -->
Under $g(x)=c$, at a regular interior point with nonzero constraint gradient, $\nabla f=\lambda\nabla g$. Geometrically, an objective contour is tangent to the feasible surface. Under regularity, $\lambda$ measures the marginal change in the optimum value from relaxing the right-hand side by one unit, with its sign depending on the Lagrangian convention. Multiple constraints require multiple multipliers.
<!-- bilingual-en:end -->

## 不等式与边界入口
<!-- bilingual-en:start -->
*Inequalities and boundary solutions*
<!-- bilingual-en:end -->

不等式约束需要可行性、乘子符号、互补松弛和驻点等 KKT 条件。约束是否绑定由解决定，不能预先全部当等式。
<!-- bilingual-en:start -->
Inequality constraints require the KKT conditions: primal feasibility, multiplier sign restrictions, complementary slackness, and stationarity. Whether a constraint binds is determined by the solution and cannot be assumed in advance for every inequality.
<!-- bilingual-en:end -->

## 隐函数与包络
<!-- bilingual-en:start -->
*Implicit functions and the envelope theorem*
<!-- bilingual-en:end -->

隐函数定理在 Jacobian 非奇异时给最优解对参数的局部变化。包络定理说明最优值对参数的一阶变化可忽略最优选择的间接变化，但需满足最优性与光滑条件。
<!-- bilingual-en:start -->
The implicit function theorem gives local movements of the optimiser with parameters when the relevant Jacobian is nonsingular. The envelope theorem says that the first-order change in the optimised value can ignore the optimiser's indirect movement, provided the necessary smoothness and optimality conditions hold.
<!-- bilingual-en:end -->

## Worked example：预算约束下的 Cobb–Douglas 效用
<!-- bilingual-en:start -->
*Worked example: Cobb–Douglas utility under a budget constraint*
<!-- bilingual-en:end -->

最大化 $\log x+\log y$，约束 $p_xx+p_yy=m$且 $x,y>0$。取 $\mathcal L=\log x+\log y+\lambda(m-p_xx-p_yy)$，一阶条件给 $1/x=\lambda p_x$、$1/y=\lambda p_y$。联合预算约束得 $x=m/(2p_x)$、$y=m/(2p_y)$，两种商品各分一半支出。由于对数效用严格凹且预算集为凸集，候选点是唯一全局最优。
<!-- bilingual-en:start -->
Maximise $\log x+\log y$ subject to $p_xx+p_yy=m$ and $x,y>0$. With $\mathcal L=\log x+\log y+\lambda(m-p_xx-p_yy)$, the first-order conditions give $1/x=\lambda p_x$ and $1/y=\lambda p_y$. Combining them with the budget yields $x=m/(2p_x)$ and $y=m/(2p_y)$, so each good receives half the budget. Strict concavity of log utility and convexity of the budget set make this the unique global optimum.
<!-- bilingual-en:end -->

## 诊断顺序
<!-- bilingual-en:start -->
*Diagnostic sequence*
<!-- bilingual-en:end -->

每个优化问题都先写目标、变量、参数和可行域；再列所有内部、边界与不可微候选；检查约束资格与乘子符号；最后用曲率、凸性/凹性或目标值比较确认。只解方程而不回到可行性是最常见的结构错误。
<!-- bilingual-en:start -->
For every optimisation problem, state the objective, variables, parameters, and feasible set; list interior, boundary, and nondifferentiable candidates; check constraint qualifications and multiplier signs; and finally verify with curvature, convexity or concavity, or direct objective comparison. Solving equations without returning to feasibility is the most common structural error.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### Lagrange 乘子为何可解释为影子价格？
<!-- bilingual-en:start -->
*Why can a Lagrange multiplier be interpreted as a shadow price?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 正则最优下，它等于约束右端小幅放宽对最优目标值的边际影响，符号取决于 Lagrangian 约定。
> <!-- bilingual-en:start -->
> At a regular optimum, it is the marginal change in the optimal value caused by a small relaxation of the constraint's right-hand side, with sign determined by the Lagrangian convention.
> <!-- bilingual-en:end -->

### 为什么不能把所有不等式预先都当成等式？
<!-- bilingual-en:start -->
*Why can every inequality constraint not be imposed as an equality in advance?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 最优点可能严格位于某个约束内部；互补松弛要求“约束松弛或对应乘子为零”，绑定状态必须由解确定。
> <!-- bilingual-en:start -->
> The optimum may lie strictly inside an inequality. Complementary slackness requires either slackness or a zero multiplier, so binding status must be determined by the solution.
> <!-- bilingual-en:end -->

### 用自己的话说明一阶条件为什么只生成候选。
<!-- bilingual-en:start -->
*Explain in your own words why first-order conditions generate only candidates.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它只排除正则内点上存在明显改善方向的情况，不区分最大、最小或鞍点，也不覆盖边界与不正则点。
> <!-- bilingual-en:start -->
> They only rule out an obvious improving direction at a regular interior point. They do not distinguish maxima, minima, and saddles or cover boundaries and irregular points.
> <!-- bilingual-en:end -->

### 包络定理为什么能忽略最优选择的间接变化？
<!-- bilingual-en:start -->
*Why can the envelope theorem ignore the optimiser's indirect movement?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 在正则内点最优处，目标沿可行选择方向的一阶导数已为零，因此选择变化对价值的一阶间接项消失，只留参数的直接效应。
> <!-- bilingual-en:start -->
> At a regular interior optimum, the objective's first-order derivative along feasible choice directions is already zero. The optimiser's indirect first-order contribution therefore vanishes, leaving the parameter's direct effect.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[01_Math/08_MathsCamp-EC400/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf]]：支持 EC400 课程范围、记号与例题。
<!-- bilingual-en:start -->
- [[01_Math/08_MathsCamp-EC400/01_Revision_Mathematics/Notes/Complete/Revision Maths Notes all.pdf|EC400 revision mathematics notes]] were reopened and checked for unconstrained conditions, Lagrange multipliers, shadow-price interpretation, KKT logic, convexity and concavity, and the envelope result in the course's scope.
<!-- bilingual-en:end -->
