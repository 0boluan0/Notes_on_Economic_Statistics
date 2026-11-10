---
aliases:
  - KKT sufficiency
  - KKT sufficient conditions for global optimality
  - KKT全局最优充分条件
student_os: knowledge-atom
atom_id: OPT-MV-009
atom_set: multivariable-optimization
atom_type: global-optimality-theorem
status: source-checked
requires:
  - "[[KKT条件]]"
  - "[[凹凸性]]"
  - "[[凹函数驻点全局最优]]"
  - "[[互补松弛]]"
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

<!-- bilingual-en:start -->
# KKT sufficiency for global optimality
<!-- bilingual-en:end -->

*KKT 的全局最优充分条件*

<!-- bilingual-en:start -->
A solution of the [[KKT条件|KKT equations]] is initially a candidate: it balances the objective gradient against constraint gradients. With the right global shapes, those same equations prove that **no feasible input has a larger objective value**. This is KKT sufficiency. The proof builds an upper bound that touches the objective at the candidate.
<!-- bilingual-en:end -->

[[KKT条件|KKT 方程]]的解起初只是候选：它把目标梯度与约束梯度配平。若函数还具有相应的全局形状，同一组方程便能证明：**没有任何可行输入取得更大的目标值**。这就是 KKT 的充分性。证明会构造一个上界，并让它恰好在候选点贴住目标。

<!-- bilingual-en:start -->
Work on one nonempty open [[凸集|convex domain]] $U\subseteq\mathbb R^n$, with integer $n\ge1$. The real-valued objective $f$ is differentiable and [[凹凸性|concave]] on $U$; each of the finitely many real-valued inequalities $g_i$ is differentiable and convex there. Equalities are $Ax=b$, with $A\in\mathbb R^{p\times n}$ and $b\in\mathbb R^p$. Either constraint count may be zero. The feasible set is $F=\{x\in U:g_i(x)\le0\text{ for every }i,\ Ax=b\}$. All restrictions, including quantity bounds, must be included in this description.
<!-- bilingual-en:end -->

在同一个非空开[[凸集]] $U\subseteq\mathbb R^n$ 上讨论，整数 $n\ge1$。实值目标 $f$ 在 $U$ 上可微且[[凹凸性|凹]]；有限条实值不等式函数 $g_i$ 在该域上可微且凸。等式写成 $Ax=b$，其中 $A\in\mathbb R^{p\times n}$、$b\in\mathbb R^p$。任一类约束都可以没有。可行集为 $F=\{x\in U:g_i(x)\le0\text{ 对所有 }i,\ Ax=b\}$，数量上下界等限制也必须计入这份描述。

> [!theorem] A global certificate · 全局最优证书
> <!-- bilingual-en:start -->
> Consider maximising a differentiable concave $f$ subject to differentiable convex inequalities $g_i(x)\le0$ and affine equalities $Ax=b$. If a feasible $x^*$ and some $\lambda^*\ge0,\nu^*$ satisfy all KKT conditions, then $x^*$ is a global maximizer. This direction does not require Slater.
> <!-- bilingual-en:end -->
>
> 考虑
> $$
> \max_x f(x)\quad\text{s.t.}\quad g_i(x)\le0,\qquad Ax=b,
> $$
> 其中 $f$ 可微且凹，每个 $g_i$ 可微且凸。若可行点 $x^*$ 与某些 $\lambda^*\ge0,\nu^*$ 满足全部 KKT 条件，则 $x^*$ 是全局最大点。这个方向不要求先验证 Slater。

<!-- bilingual-en:start -->
Here $x^*\in F$ is the candidate input. The inequality coefficients $\lambda_i^*$ are nonnegative; equality coefficients $\nu_j^*$ may have either sign. Besides feasibility, the required equations are $\nabla f(x^*)-\sum_i\lambda_i^*\nabla g_i(x^*)-A^T\nu^*=0$ and $\lambda_i^*g_i(x^*)=0$ for every $i$. The star identifies this one candidate and its coefficients. The [[全微分#实值函数、梯度与全微分|gradient]] is the column of first partial derivatives.
<!-- bilingual-en:end -->

这里 $x^*\in F$ 是候选输入；不等式系数 $\lambda_i^*$ 非负，等式系数 $\nu_j^*$ 可正可负。除了可行性，还要求 $\nabla f(x^*)-\sum_i\lambda_i^*\nabla g_i(x^*)-A^T\nu^*=0$，以及对每个 $i$ 都有 $\lambda_i^*g_i(x^*)=0$。星号只标记这一候选点及其系数；[[全微分#实值函数、梯度与全微分|梯度]]是各一阶偏导排成的列。

<!-- bilingual-en:start -->
## Why the Lagrangian is a touching upper bound
<!-- bilingual-en:end -->

## 为什么 KKT 足以证明全局最优

<!-- bilingual-en:start -->
Under the convention $\mathcal L=f-\lambda^Tg-\nu^T(Ax-b)$, fixing $\lambda^*\ge0$ makes the Lagrangian concave in $x$: $f$ is concave and every $-\lambda_i^*g_i$ is concave. Stationarity therefore makes $x^*$ a global maximizer of this concave Lagrangian.
<!-- bilingual-en:end -->

在统一约定
$$
\mathcal L(x,\lambda,\nu)=f(x)-\lambda^Tg(x)-\nu^T(Ax-b)
$$
下，固定 $\lambda^*\ge0$ 时，$f$ 凹且 $-\lambda_i^*g_i$ 凹，所以 $\mathcal L(\cdot,\lambda^*,\nu^*)$ 凹。驻点条件使 $x^*$ 成为这个凹函数的全局最大点。

<!-- bilingual-en:start -->
The multipliers are held fixed while $x$ varies over all of $U$. Each $-\lambda_i^*g_i$ is concave because multiplying a convex function by a nonpositive scalar reverses its defining inequality. The equality term is affine regardless of the sign of $\nu^*$. The [[凹函数非负加权和|sum rule]] therefore gives a concave Lagrangian. Its gradient is zero at $x^*$ by stationarity, so [[凹函数驻点全局最优|the concave stationary-point theorem]] bounds its value everywhere on $U$. A feasible boundary of $F$ is allowed: $x^*$ is still inside the open function domain $U$.
<!-- bilingual-en:end -->

让 $x$ 在整个 $U$ 中变化时，乘子保持固定。凸函数乘以非正数会翻转其定义不等式，所以每个 $-\lambda_i^*g_i$ 都凹；无论 $\nu^*$ 正负，等式项始终仿射。[[凹函数非负加权和|求和规则]]因此使 Lagrangian 为凹函数。驻点条件又使它在 $x^*$ 的梯度为零，[[凹函数驻点全局最优|凹函数驻点定理]]就给出整个 $U$ 上的上界。候选可以处在可行集 $F$ 的边界，因为它仍位于函数开域 $U$ 内。

<!-- bilingual-en:start -->
For any other feasible $x$, $g(x)\le0$ and $Ax=b$, so $\mathcal L(x,\lambda^*,\nu^*)=f(x)-{\lambda^*}^Tg(x)\ge f(x)$. At $x^*$, complementary slackness and equality feasibility give $\mathcal L(x^*,\lambda^*,\nu^*)=f(x^*)$. Global maximality of the Lagrangian then yields $f(x^*)\ge f(x)$.
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
Putting the three comparisons in their order of use makes the certificate explicit. The first uses feasibility and nonnegative inequality multipliers, the second uses concavity and stationarity, and the final equality uses complementary slackness.
<!-- bilingual-en:end -->

把三个比较按使用顺序连起来，证书就完整了：第一步来自可行性与非负不等式乘子，第二步来自凹性与驻点，最后的等号来自互补松弛。

$$f(x)\le\mathcal L(x,\lambda^*,\nu^*)\le\mathcal L(x^*,\lambda^*,\nu^*)=f(x^*)\qquad(x\in F).$$

> [!example] A binding upper bound · 上界恰好绑定
> <!-- bilingual-en:start -->
> Maximize $f(x)=-(x-2)^2$ on $\mathbb R$ subject to $g(x)=x-1\le0$. Take $x^*=1$ and $\lambda^*=2$. Feasibility gives $g(1)=0$; stationarity gives $f^\prime(1)-2g^\prime(1)=2-2=0$; the multiplier is nonnegative and its complementary product is zero. Completing the square gives $\mathcal L(x,2)=-(x-2)^2-2(x-1)=-(x-1)^2-1\le-1$. At $x^*=1$, $f(1)=\mathcal L(1,2)=-1$. Every feasible objective is below this same bound, so $1$ is a global maximizer.
> <!-- bilingual-en:end -->
>
> 在 $\mathbb R$ 上最大化 $f(x)=-(x-2)^2$，约束为 $g(x)=x-1\le0$。取 $x^*=1$、$\lambda^*=2$：$g(1)=0$ 保证可行，$f^\prime(1)-2g^\prime(1)=2-2=0$ 保证驻点，乘子非负且互补乘积为零。配方得到 $\mathcal L(x,2)=-(x-2)^2-2(x-1)=-(x-1)^2-1\le-1$。在 $x^*=1$，$f(1)=\mathcal L(1,2)=-1$，任何可行目标值都不超过这个共同上界，所以 $1$ 是全局最大点。

<!-- bilingual-en:start -->
## Why this direction does not need Slater
<!-- bilingual-en:end -->

## 为什么这里不需要 Slater

<!-- bilingual-en:start -->
The proof starts from an existing KKT triple and never uses strict feasibility. [[Slater强对偶|Slater strong duality]] supplies the reverse direction from an optimum to the existence of KKT multipliers.
<!-- bilingual-en:end -->

上面的证明从一个已经存在的 KKT 三元组出发，没有使用严格可行点。因此 KKT 到全局最优的充分方向不依赖 Slater。最优点何时一定拥有 KKT 乘子，见 [[Slater强对偶]]。

<!-- bilingual-en:start -->
Convex structure plus KKT implies global optimality.
<!-- bilingual-en:end -->

$$
\text{凸结构 + KKT}\Rightarrow\text{全局最优},
$$

> [!example] A certificate with no strictly feasible point · 没有严格可行点也可认证
> <!-- bilingual-en:start -->
> Maximize $-x^2$ subject to $x^2\le0$ on $\mathbb R$. The only feasible input is $0$, and no input satisfies $x^2<0$, so Slater fails. Nevertheless $x^*=0$, $\lambda^*=0$ satisfies all KKT conditions. The objective is concave, the inequality function is convex, and the theorem correctly certifies the attained value $0$.
> <!-- bilingual-en:end -->
>
> 在 $\mathbb R$ 上最大化 $-x^2$，约束为 $x^2\le0$。唯一可行输入为 $0$，没有任何输入满足 $x^2<0$，因此 Slater 失败；但 $x^*=0$、$\lambda^*=0$ 满足全部 KKT 条件。目标凹、不等式函数凸，定理仍正确认证已取得的值 $0$。

<!-- bilingual-en:start -->
## What the certificate settles, and what it assumes
<!-- bilingual-en:end -->

## 本定理不负责的唯一性问题

<!-- bilingual-en:start -->
The theorem is conditional on finding a KKT candidate. It cannot produce a candidate before one is known. Once such a candidate is verified under the stated shapes, however, the proof does establish that the maximum is attained at that candidate. The following existence boundary concerns the problem before that certificate is available.
<!-- bilingual-en:end -->

该定理以已经找到 KKT 候选为前提，不能先替你产生候选。但一旦候选在给定形状条件下核验通过，证明就确实说明最大值在此取得。下面关于存在性的边界，指尚未获得这份证书时的问题。

<!-- bilingual-en:start -->
This theorem certifies an existing KKT candidate as globally optimal. It does not establish that a KKT candidate exists, or that the primal choice or multipliers are unique. Those boundaries are explained by [[严格凹不保证解存在|strict concavity without attainment]] and [[严格凹不保证乘子唯一|strict concavity without unique multipliers]].
<!-- bilingual-en:end -->

这条定理把已经找到的 KKT 候选认证为全局最优；它不先证明 KKT 候选一定存在，也不保证选择唯一或乘子唯一。严格凹的取得性边界见 [[严格凹不保证解存在]]；即使选择唯一，乘子仍可能不唯一，见 [[严格凹不保证乘子唯一]]。

<!-- bilingual-en:start -->
For minimisation, reverse the objective direction: a convex objective with convex inequalities and affine equalities makes a KKT point globally minimal under the corresponding sign convention.
<!-- bilingual-en:end -->

最小化版本把方向翻转：目标凸、不等式凸、等式仿射，满足相应号约定的 KKT 点为全局最小。

<!-- bilingual-en:start -->
For the minimization version, write the objective as $\varphi$ and use $\mathcal L_{\min}=\varphi+\sum_i\lambda_i g_i+\nu^T(Ax-b)$, with $\lambda_i\ge0$. Stationarity is $\nabla\varphi+\sum_i\lambda_i\nabla g_i+A^T\nu=0$. Keeping the same inequality direction while changing the objective direction also changes these Lagrangian signs.
<!-- bilingual-en:end -->

最小化版本把目标记为 $\varphi$，采用 $\mathcal L_{\min}=\varphi+\sum_i\lambda_i g_i+\nu^T(Ax-b)$，仍要求 $\lambda_i\ge0$。驻点式为 $\nabla\varphi+\sum_i\lambda_i\nabla g_i+A^T\nu=0$。不等式方向保持不变、优化方向翻转时，Lagrangian 的这些符号也要一起改变。

<!-- bilingual-en:start -->
## Where the shape assumptions enter
<!-- bilingual-en:end -->

*形状假设缺失时，哪一步会失效？*

<!-- bilingual-en:start -->
Without a concave objective, a KKT point can fail even to be a local maximum. Maximize $x^2$ with $x-1\le0$ and $-x-1\le0$. At $x^*=0$, both constraints are slack and both multipliers can be zero; stationarity holds because $f^\prime(0)=0$. Yet feasible inputs $\pm1$ give $1>0$, and any sufficiently small nonzero input also improves the objective. The concave-Lagrangian step is unavailable.
<!-- bilingual-en:end -->

没有凹目标时，KKT 点甚至可能不是局部最大。最大化 $x^2$，约束为 $x-1\le0$ 与 $-x-1\le0$。在 $x^*=0$，两条约束都松弛、两个乘子都可为零，$f^\prime(0)=0$ 又满足驻点；但可行点 $\pm1$ 取得 $1>0$，任意足够小的非零输入也改善目标。这里无法使用 Lagrangian 凹性那一步。

<!-- bilingual-en:start -->
Convex equality functions cannot simply replace affine equalities. Maximize $f(x)=x$ subject to $h(x)=x^2-1=0$. The objective is concave and $h$ is convex, but $x^*=-1$, $\nu^*=-1/2$ satisfies $f^\prime-\nu^*h^\prime=1+2\nu^*=0$ while the other feasible input $1$ is better. Multiplying a nonlinear equality by its unrestricted-sign coefficient need not preserve concavity; here $\mathcal L=x+(x^2-1)/2$ is convex.
<!-- bilingual-en:end -->

不能直接把仿射等式换成凸等式函数。最大化 $f(x)=x$，约束 $h(x)=x^2-1=0$；目标凹、$h$ 也凸，但 $x^*=-1$、$\nu^*=-1/2$ 满足 $f^\prime-\nu^*h^\prime=1+2\nu^*=0$，另一个可行输入 $1$ 却更好。非线性等式乘上不限制符号的系数，不能保证保持凹性；这里 $\mathcal L=x+(x^2-1)/2$ 反而凸。

> [!question]- Why sufficiency needs no strict point · 充分性为什么不用严格可行点？
> <!-- bilingual-en:start -->
> Why is “convex problem plus Slater” often used to state KKT as necessary and sufficient, even though KKT sufficiency does not depend on Slater?
> <!-- bilingual-en:end -->
>
> 为什么“凸问题 + Slater”常被写成 KKT 的充要条件，却不能说 KKT 的充分性依赖 Slater？
>
> <!-- bilingual-en:start -->
> **Answer:** Convexity gives KKT $\Rightarrow$ optimality. Slater adds only the reverse implication, optimality $\Rightarrow$ existence of KKT multipliers. Together they form an equivalence.
> <!-- bilingual-en:end -->
>
> **答案：** 凸性给 KKT $\Rightarrow$ 最优；Slater 只补上最优 $\Rightarrow$ 存在 KKT 乘子的反向，二者合起来才是充要。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- Boyd and Vandenberghe, *Convex Optimization*, §5.5.3 was checked directly for KKT sufficiency in a convex problem.
- The Stanford EE364A Lecture 9 transcript was used to cross-check the roles of KKT, strong duality, and Slater.
- The EC400 SOFP lecture notes were checked for the course proof using a concave objective, convex inequalities, and the Lagrangian. Equality constraints were corrected to affine ones, and the sufficient and necessary directions were separated.
<!-- bilingual-en:end -->

- [Boyd and Vandenberghe, *Convex Optimization*, §5.5.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：直接核对凸问题中 KKT 本身的充分性。
- [Stanford EE364A, Lecture 9 transcript](https://see.stanford.edu/materials/lsocoee364a/transcripts/ConvexOptimizationI-Lecture09.html)：交叉核对 KKT、强对偶与 Slater 的分工。
- LSE EC400，*SOFP Lecture Notes*（课程讲义）：核对课程中的凹目标、凸不等式与 Lagrangian 证明；仿射等式假设及充分、必要方向的区分由上述两个外部来源交叉核对。

<!-- bilingual-en:start -->
Precise source scope: Boyd §5.5.3, PDF pp.257–258 / book pp.243–244, states differentiable open domains, affine equalities and sufficiency separately from Slater; the maximization proof here applies its argument to $-f$. Stanford Lecture 9, the discussion immediately following the four KKT conditions, explains the same certificate chain. EC400 SOFP Notes PDF p.33 provides the course maximization convention and upper-bound proof. The more general equality statement on p.34 is read with the affine condition verified against Boyd; the nonlinear-equality counterexample above explains why that condition matters. All new examples are calculated directly.
<!-- bilingual-en:end -->

精确范围：Boyd §5.5.3，PDF257–258 / 书页243–244，核对可微开域、仿射等式、充分性与 Slater 的独立分工；本文对其最小化目标取负，得到最大化证明。Stanford Lecture 9 在列出四类 KKT 条件之后解释同一证书链。EC400 SOFP Notes 的 PDF33页提供课程的最大化号约定与上界证明；34页涉及等式的更一般措辞须按 Boyd 核对为仿射条件，上面的非线性等式反例说明其必要作用。所有新增例子直接复算。

