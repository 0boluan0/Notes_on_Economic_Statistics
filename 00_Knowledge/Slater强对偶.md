---
aliases:
- Strong duality under Slater's condition
- Slater strong duality
- Slater 强对偶定理
student_os: knowledge-atom
atom_id: OPT-MV-022
atom_set: multivariable-optimization
atom_type: duality-theorem
status: source-checked
requires:
- '[[Slater条件]]'
- '[[KKT条件]]'
related:
- '[[KKT充分性]]'
leads_to:
- '[[KKT必要性]]'
part_of:
- '[[多元优化.canvas|多元优化]]'
---

<!-- bilingual-en:start -->
# Strong duality under Slater's condition
<!-- bilingual-en:end -->

*Slater 强对偶*

<!-- bilingual-en:start -->
There are two ways to pin down the largest feasible objective value: find better feasible choices, or build upper bounds that no feasible choice can exceed. **Strong duality** says the best bound has exactly the same value as the original optimization problem. [[Slater条件|Slater's condition]], the existence of an appropriately strict feasible point, is a sufficient hypothesis for this equality in convex optimization. Under the finite-value assumption below, it also makes the best bound attainable by a finite vector of multipliers.
<!-- bilingual-en:end -->

确定最大可行目标值有两条路：寻找更好的可行选择，或者构造任何可行选择都越不过的上界。**强对偶**说这两条路给出相同的最优值。[[Slater条件]]要求存在适当的严格可行点，是凸优化中保证这种相等的一项充分条件。在下面的有限最优值假设下，它还保证某组有限乘子能够真正给出这个最好的上界。

<!-- bilingual-en:start -->
## What the dual problem is bounding
<!-- bilingual-en:end -->

## 对偶问题在给什么作界

<!-- bilingual-en:start -->
Use the concave maximization setup of [[Slater条件|the Slater definition]]: $f$ is concave, each $g_i$ convex, equalities are $Ax=b$, and $D$ is the common convex domain. The feasible set is $F=\{x\in D:g_i(x)\le0,Ax=b\}$. Write $p^*=\sup_{x\in F}f(x)$ for the primal optimal value. A [[上确界|supremum]] is the least upper bound; a finite supremum need not equal $f(x)$ at any feasible point. Throughout the theorem, assume $p^*\in\mathbb R$.
<!-- bilingual-en:end -->

采用 [[Slater条件]] 的凹最大化背景：$f$ 凹，每个 $g_i$ 凸，等式为 $Ax=b$，共同凸定义域为 $D$。可行集是 $F=\{x\in D:g_i(x)\le0,Ax=b\}$，原问题最优值记为 $p^*=\sup_{x\in F}f(x)$。[[上确界]]是最小的上界；即便上确界有限，也未必有可行点的目标值恰好等于它。本定理明确假设 $p^*\in\mathbb R$。

<!-- bilingual-en:start -->
Choose nonnegative weights $\lambda_i$ for inequalities and unrestricted weights $\nu_j$ for equalities. With the maximization sign convention, define the Lagrangian $\mathcal L$ and dual function $q$ by the following formulas. In $q$, the supremum runs over the common domain $D$, without imposing the displayed constraints again. Its value may be $+\infty$.
<!-- bilingual-en:end -->

给不等式选取非负权重 $\lambda_i$，给等式选取不限制符号的权重 $\nu_j$。按照最大化的符号约定，定义下面的 Lagrangian $\mathcal L$ 和对偶函数 $q$。计算 $q$ 时，在共同定义域 $D$ 上取上确界，不再额外施加写出的约束；$q$ 可能等于 $+\infty$。

$$
\mathcal L(x,\lambda,\nu)=f(x)-\sum_{i=1}^m\lambda_i g_i(x)-\nu^T(Ax-b),\qquad q(\lambda,\nu)=\sup_{x\in D}\mathcal L(x,\lambda,\nu),\quad\lambda\ge0.
$$

<!-- bilingual-en:start -->
For a feasible $x$, the equality term vanishes and each $-\lambda_i g_i(x)$ is nonnegative. Therefore $f(x)\le\mathcal L(x,\lambda,\nu)\le q(\lambda,\nu)$. Every choice of allowed multipliers produces an upper bound on every feasible objective value. The dual problem searches for the smallest such bound, so weak duality has the direction $p^*\le d^*$ here.
<!-- bilingual-en:end -->

若 $x$ 可行，等式项为零，各项 $-\lambda_i g_i(x)$ 非负，因此 $f(x)\le\mathcal L(x,\lambda,\nu)\le q(\lambda,\nu)$。每组允许的乘子都给出所有可行目标值的一个上界；对偶问题要找这些上界中最小的一个，所以这里弱对偶的方向是 $p^*\le d^*$。

$$
d^*=\inf_{\lambda\ge0,\ \nu\in\mathbb R^p}q(\lambda,\nu),\qquad p^*\le d^*.
$$

> [!important] Equality of values and dual attainment · 值相等与对偶最优取得
> <!-- bilingual-en:start -->
> For the convex problem just specified, assume $p^*\in\mathbb R$ and either the basic or refined [[Slater条件|Slater condition]]. Then $d^*=p^*$, and some $\lambda^*\ge0$, $\nu^*\in\mathbb R^p$ attain the dual infimum: $q(\lambda^*,\nu^*)=p^*$. **Strong duality** names the equality of values; **dual attainment** says that the infimum is achieved by actual multipliers. The theorem gives both, but does not assert primal attainment.
> <!-- bilingual-en:end -->
>
> 对上述凸问题，假设 $p^*\in\mathbb R$，并满足基本或精细 [[Slater条件]]。则 $d^*=p^*$，而且存在 $\lambda^*\ge0$、$\nu^*\in\mathbb R^p$ 取得对偶下确界：$q(\lambda^*,\nu^*)=p^*$。**强对偶**指最优值相等；**对偶最优取得**指存在实际乘子实现这个下确界。定理同时给出这两点，却没有宣称原问题最优点一定存在。

<!-- bilingual-en:start -->
To connect this value theorem to the ordinary gradient form of KKT, the later sections additionally assume a common open convex domain and differentiability of the objective and constraints there, as well as an attained primal optimum. Read the following familiar summary in that setting.
<!-- bilingual-en:end -->

要把这个关于最优值的定理接到通常的梯度形式 KKT，后文还会要求共同域为开凸集、目标和约束在其中可微，并且原问题最优值已由某个点取得。下面的常用概括在这个范围内理解。

<!-- bilingual-en:start -->
In a convex maximisation problem with a concave objective, convex inequalities, and affine equalities, Slater's condition gives equality of primal and dual optimal values and attainment of optimal multipliers under the usual finite-value assumptions. In the differentiable setting, a primal optimum can therefore be paired with multipliers satisfying KKT.
<!-- bilingual-en:end -->

对凹最大化、凸不等式与仿射等式组成的凸问题，若 [[Slater条件]] 成立，并满足通常的有限最优值假设，则原问题与对偶问题的最优值相等，而且最优对偶乘子可以取得。对可微问题，这使每个原问题最优点能够配上乘子满足 [[KKT条件]]。

<!-- bilingual-en:start -->
## A bound that meets the best feasible value
<!-- bilingual-en:end -->

## 上界与可行最优值相遇的例子

<!-- bilingual-en:start -->
Maximize $f(x)=-(x-2)^2$ subject to $x\le1$, with $D=\mathbb R$. Zero is strictly feasible. Among feasible points, $x=1$ is closest to two, so $p^*=f(1)=-1$. Write the constraint as $g(x)=x-1\le0$. Completing the square in $x$ gives the whole family of upper bounds:
<!-- bilingual-en:end -->

最大化 $f(x)=-(x-2)^2$，约束为 $x\le1$，共同域 $D=\mathbb R$。零点严格可行；在可行点中，$x=1$ 最接近二，所以 $p^*=f(1)=-1$。将约束写成 $g(x)=x-1\le0$，对 $x$ 配方，就能得到整族上界：

$$
\begin{aligned}\mathcal L(x,\lambda)&=-(x-2)^2-\lambda(x-1)\\&=-\left(x-\left(2-\frac\lambda2\right)\right)^2+\frac{\lambda^2}{4}-\lambda,\\q(\lambda)&=\frac{\lambda^2}{4}-\lambda=\frac{(\lambda-2)^2}{4}-1,\qquad\lambda\ge0.\end{aligned}
$$

<!-- bilingual-en:start -->
For each fixed $\lambda$, the negative square is at most zero and becomes zero at $x=2-\lambda/2$, proving the displayed supremum. The smallest $q$ occurs at $\lambda^*=2$ and equals $-1$. Thus $x^*=1$ reaches the bound that $\lambda^*=2$ certifies: a feasible value and a valid upper bound agree.
<!-- bilingual-en:end -->

固定 $\lambda$ 后，负平方项至多为零，并在 $x=2-\lambda/2$ 取零，因而上式确实算出了上确界。$q$ 在 $\lambda^*=2$ 取得最小值 $-1$。于是可行点 $x^*=1$ 达到了乘子 $\lambda^*=2$ 所证明的上界：可行值和有效上界相遇。

<!-- bilingual-en:start -->
## A finite value need not be reached by a primal point
<!-- bilingual-en:end -->

## 有限最优值未必有原问题最优点

> [!example] Strong duality without a primal maximizer · 强对偶不保证原问题取得
> <!-- bilingual-en:start -->
> Maximize $-e^{-x}$ subject to $x\ge0$, written as $g(x)=-x\le0$, with $D=\mathbb R$. The objective is concave since its second derivative is $-e^{-x}<0$, and $x=1$ is a Slater witness. Feasible values approach zero as $x\to\infty$, but are always negative, so $p^*=0$ and no maximizer exists. At $\lambda=0$, $q(0)=\sup_{x\in\mathbb R}(-e^{-x})=0$. For $\lambda>0$, $\mathcal L=-e^{-x}+\lambda x\to+\infty$ as $x\to\infty$, so $q(\lambda)=+\infty$. The dual minimum is attained at zero even though the primal supremum is not attained.
> <!-- bilingual-en:end -->
>
> 最大化 $-e^{-x}$，约束 $x\ge0$ 写为 $g(x)=-x\le0$，共同域为 $\mathbb R$。目标二阶导数 $-e^{-x}<0$，所以凹；$x=1$ 是 Slater 见证点。可行目标值在 $x\to\infty$ 时趋近零，却始终为负，因此 $p^*=0$，没有最大点。取 $\lambda=0$，$q(0)=\sup_{x\in\mathbb R}(-e^{-x})=0$；若 $\lambda>0$，则 $\mathcal L=-e^{-x}+\lambda x$ 在 $x\to\infty$ 时趋于 $+\infty$，故 $q(\lambda)=+\infty$。对偶最小值由零乘子取得，原问题上确界却没有被任何点取得。

<!-- bilingual-en:start -->
Finiteness is also a real hypothesis, not another name for strict feasibility. Maximizing $x$ subject to $x\ge0$ has a strict feasible point, but $p^*=+\infty$; no finite upper-bound certificate of a finite optimum can follow. The finite-value theorem above simply does not cover it.
<!-- bilingual-en:end -->

有限性也是实质假设，不是严格可行性的另一种说法。最大化 $x$，约束 $x\ge0$，虽有严格可行点，却有 $p^*=+\infty$，无法由此得到某个有限最优值的有限上界证书。上面的有限值定理不覆盖这个问题。

<!-- bilingual-en:start -->
## Which logical direction it supplies
<!-- bilingual-en:end -->

## 它补的是哪一个方向

<!-- bilingual-en:start -->
Now suppose the common domain $U$ is open and convex, all functions are differentiable on $U$, and a feasible $x^*$ attains the finite primal optimum. With dual-optimal multipliers supplied by Slater, the weak-duality chain begins and ends at the same number:
<!-- bilingual-en:end -->

现在进一步假设共同定义域 $U$ 为开凸集，各函数在 $U$ 上可微，可行点 $x^*$ 取得有限的原问题最优值。用 Slater 保证存在的对偶最优乘子，弱对偶链的两端成为同一个数：

$$
f(x^*)=p^*\le\mathcal L(x^*,\lambda^*,\nu^*)\le q(\lambda^*,\nu^*)=p^*.
$$

<!-- bilingual-en:start -->
Every inequality in this chain is equality. The difference $\mathcal L(x^*,\lambda^*,\nu^*)-f(x^*)$ is a sum of nonnegative terms $-\lambda_i^*g_i(x^*)$; their zero sum makes each product zero, giving complementary slackness. Equality with $q$ says that $x^*$ maximizes $\mathcal L$ over $U$. Since $U$ is open and $\mathcal L$ differentiable, its gradient at $x^*$ vanishes. Primal feasibility and multiplier signs were already available. These are all four parts of [[KKT条件|KKT]], with the same maximization signs.
<!-- bilingual-en:end -->

链上每个不等式都必须取等号。$\mathcal L(x^*,\lambda^*,\nu^*)-f(x^*)$ 是非负项 $-\lambda_i^*g_i(x^*)$ 的和；总和为零使每个乘积为零，得到互补松弛。与 $q$ 相等说明 $x^*$ 在 $U$ 上最大化 $\mathcal L$；由于 $U$ 为开集且 $\mathcal L$ 可微，它在 $x^*$ 的梯度为零。原问题可行性和乘子符号已经具备，合起来就是同一最大化符号约定下 [[KKT条件]] 的四部分。

<!-- bilingual-en:start -->
Under these differentiability, open-domain, finite-value, and attainment assumptions, the following short implication is valid. It is a necessity route, not the definition of KKT.
<!-- bilingual-en:end -->

在上述可微性、开域、有限值及最优值取得的前提下，下面的简写推论成立。它是必要性路线，不是 KKT 的定义。

<!-- bilingual-en:start -->
Slater supplies the direction from a convex optimum to the existence of KKT multipliers. The reverse certification from an existing KKT point to global optimality follows from convex structure alone and does not require Slater.
<!-- bilingual-en:end -->

Slater 强对偶给出
$$
\text{凸结构 + Slater + 最优}
\Rightarrow
\text{存在 KKT 乘子}.
$$

它是一条 KKT 必要性路线。相反，若一个凸问题已经找到满足 KKT 的可行点，[[KKT充分性]] 可以直接把它认证为全局最优，不需要先验证 Slater。

> [!warning] Account for the domain in stationarity · 驻点条件须计入定义域
> <!-- bilingual-en:start -->
> The relative-interior Slater theorem allows domains that are not open or are lower-dimensional. It does not automatically turn constrained maximization over such a domain into the ambient equation $\nabla_x\mathcal L=0$. For example, maximize $f(x)=-x$ on $D=[0,\infty)$ with no displayed constraints. The relative-interior test holds, the maximum is zero at zero, and the dual value is zero, yet the ambient extension $f(x)=-x$ has derivative $-1$ at zero. The domain boundary is still a constraint. Write it explicitly (here $-x\le0$), work in appropriate affine coordinates when relevant, or use a stationarity condition that accounts for the domain.
> <!-- bilingual-en:end -->
>
> 相对内部版本的 Slater 定理允许非开域或低维定义域；它不会自动把这类域上的约束最大化改写为环境空间中的 $\nabla_x\mathcal L=0$。例如，在 $D=[0,\infty)$ 上最大化 $f(x)=-x$，不另列约束。相对内部检查通过，原问题在零点取得最大值零，对偶值也为零，但环境空间上的延拓 $f(x)=-x$ 在零点的导数为 $-1$。定义域边界依然是一项限制；可将它显式写成约束（这里是 $-x\le0$），在适合时使用仿射坐标，或采用把定义域计入的驻点条件。

<!-- bilingual-en:start -->
## Why it often appears in an if-and-only-if statement
<!-- bilingual-en:end -->

## 为什么常被写成充要条件

<!-- bilingual-en:start -->
Combining Slater-based necessity with convex KKT sufficiency yields the familiar equivalence: under Slater, a point is optimal if and only if it can be paired with KKT multipliers. Slater belongs only to the optimum-to-KKT direction.
<!-- bilingual-en:end -->

把两个独立方向合在一起便得到：在可微凸问题且 Slater 成立时，$x^*$ 最优，当且仅当存在乘子使 $(x^*,\lambda^*,\nu^*)$ 满足 KKT。这里 Slater 只负责“最优推出 KKT”，不是 KKT 系统的定义，也不是“KKT 推出最优”的假设。

<!-- bilingual-en:start -->
This equivalence retains the open-domain gradient convention just stated. Also, Slater is sufficient for the duality guarantee, not necessary for it. For example, maximizing $-x^2$ subject to $x^2\le0$ has no strict feasible point, but $p^*=0$ at zero and $q(\lambda)=\sup_{x\in\mathbb R}[-(1+\lambda)x^2]=0$ for every $\lambda\ge0$. Strong duality and dual attainment still hold in this particular problem.
<!-- bilingual-en:end -->

这个充要表述仍保留前述开域下的梯度约定。另外，Slater 是对偶保证的充分条件，并非必要条件。例如，最大化 $-x^2$ 且 $x^2\le0$，没有严格可行点；但原问题在零点取得 $p^*=0$，且每个 $\lambda\ge0$ 都给出 $q(\lambda)=\sup_{x\in\mathbb R}[-(1+\lambda)x^2]=0$。这个具体问题依然具有强对偶，并取得对偶最优值。

> [!question]- Separate the two directions · 自检
> <!-- bilingual-en:start -->
> In a convex problem, must Slater still be verified before an existing KKT point can be declared globally optimal?
>
> **Answer:** No. KKT sufficiency does not depend on Slater. Slater supplies the reverse direction that guarantees KKT multipliers at an optimum.
> <!-- bilingual-en:end -->
>
> 凸问题中已经找到一个满足全部 KKT 条件的点，还要验证 Slater 才能宣布全局最优吗？
>
> **答案：** 不要。KKT 充分性不依赖 Slater；Slater 用于保证最优点拥有 KKT 乘子的反向。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- Boyd and Vandenberghe, *Convex Optimization*, §§5.2.3 and 5.5.3 was checked for Slater, strong duality, dual attainment, and the two KKT directions.
  [Book text](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), printed pp. 226–227 and 243–244 / PDF pp. 240–241 and 257–258: Slater's finite-value dual attainment and the open-domain differentiable KKT directions. Section 5.1.2–3, printed p. 216 / PDF p. 230, defines the dual function and bound; negating the minimization objective yields this card's upper-bound signs. The theorem does not supply primal attainment.
<!-- bilingual-en:end -->

- [Boyd and Vandenberghe, *Convex Optimization*, §§5.2.3 and 5.5.3](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)：直接核对 Slater、强对偶、对偶最优取得与 KKT 必要/充分方向。
  精确位置：书页 226–227、243–244／PDF 页 240–241、257–258，核对有限值下的对偶最优取得，以及开域可微 KKT 的必要和充分方向；§5.1.2–3，书页 216／PDF 页 230，给出对偶函数与界。对最小化目标取负得到本卡的上界符号；定理不保证原问题最优值被取得。

<!-- bilingual-en:start -->
- The Stanford EE364A Lecture 9 transcript was used to cross-check the division of roles between strong duality and KKT certification.
  [Lecture transcript](https://see.stanford.edu/materials/lsocoee364a/transcripts/ConvexOptimizationI-Lecture09.html): the primal/dual optimum equality chain yields complementary slackness; the later discussion separates convex KKT sufficiency from Slater-based necessity. Informal references to optimal points assume their existence and do not establish primal attainment.
<!-- bilingual-en:end -->

- [Stanford EE364A, Lecture 9 transcript](https://see.stanford.edu/materials/lsocoee364a/transcripts/ConvexOptimizationI-Lecture09.html)：交叉核对强对偶与 KKT 证书的分工。
  逐字稿中原问题／对偶最优点的等号链推出互补松弛；随后区分凸 KKT 充分性与 Slater 支持的必要性。口头提及最优点时须另有其存在性，不能据此推出原问题一定取得最优值。

<!-- bilingual-en:start -->
The quadratic, exponential, domain-boundary, and failed-Slater examples are calculated explicitly in this card; their supremum and multiplier arguments are checked directly, rather than attributed as examples copied from the sources.
<!-- bilingual-en:end -->

二次、指数、定义域边界及 Slater 失败的例子均在正文直接计算，逐项核验上确界和乘子，没有将自推算例冒充来源中的例子。
