---
aliases:
  - '若固定选择集非空紧、目标连续且对参数凸可微、参数梯度联合连续，则最大值函数的方向导数等于当前最优解中最大的直接方向导数'
  - If the fixed choice set is nonempty and compact, the objective is continuous and convex-differentiable in the parameter, and its parameter gradient is jointly continuous, then the directional derivative of the maximum-value function is the largest direct directional derivative among the current optimizers
student_os: knowledge-atom
atom_id: OPT-MV-026
atom_set: multivariable-optimization
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[值函数]]"
  - "[[最优解对应]]"
  - "[[方向导数]]"
related:
  - "[[包络定理]]"
  - "[[影子价格]]"
leads_to:
  - "[[优化器跳跃不推价值不可微]]"
part_of:
  - "[[多元优化.canvas|多元优化]]"
---

# 若固定选择集非空紧、目标连续且对参数凸可微、参数梯度联合连续，则最大值函数的方向导数等于当前最优解中最大的直接方向导数
<!-- bilingual-en:start -->
*If the fixed choice set is nonempty and compact, the objective is continuous and convex-differentiable in the parameter, and its parameter gradient is jointly continuous, then the directional derivative of the maximum-value function is the largest direct directional derivative among the current optimizers*
<!-- bilingual-en:end -->

> [!summary] 定理说了什么
> 令 $X$ 为非空紧集，定义
> $$
> v(\theta)=\max_{x\in X}f(x,\theta),
> \qquad
> S(\theta)=\operatorname*{argmax}_{x\in X}f(x,\theta).
> $$
> 假设 $f$ 连续，$f(x,\cdot)$ 对每个 $x$ 凸且可微，并且 $\nabla_\theta f$ 联合连续。则对任意方向 $d$，
> $$
> v'(\theta;d)
> =\lim_{t\downarrow0}
> \frac{v(\theta+td)-v(\theta)}{t}
> =\max_{x\in S(\theta)}
> \nabla_\theta f(x,\theta)^Td.
> $$
> 只有当前最优解会决定价值的一阶方向变化。
>
> <!-- bilingual-en:start -->
> Let $X$ be nonempty and compact, with $v(\theta)=\max_{x\in X}f(x,\theta)$ and optimizer set $S(\theta)$. Suppose $f$ is continuous, each $f(x,\cdot)$ is convex and differentiable, and $\nabla_\theta f$ is jointly continuous. Then $v'(\theta;d)=\max_{x\in S(\theta)}\nabla_\theta f(x,\theta)^Td$. Only current optimizers determine the first-order directional change in the value.
> <!-- bilingual-en:end -->

## 为什么是“最大的”直接效应
<!-- bilingual-en:start -->
*Why the largest direct effect appears*
<!-- bilingual-en:end -->

在 $\theta$ 处，所有 $x\in S(\theta)$ 都给出同一当前价值。参数沿 $d$ 微小移动后，这些并列的最优选择可能产生不同的一阶增量。最大值函数会沿其中增加最快的那个分支移动，所以公式对活动最优解取最大。

<!-- bilingual-en:start -->
All points in $S(\theta)$ share the same current value. After a small movement in direction $d$, their direct first-order changes may differ. The maximum-value function follows the branch with the largest first-order increase, which explains the maximum over active optimizers.
<!-- bilingual-en:end -->

若 $S(\theta)$ 只有一个元素 $x^*$，则公式化为
$$
v'(\theta;d)=\nabla_\theta f(x^*,\theta)^Td.
$$
在上述凸参数设定中，这给出 $v$ 的普通梯度 $\nabla v(\theta)=\nabla_\theta f(x^*,\theta)$。定理不要求先求 $dx^*/d\theta$。

<!-- bilingual-en:start -->
With a unique optimizer $x^*$, the formula reduces to $v'(\theta;d)=\nabla_\theta f(x^*,\theta)^Td$, and the value is differentiable with gradient $\nabla_\theta f(x^*,\theta)$ in this convex-parameter setting. No derivative of the optimizer is needed.
<!-- bilingual-en:end -->

## 最小例子：两个活动最优解产生折点
<!-- bilingual-en:start -->
*Minimal example: two active optimizers create a kink*
<!-- bilingual-en:end -->

取 $X=\{-1,1\}$ 与 $f(x,\theta)=x\theta$。在 $\theta=0$ 时，
$$
S(0)=\{-1,1\},
$$
两个活动梯度分别为 $-1$ 和 $1$。因此
$$
v'(0;d)=\max\{-d,d\}=|d|.
$$
特别地，$v'(0;1)=v'(0;-1)=1$，但价值 $v(\theta)=|\theta|$ 在零点没有普通导数。

这里要区分两种记号：方向 $d=-1$ 的方向导数是 $v'(0;-1)=1$；按标量变量通常定义的左导数则是
$$
\lim_{h\uparrow0}\frac{v(h)-v(0)}{h}=-1.
$$
方向导数关于 $d$ 不是线性的，正是零点不存在一个普通梯度的信号。

<!-- bilingual-en:start -->
For $X=\{-1,1\}$ and $f(x,\theta)=x\theta$, both choices are optimal at zero and their parameter gradients are $-1$ and $1$. Hence $v'(0;d)=\max\{-d,d\}=|d|$, consistent with the kink in $v(\theta)=|\theta|$. The directional derivative at $d=-1$ is $1$, whereas the ordinary scalar left derivative is $-1$; failure of linearity in $d$ shows that no ordinary gradient exists at zero.
<!-- bilingual-en:end -->

## 公式的使用边界
<!-- bilingual-en:start -->
*Boundary of the formula*
<!-- bilingual-en:end -->

这里采用的版本要求选择集紧且不随参数变化，并明确假设目标对参数凸且可微。若选择集 $X(\theta)$ 随参数变化，约束的直接参数效应必须通过相应的包络或扰动定理处理。若选择集不紧或最优值不取得，则 $S(\theta)$ 可能为空，上式也失去取最大的对象。

<!-- bilingual-en:start -->
This version assumes a compact parameter-independent choice set and convex differentiable dependence on the parameter. Parameter-dependent constraints require the corresponding constrained envelope or perturbation theorem. Without compactness or attainment, the optimizer set may be empty and the active-optimizer formula cannot be used as written.
<!-- bilingual-en:end -->

> [!question]- 自检
> 在多个当前最优解中，为什么不能随便挑一个 $\nabla_\theta f$ 报告为价值的方向导数？
>
> <!-- bilingual-en:start -->
> With several current optimizers, why can one not report an arbitrary $\nabla_\theta f$ as the value's directional derivative?
> <!-- bilingual-en:end -->
>
> **答案：** 不同活动最优解可以有不同的直接一阶效应；最大值函数在该方向沿其中最大的一个变化。
>
> <!-- bilingual-en:start -->
> **Answer:** Active optimizers can have different direct first-order effects. The maximum-value function follows the largest of those effects in the chosen direction.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Bertsekas, *Convex Optimization Theory: A Summary*, Proposition 5.4.9](https://faculty.engineering.asu.edu/bertsekas/wp-content/uploads/sites/129/2019/10/convexdualitycondenced.pdf)：直接核对紧选择集、参数凸性、活动最优解与方向导数取最大的 Danskin 版本。
- [Milgrom and Segal, “Envelope Theorems for Arbitrary Choice Sets”](https://web.stanford.edu/~milgrom/publishedarticles/Envelope%20Theorems.pdf)：为经济学中更一般的最优值单侧导数提供补充边界。

<!-- bilingual-en:start -->
- Bertsekas, *Convex Optimization Theory: A Summary*, Proposition 5.4.9 was checked for the compact-set convex-parameter version of Danskin's theorem and its active-maximizer directional derivative.
- Milgrom and Segal, “Envelope Theorems for Arbitrary Choice Sets,” was used only to locate this smooth compact result within the broader economic envelope framework.
<!-- bilingual-en:end -->
