---
aliases:
  - 给定储蓄率与家庭最优化是 AK 模型的两种不同闭合方式
  - Exogenous saving versus optimising AK closure
student_os: knowledge-atom
atom_id: MACRO-ENDO-007
atom_type: distinction
status: source-checked
part_of:
  - "[[内生增长与创新.canvas]]"
---

# 给定储蓄率与家庭最优化是 AK 模型的两种不同闭合方式
<!-- bilingual-en:start -->
*An exogenous saving rate and household optimisation are two distinct ways to close the AK model.*
<!-- bilingual-en:end -->

[[AK模型]] 的技术和资源约束只告诉我们可以生产多少、消费后还剩多少用于积累；**闭合方式**补上消费与投资怎样决定的关系。给定 $s$ 时，家庭按规定比例储蓄；最优化时，家庭依偏好与回报选择消费，储蓄率成为结果。两种模型可以有相同的 $Y=AK$，却有不同的增长率决定式。
<!-- bilingual-en:start -->
[[AK模型|AK technology]] and resource constraints determine production and feasible accumulation after consumption. Model closure supplies how consumption and investment are chosen. With exogenous $s$, households save a prescribed share; under optimisation, preferences and returns determine consumption and saving becomes an outcome. The same $Y=AK$ can therefore support different growth-rate equations.
<!-- bilingual-en:end -->

两种写法都采用连续时间、封闭单商品经济，固定 $A>0$、折旧率 $\delta\ge0$、人口增长率 $n\ge0$，并令 $k=K/L$、$c=C/L$。人均资源约束相同：
<!-- bilingual-en:start -->
Both formulations use continuous time, a closed single-good economy, fixed $A>0$, depreciation $\delta\ge0$ and population growth $n\ge0$, with $k=K/L$ and $c=C/L$. Their per-capita resource constraint is the same:
<!-- bilingual-en:end -->

$$
\dot k=(A-\delta-n)k-c,\qquad k(0)=k_0>0.
$$

**常储蓄率闭合：** 给定 $0<s<1$，令 $c=(1-s)Ak$，代入立即得到 $\dot k/k=sA-\delta-n$，见 [[AK人均增长条件]]。这里没有效用最大化问题；$s$ 是输入，不能仅因路径可行就称其最优。
<!-- bilingual-en:start -->
**Constant-saving closure:** Given $0<s<1$, set $c=(1-s)Ak$. Substitution gives $\dot k/k=sA-\delta-n$; see [[AK人均增长条件|the AK growth condition]]. No utility-maximisation problem has been specified: $s$ is an input, and feasibility alone does not establish optimality.
<!-- bilingual-en:end -->

**家庭最优化闭合：** 采用人口加权的无限寿命代表家庭，最大化下式，其中 $\rho>n$ 为纯时间贴现率，$\theta>0$ 为 CRRA 曲率，$1/\theta$ 为跨期替代弹性；$\theta=1$ 时采用对数效用极限。
<!-- bilingual-en:start -->
**Optimising closure:** An infinitely lived representative household maximises population-weighted utility below. Here $\rho>n$ is pure time preference, $\theta>0$ is CRRA curvature and $1/\theta$ is the intertemporal elasticity of substitution. At $\theta=1$, use the logarithmic limit.
<!-- bilingual-en:end -->

$$
\max_{c(t)}\int_0^\infty e^{-(\rho-n)t}
\frac{c(t)^{1-\theta}-1}{1-\theta}\,dt,
\qquad
\dot k=(A-\delta-n)k-c.
$$

在无外部性、竞争资本市场的纯 AK 中，净利率为 $r=A-\delta$。结合预算与无庞氏约束，最优消费满足 Euler 方程；相应最优路径还须满足终端条件：
<!-- bilingual-en:start -->
In pure AK with competitive capital markets and no externalities, the net interest rate is $r=A-\delta$. Together with the budget and no-Ponzi constraint, optimal consumption obeys the Euler equation; the optimal path must also satisfy the terminal condition:
<!-- bilingual-en:end -->

$$
\frac{\dot c}{c}=\frac{A-\delta-\rho}{\theta}\equiv g^*,
\qquad
\lim_{t\to\infty}k(t)e^{-(A-\delta-n)t}=0.
$$

Euler 方程只确定消费增长率，还没有给出可行的初始消费。令 $q=A-\delta-n-g^*$；在 $g^*>0$ 且 $q>0$ 时，预算和终端条件把路径定为
<!-- bilingual-en:start -->
The Euler equation fixes consumption growth, but not a feasible initial consumption level. Define $q=A-\delta-n-g^*$. When $g^*>0$ and $q>0$, the budget and terminal condition select:
<!-- bilingual-en:end -->

$$
k(t)=k_0e^{g^*t},\qquad c(t)=qk(t),\qquad c(0)=qk_0,
\qquad
s^*=\frac{\dot K+\delta K}{Y}
=\frac{g^*+n+\delta}{A}.
$$

可以直接检验：将 $c=qk$ 代回资源约束得到 $\dot k=g^*k$；终端式变为 $k_0e^{-qt}\to0$。$q>0$ 保证正消费和这条路径的终端条件，且上述条件给出 $0<s^*<1$。优化版的人均 $k,c,y$ 按 $g^*$ 增长，总量 $K,C,Y$ 则按 $g^*+n$ 增长。
<!-- bilingual-en:start -->
Substituting $c=qk$ into the resource constraint gives $\dot k=g^*k$, and the terminal expression becomes $k_0e^{-qt}\to0$. Positive $q$ ensures positive consumption and the terminal condition on this path; the stated restrictions also give $0<s^*<1$. Per-capita $k,c,y$ grow at $g^*$, while aggregate $K,C,Y$ grow at $g^*+n$.
<!-- bilingual-en:end -->

这里 $g^*$ 没有显式的 $n$，与所采用的**人口加权效用**有关；人口项仍在资源约束、初始消费和储蓄率中。若改变效用中的人口权重，Euler 方程也须重新推导。也不能一面使用优化结果，一面又独立指定任意 $s$。
<!-- bilingual-en:start -->
The absence of an explicit $n$ in $g^*$ depends on the population weighting of utility. Population still enters resources, initial consumption and saving. A different welfare weighting requires rederiving the Euler equation. Nor can an arbitrary independent $s$ be imposed alongside the optimising solution.
<!-- bilingual-en:end -->

构造例子：$A=0.10$、$\delta=0.03$、$n=0.01$、$\rho=0.05$、$\theta=2$，得到 $g^*=0.01$、$q=0.05$、$s^*=0.50$。若 $k_0=10$，则 $y_0=1$、$c_0=0.5$、$\dot k_0=0.1$。资源检查为 $(0.10-0.03-0.01)\times10-0.5=0.1$，各条件成立。
<!-- bilingual-en:start -->
For a constructed example, $A=0.10$, $\delta=0.03$, $n=0.01$, $\rho=0.05$ and $\theta=2$ give $g^*=0.01$, $q=0.05$ and $s^*=0.50$. At $k_0=10$, output is $y_0=1$, consumption $c_0=0.5$ and accumulation $\dot k_0=0.1$. The resource check is $(0.10-0.03-0.01)\times10-0.5=0.1$, and the parameter restrictions hold.
<!-- bilingual-en:end -->

## 来源与核验

- [Acemoglu，MIT 14.452，2016 Lectures 2–3](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2b68057aa4e74410d00ae89a0c49752f_MIT14_452F16_Lec2and3.pdf#page=57)，PDF第57–59页：支持外生常储蓄率闭合的 AK 路径。
- [Acemoglu，MIT 14.452，2016 Lecture 8](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/6f068789dff6c81a6ea2deee055829d8_MIT14_452F16_Lec8.pdf#page=3)，PDF第3–8页，式(1)–(12)：支持人口加权 CRRA 效用、预算、无庞氏与终端条件、净回报及 Euler 方程；第9–13页，式(13)–(17)：支持从终端条件选出路径、初始消费和内生储蓄率。$q$ 是原式(16)中系数的简写，正增长与 $q>0$ 对应原式(12)的两项严格不等式。数例及资源代入独立核算。
<!-- bilingual-en:start -->
Lectures 2–3 supply exogenous-saving AK. Lecture 8 supplies the utility weighting, constraints, Euler equation, terminal selection and endogenous saving. The coefficient $q$ rewrites equation (16), and the two growth restrictions rewrite equation (12). The example and resource substitution are independently checked.
<!-- bilingual-en:end -->
