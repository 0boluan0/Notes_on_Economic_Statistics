# DPDE 第一讲：消费储蓄、乘子与欧拉方程

<!-- bilingual-en:start -->
*DPDE Lecture 1: Consumption, Saving, Multipliers, and the Euler Equation*
<!-- bilingual-en:end -->

**本次学到的核心：在可自由调整储蓄的内点最优路径上，今天消费一单位的边际效用，等于把它存到明天所带来的折现边际效用。**

<!-- bilingual-en:start -->
**The key result so far: along an interior optimal path with freely adjustable saving, the marginal utility of consuming one unit today equals the discounted marginal utility obtained by saving it for tomorrow.**
<!-- bilingual-en:end -->

$$
\boxed{u'(C_t)=\beta R\,u'(C_{t+1})}
$$

复习入口：[[#六、λ 是什么：先理解报价，再写拉格朗日函数|λ 的含义]] · [[#七、三期模型：同一笔储蓄为何产生两项导数|资产求导的两项]] · [[#九、易错点清单|易错点]] · [[#十、本次复习的完成条件|白纸复习]]。

## 一、这部分在整门课中的位置

<!-- bilingual-en:start -->
*Where this material fits in the course*
<!-- bilingual-en:end -->

动态优化（dynamic optimization）选择的是一条随时间展开的决策路径。方法可以按「时间如何表示」和「如何组织求解」分类：

<!-- bilingual-en:start -->
Dynamic optimization chooses a sequence of decisions over time. Methods can be classified by how time is represented and how the problem is organized:
<!-- bilingual-en:end -->

| 求解方式 / Approach | 离散时间 / Discrete time | 连续时间 / Continuous time |
|---|---|---|
| 序贯法 / Sequential | 拉格朗日函数 / Lagrangian | 哈密顿函数 / Hamiltonian |
| 递归法 / Recursive | 贝尔曼方程 / Bellman equation | HJB 方程 / HJB equation |

Lecture 1 讨论第一行，本次停在左上格。**有限期、确定性**的动态问题可以看作高维静态问题：把 $C_0,C_1,\ldots,C_T$ 看成不同日期交付的商品，在第 0 期选择整条路径。递归法把问题改写为「给定当前状态，今天怎么选」；它属于后续 **Lecture 3**。

<!-- bilingual-en:start -->
Lecture 1 covers the first row; this session covers its discrete-time entry. A **finite-horizon, deterministic** problem can be treated as a high-dimensional static problem: consumption at different dates represents different dated goods, and the entire path is chosen at date 0. The recursive approach instead asks what to choose today given the current state; it is developed in **Lecture 3**.
<!-- bilingual-en:end -->

> [!note] 方法的适用边界
> 无限期和不确定性会让求解更困难，但不能概括为「静态／序贯方法失效」。尤其在不确定性下，通常不能仅靠欧拉方程把下一期消费直接解出来，再逐期代入预算。这是后续学习其他求解方法的动机。
>
> <!-- bilingual-en:start -->
> Infinite horizons and uncertainty make the problem harder; they do not make sequential methods invalid. Under uncertainty, one generally cannot isolate next-period consumption from the Euler equation and substitute it forward through the budget constraints. This motivates the later solution methods.
> <!-- bilingual-en:end -->

### 符号与假设分别负责什么

<!-- bilingual-en:start -->
*Notation and the role of each assumption*
<!-- bilingual-en:end -->

| 符号 / Symbol | 含义 / Meaning |
|---|---|
| $C_t$ | 第 $t$ 期消费 / Consumption at date $t$ |
| $u(C_t)$、$u'(C_t)$ | 当期效用、边际效用 / Period utility and marginal utility |
| $\beta$ | 贴现因子；越小越不耐心 / Discount factor; a smaller value means less patience |
| $R=1+r$ | 毛利率，即本金加利息的回报倍数；$r$ 是净利率 / Gross return; $r$ is the net interest rate |
| $Y_t$ | 外生收入 / Exogenous income |
| $B_t$ | 按讲义约定，上期留下、在本期取得 $RB_t$ 的本金 / Assets carried into date $t$, paying $RB_t$ at that date |
| $\lambda_t$ | 现值乘子，以第 0 期效用衡量第 $t$ 期资源的边际价值 / Present-value multiplier for date-$t$ resources |
| $\sigma$ | CRRA 曲率参数；跨期替代弹性为 $1/\sigma$ / CRRA curvature parameter; the EIS is $1/\sigma$ |

下文的内点推导取 $R>0$、$0<\beta<1$，并假定可用总资源为正、各期之间可以借贷转移资源。讲义写 $\beta\in[0,1)$；其中 $\beta=0$ 是完全不重视未来的特殊情形，不能直接套用各期消费均为正的推导。最开始「不贴现」的两期练习取 $\beta=1$，有限期这样做没有问题。

<!-- bilingual-en:start -->
The interior derivations below use $R>0$ and $0<\beta<1$, positive total resources, and borrowing and saving between dates. The lecture states $\beta\in[0,1)$, but $\beta=0$ is a degenerate case with no value attached to future utility, so the positive-consumption derivation does not apply directly. The initial finite-horizon exercise uses $\beta=1$, which is valid when there is no discounting.
<!-- bilingual-en:end -->

| 假设 | 在本次模型中的作用 | 需要分清的条件 |
|---|---|---|
| $u$ 递增，例题中 $u'>0$ | 有资源时愿意多消费；没有遗赠动机时，最后留着可消费的财富没有收益 | 必须结合终端不能欠债的限制，才能说明最后最优资产为零 |
| $u$ 凹（concave） | 目标函数凹、可行集凸时，满足可行性和适当的一阶条件的解是全局最优解 | 不能只求导而不检查约束 |
| $u$ 严格凹，例题中 $u''<0$ | 边际效用严格递减，因此 $u'(C_t)=u'(C_{t+1})$ 才能唯一推出消费相等 | 普通凹性只保证边际效用不增加，未必可逆 |
| Inada 条件：$\lim_{C\to0^+}u'(C)=\infty$ | 在资源可转移、贴现权重为正等条件下，帮助排除零消费角点 | 它不能凭空创造资源，也不能消除额外借贷约束 |

<!-- bilingual-en:start -->
	Increasing utility makes additional feasible consumption desirable; zero terminal assets also require the no-terminal-debt restriction and no bequest motive. Concavity makes the appropriate first-order conditions sufficient on a convex feasible set. **Strict** concavity makes marginal utility strictly decreasing, so equal marginal utilities imply equal consumption. The Inada condition helps rule out zero consumption when resources can be transferred and every period has positive weight; it does not remove resource or borrowing restrictions.
<!-- bilingual-en:end -->

这里要把凹性的两项用途分开：**「验证最优」与「由边际效用比较消费」是两件事。** 另外，$\beta<1$ 表示未来效用被折现；它本身并不保证任意无限期效用总和收敛。本次无需处理该收敛问题。

<!-- bilingual-en:start -->
Keep the two uses of concavity separate: **establishing optimality and translating marginal-utility comparisons into consumption comparisons are different tasks.** Also, $\beta<1$ discounts future utility but does not by itself guarantee convergence of every infinite-horizon utility sum. That issue is outside this session.
<!-- bilingual-en:end -->

## 二、从两期问题一步步搭起来

<!-- bilingual-en:start -->
*Building the two-period problem step by step*
<!-- bilingual-en:end -->

> [!example] 理解辅助：三道递进练习
> 以下使用第 0 期资源 100、第 1 期收入为零的例子，来自辅导过程。先理解凹性，再分别加入耐心和储蓄回报。$u(C)=\sqrt C$ 是统一的计算例子。
>
> <!-- bilingual-en:start -->
> These tutoring exercises assume 100 units of resources at date 0 and no income at date 1. They introduce concavity, patience, and the return to saving in that order, using $u(C)=\sqrt C$ throughout.
> <!-- bilingual-en:end -->

### 1. 无贴现、无利息：为什么会平分

<!-- bilingual-en:start -->
*No discounting or interest: why split resources equally?*
<!-- bilingual-en:end -->

$$
\max_{C_0,C_1\ge0}\ u(C_0)+u(C_1)
\qquad\text{s.t.}\quad C_0+C_1=100.
$$

将 $C_1=100-C_0$ 代入后，只需选 $C_0$。在内点最优解处，往今天多挪一点与往明天多挪一点，都不能带来一阶收益，所以：

<!-- bilingual-en:start -->
Substituting $C_1=100-C_0$ leaves one choice variable. At an interior optimum, shifting a small amount in either direction cannot deliver a first-order gain:
<!-- bilingual-en:end -->

$$
\frac{d}{dC_0}\big[u(C_0)+u(100-C_0)\big]
=u'(C_0)-u'(C_1)=0.
$$

若 $u$ 严格凹，消费少的那一期边际效用更高；从消费多的一期挪钱过去会提高总效用，直到两期边际效用相等。对 $u(C)=\sqrt C$，得到 $C_0=C_1=50$。这里的平分同时依赖**相同效用函数、相同权重、无利息**；凹性并不意味着任何跨期问题都平分。

<!-- bilingual-en:start -->
With strict concavity, the period with lower consumption has higher marginal utility. Moving resources toward it raises total utility until marginal utilities are equal. Square-root utility gives $C_0=C_1=50$. Equal division also relies on identical utility functions, equal weights, and no interest; concavity alone does not imply equal consumption in every intertemporal problem.
<!-- bilingual-en:end -->

### 2. 加入 β：未来效用要打折

<!-- bilingual-en:start -->
*Adding patience: discount future utility by beta*
<!-- bilingual-en:end -->

$$
\max\ u(C_0)+\beta u(C_1)
\qquad\text{s.t.}\quad C_0+C_1=100.
$$

$$
u'(C_0)=\beta u'(C_1).
$$

现在比较的是「今天的边际效用」与「折现后的明天边际效用」。对平方根效用：

<!-- bilingual-en:start -->
The comparison is now between today's marginal utility and tomorrow's discounted marginal utility. With square-root utility:
<!-- bilingual-en:end -->

$$
\frac{1}{2\sqrt{C_0}}=\frac{\beta}{2\sqrt{C_1}}
\quad\Longrightarrow\quad C_1=\beta^2C_0
\quad\Longrightarrow\quad
C_0=\frac{100}{1+\beta^2}.
$$

$\beta$ 越小，越不耐心，今天消费越多。

<!-- bilingual-en:start -->
A smaller $\beta$ means less patience and more consumption today.
<!-- bilingual-en:end -->

### 3. 加入 R：一单位储蓄换来 R 单位未来消费

<!-- bilingual-en:start -->
*Adding returns: one unit saved finances R units of future consumption*
<!-- bilingual-en:end -->

$$
C_1=R(100-C_0)
\quad\Longleftrightarrow\quad
C_0+\frac{C_1}{R}=100.
$$

$$
\frac{d}{dC_0}\big[u(C_0)+\beta u(R(100-C_0))\big]
=u'(C_0)-\beta R\,u'(C_1)=0.
$$

这就是两期欧拉方程。对平方根效用：

<!-- bilingual-en:start -->
This is the two-period Euler equation. With square-root utility:
<!-- bilingual-en:end -->

$$
\sqrt{C_1}=\beta R\sqrt{C_0}
\quad\Longrightarrow\quad C_1=\beta^2R^2C_0,
$$

$$
C_0+\frac{\beta^2R^2C_0}{R}=100
\quad\Longrightarrow\quad
\boxed{C_0=\frac{100}{1+\beta^2R}}.
$$

> [!warning] 手写笔记中的链式法则难点
> 外层与内层都要求导，尤其别漏掉平方根导数中的 $1/2$：
>
> <!-- bilingual-en:start -->
> Differentiate both layers, including the factor $1/2$ from the square root:
> <!-- bilingual-en:end -->
>
> $$
> \frac{d}{dC_0}\left[\beta\sqrt{R(100-C_0)}\right]
> =\beta\frac{1}{2\sqrt{R(100-C_0)}}(-R)
> =-\frac{\beta\sqrt R}{2\sqrt{100-C_0}}.
> $$
>
> 内层贡献 $R$，外层分母含 $\sqrt R$，化简后剩 $\sqrt R$。但整理成 $u'$ 的通用形式时，始终是 $u'(C_0)=\beta R u'(C_1)$。
>
> <!-- bilingual-en:start -->
> The inner derivative contributes $R$, while the outer derivative has $\sqrt R$ in its denominator, leaving $\sqrt R$ after simplification. In the general marginal-utility notation, the condition remains $u'(C_0)=\beta R u'(C_1)$.
> <!-- bilingual-en:end -->

## 三、欧拉方程在说什么

<!-- bilingual-en:start -->
*What the Euler equation says*
<!-- bilingual-en:end -->

欧拉方程（Euler equation, EE）是**相邻两期之间的边际最优条件**。这个名称与变分法中选择最优路径的 Euler 条件有关；当前最有用的读法是：

<!-- bilingual-en:start -->
The Euler equation is a **marginal optimality condition linking adjacent periods**. Its name is associated with the Euler conditions for optimal paths in the calculus of variations. For this model, read it as follows:
<!-- bilingual-en:end -->

$$
\underbrace{u'(C_t)}_{\text{今天消费一单位的边际收益}}
=\underbrace{\beta}_{\text{折现}}
\underbrace{R}_{\text{储蓄回报}}
\underbrace{u'(C_{t+1})}_{\text{明天每单位消费的边际价值}}.
$$

**在内点最优路径上，把一点资源从今天挪到明天，不赚也不亏。** 这里的「不赚不亏」指效用的一阶变化为零。

<!-- bilingual-en:start -->
**At an interior optimum, shifting a small amount of resources from today to tomorrow produces no first-order utility gain or loss.** The three factors on the right are discounting, the gross return, and tomorrow's marginal utility.
<!-- bilingual-en:end -->

### 扰动法：保持后面的路径不动

<!-- bilingual-en:start -->
*Perturbation: leave the later path unchanged*
<!-- bilingual-en:end -->

暂时把 $C_t$ 增加 $\varepsilon$，就要把本期储蓄 $B_{t+1}$ 减少 $\varepsilon$。下一期可用资源因此减少 $R\varepsilon$；让 $C_{t+1}$ 同样减少 $R\varepsilon$，便能保持 $B_{t+2}$ 不变，之后回到原路径。

<!-- bilingual-en:start -->
Increase $C_t$ by $\varepsilon$ and reduce saving $B_{t+1}$ by the same amount. Next-period resources fall by $R\varepsilon$. Reducing $C_{t+1}$ by $R\varepsilon$ keeps $B_{t+2}$ unchanged, so the path returns to its original course thereafter.
<!-- bilingual-en:end -->

$$
dU=\beta^t u'(C_t)\varepsilon
-\beta^{t+1}u'(C_{t+1})R\varepsilon.
$$

当足够小的正、负 $\varepsilon$ 都可行时，最优要求 $\varepsilon$ 的系数为零，约去 $\beta^t$ 就得到 EE。**若额外借贷约束正在绑定，双向扰动未必可行，不能机械套用等号。**

<!-- bilingual-en:start -->
If sufficiently small positive and negative perturbations are both feasible, optimality requires the coefficient of $\varepsilon$ to vanish. Dividing by $\beta^t$ gives the Euler equation. **If an additional borrowing constraint binds, a two-way perturbation may be unavailable, so the equality need not apply.**
<!-- bilingual-en:end -->

### 形状与水平是两件事

<!-- bilingual-en:start -->
*The path's shape and its level are different*
<!-- bilingual-en:end -->

在边际效用严格递减的条件下，EE 可以比较两期消费：

<!-- bilingual-en:start -->
When marginal utility is strictly decreasing, the Euler equation orders adjacent consumption levels:
<!-- bilingual-en:end -->

| 条件 / Condition | 边际效用 / Marginal utility | 消费路径 / Consumption path |
|---|---|---|
| $\beta R=1$ | $u'(C_t)=u'(C_{t+1})$ | $C_{t+1}=C_t$，完全消费平滑 / Constant consumption |
| $\beta R>1$ | $u'(C_t)>u'(C_{t+1})$ | $C_{t+1}>C_t$，递增 / Increasing |
| $\beta R<1$ | $u'(C_t)<u'(C_{t+1})$ | $C_{t+1}<C_t$，递减 / Decreasing |

> [!tip] 欧拉方程给形状，预算约束给水平
> EE 给出相邻消费的关系；要确定究竟消费多少，还要结合资源与边界条件。手写笔记中的「EE 不受预算影响」可以理解为：本模型 EE 的表达式没有显式出现财富或收入。实际消费选择仍受预算限制。一般效用下，EE 也未必给出一个固定增长率；CRRA 才有本次见到的简洁比例形式。
>
> <!-- bilingual-en:start -->
> **The Euler equation gives the shape; the budget gives the level.** The equation relates adjacent consumption levels; resources and boundary conditions are also needed to determine their values. Wealth and income do not appear explicitly in this model's Euler equation, but consumption still depends on the budget. A constant growth factor is a feature of CRRA utility here, not of every utility function.
> <!-- bilingual-en:end -->

按讲义的比值方向，EE 也可写为边际替代率等于边际转换率（MRS = MRT）：

<!-- bilingual-en:start -->
Using the ratio orientation in the lecture, the Euler equation can also be written as MRS = MRT:
<!-- bilingual-en:end -->

$$
\frac{\beta u'(C_{t+1})}{u'(C_t)}=\frac{1}{R}.
$$

两边都以「一单位明天消费对应多少今天消费」为单位。如果画图时以 $C_0$ 为横轴、$C_1$ 为纵轴，斜率绝对值用倒数形式 $u'(C_0)/(\beta u'(C_1))=R$；不要把一个方向的 MRS 和另一个方向的 MRT 配在一起。

<!-- bilingual-en:start -->
Both sides are measured in units of today's consumption per unit of tomorrow's consumption. With $C_0$ on the horizontal axis and $C_1$ on the vertical axis, absolute slopes use the reciprocal form, $u'(C_0)/(\beta u'(C_1))=R$. Keep the orientation consistent.
<!-- bilingual-en:end -->

## 四、R 上升：为什么今天消费不一定减少

<!-- bilingual-en:start -->
*Why a higher return does not always reduce consumption today*
<!-- bilingual-en:end -->

> [!example] 理解辅助：仍用两期、初始资源 100 的例子
> 本节的比较静态固定第 0 期资源为 100、第 1 期收入为零。它解释手写笔记中的预算线旋转、收入效应和替代效应。
>
> <!-- bilingual-en:start -->
> This two-period comparative-static exercise holds date-0 resources at 100 and date-1 income at zero. It develops the budget-line rotation, income effect, and substitution effect in the handwritten notes.
> <!-- bilingual-en:end -->

从 $C_0+C_1/R=100$ 可以读出相对价格：今天消费的价格是 1，明天消费的价格是 $1/R$。所以 $R$ 上升时，**未来消费相对降价**。

<!-- bilingual-en:start -->
In $C_0+C_1/R=100$, today's consumption has price 1 and tomorrow's has price $1/R$. A higher $R$ therefore makes **future consumption relatively cheaper**.
<!-- bilingual-en:end -->

预算线为 $C_1=R(100-C_0)$。以 $C_0$ 为横轴、$C_1$ 为纵轴，它的斜率是 $-R$，横截距始终为 100，纵截距为 $100R$。$R$ 上升时，预算线绕禀赋点（endowment point）$(100,0)$ 向外旋转。

<!-- bilingual-en:start -->
With $C_0$ horizontal and $C_1$ vertical, the budget line $C_1=R(100-C_0)$ has slope $-R$, a fixed horizontal intercept of 100, and a vertical intercept of $100R$. Raising $R$ rotates it outward around the endowment point $(100,0)$.
<!-- bilingual-en:end -->

| 效应 | 比较的是什么 | 在本例中对 $C_0$ 的方向 |
|---|---|---|
| 替代效应（substitution effect） | 在补偿购买力后，未来消费相对变便宜 | 转向未来消费，$C_0\downarrow$ |
| 收入效应（income effect） | 相对价格变化也提高了本例储蓄者的实际购买力 | 对本节 CRRA／log 偏好，今天消费是正常品，$C_0\uparrow$ |

<!-- bilingual-en:start -->
The substitution effect compares choices after compensating for the change in purchasing power: cheaper future consumption reduces $C_0$. The income effect reflects the gain in purchasing power for the saver in this example: under the CRRA and log preferences used here, current consumption is a normal good, so this effect raises $C_0$.
<!-- bilingual-en:end -->

「变陡」帮助识别相对价格变化，「向外」帮助理解购买力变化；一条新预算线同时包含两种作用，严格分解时还需要补偿预算线。**今天消费的总变化，要看两种效应谁更强。**

<!-- bilingual-en:start -->
The steeper slope identifies the relative-price change, while the outward rotation helps explain the purchasing-power change. The new budget line contains both effects; a formal decomposition also requires a compensated budget line. **The net change in current consumption depends on which effect dominates.**
<!-- bilingual-en:end -->

### 用 CRRA 曲率看两种效应的强弱

<!-- bilingual-en:start -->
*Using CRRA curvature to compare the two effects*
<!-- bilingual-en:end -->

不变相对风险厌恶效用（constant relative risk aversion, CRRA）取：

<!-- bilingual-en:start -->
Use constant relative risk aversion utility:
<!-- bilingual-en:end -->

$$
u(C)=\frac{C^{1-\sigma}}{1-\sigma},\qquad
\sigma>0,\quad\sigma\ne1;
\qquad u(C)=\log C\ \text{when }\sigma=1.
$$

其边际效用为 $u'(C)=C^{-\sigma}$。将它代入两期 EE，再代入两期预算：

<!-- bilingual-en:start -->
Its marginal utility is $u'(C)=C^{-\sigma}$. Substitute it into the two-period Euler equation and then the two-period budget:
<!-- bilingual-en:end -->

$$
C_0^{-\sigma}=\beta R C_1^{-\sigma}
\quad\Longrightarrow\quad
\frac{C_1}{C_0}=(\beta R)^{1/\sigma},
$$

$$
\boxed{C_0=\frac{100}{1+\beta^{1/\sigma}R^{(1-\sigma)/\sigma}}}.
$$

| 曲率 / Curvature | $R$ 的指数 / Exponent of $R$ | $R\uparrow$ 时的 $C_0$ / Response | 解释 / Interpretation |
|---|---|---|---|
| $0<\sigma<1$ | 正 / Positive | 减少 / Falls | 替代效应占优 / Substitution dominates |
| $\sigma=1$ | 零 / Zero | 不变 / Unchanged | 两种效应抵消 / Effects offset |
| $\sigma>1$ | 负 / Negative | 增加 / Rises | 收入效应占优 / Income dominates |

$\sqrt C$ 与 $\sigma=1/2$ 的 CRRA 相差一个正的常数倍，因此消费选择相同，但效用数值和 λ 的数值会随效用单位改变。跨期替代弹性（elasticity of intertemporal substitution, EIS）是 **$1/\sigma$**：$\sigma$ 越大，同样的 $\beta R$ 变化引起的消费比率反应越小。

<!-- bilingual-en:start -->
Square-root utility differs from CRRA with $\sigma=1/2$ only by a positive scale factor, so it gives the same consumption choices; utility and multiplier values depend on that scale. The elasticity of intertemporal substitution is **$1/\sigma$**: a larger $\sigma$ means a smaller consumption-ratio response to the same change in $\beta R$.
<!-- bilingual-en:end -->

### log 情形：R 为什么正好消失

<!-- bilingual-en:start -->
*Log utility: why the return cancels*
<!-- bilingual-en:end -->

$$
\frac{1}{C_0}=\frac{\beta R}{C_1}
\quad\Longrightarrow\quad C_1=\beta RC_0.
$$

$$
\beta RC_0=R(100-C_0)
\quad\Longrightarrow\quad
\boxed{C_0=\frac{100}{1+\beta}}.
$$

若用代入法，消失得更直接：

<!-- bilingual-en:start -->
Direct substitution makes the cancellation equally clear:
<!-- bilingual-en:end -->

$$
\frac{d}{dC_0}\left[\beta\log\big(R(100-C_0)\big)\right]
=\frac{\beta}{R(100-C_0)}(-R)
=-\frac{\beta}{100-C_0}.
$$

因此，「链式法则会带来 $R$」与「化简后必须保留 $R$」不是一回事。这里 $C_0$ 不随 $R$ 变，是**固定初始资源、无未来收入的这个例子**的结论；若财富现值本身随 $R$ 改变，不能照搬。

<!-- bilingual-en:start -->
The chain rule introduces a factor of $R$, but that factor need not survive simplification. The independence of $C_0$ from $R$ is a result for **this example with fixed initial resources and no future income**. It cannot be transferred unchanged to a setting where the present value of wealth varies with $R$.
<!-- bilingual-en:end -->

## 五、从两期到多期：B_t 在记什么账

<!-- bilingual-en:start -->
*From two periods to many: what assets record*
<!-- bilingual-en:end -->

两期例子里，选完 $C_0$，剩下的钱自动决定 $C_1=R(100-C_0)$。多期时，每一期都要知道「从过去带来了多少资源」，于是用资产 $B_t$ 记录历史留下的结果。这就是状态变量（state variable）的作用：在给定未来外生收入等信息后，为当前决策概括相关历史。$C_t$ 则是本期选择的控制变量（control variable）。

<!-- bilingual-en:start -->
In the two-period example, choosing $C_0$ determines $C_1$ from the remaining resources. With many periods, each date needs a record of what was carried forward. Assets $B_t$ are a state variable: together with the relevant exogenous information, they summarize the history needed for current decisions. Consumption $C_t$ is a control variable chosen at that date.
<!-- bilingual-en:end -->

讲义的逐期预算约束（flow budget constraint）是：

<!-- bilingual-en:start -->
The lecture uses the following flow budget constraint:
<!-- bilingual-en:end -->

$$
\boxed{B_{t+1}=RB_t+Y_t-C_t}.
$$

读作：**本期末留下的本金 = 上期本金连本带利 + 本期收入 − 本期消费。** $B_{t+1}>0$ 表示储蓄，$B_{t+1}<0$ 表示负债；基准模型允许中间各期借贷，但要求终点不能留下未偿债务。手写笔记中的常数 $Y$ 是 $Y_t$ 不随时间变化的特例。

<!-- bilingual-en:start -->
Read this as: **assets left at the end of the period equal inherited principal plus its return, plus current income, minus current consumption.** Positive assets mean saving and negative assets mean debt. The baseline model permits borrowing at intermediate dates but requires no unpaid debt at the terminal date. The constant $Y$ in the handwritten notes is the special case of time-invariant $Y_t$.
<!-- bilingual-en:end -->

### 记账约定 A 与 B

<!-- bilingual-en:start -->
*Accounting conventions A and B*
<!-- bilingual-en:end -->

| 比较项 | A：讲义约定 | B：另一种常见约定 |
|---|---|---|
| 预算式 | $B_{t+1}=RB_t+Y_t-C_t$ | $A_{t+1}=R(A_t+Y_t-C_t)$ |
| 资产符号记录的时点 | 本期计息前的本金 $B_t$ | 本期已经可用的财富 $A_t$ |
| 本期可用资源 | $RB_t+Y_t$ | $A_t+Y_t$ |
| 新储蓄的利息放在哪里 | 到下一期的 $RB_{t+1}$ 中体现 | 已计入下一期财富 $A_{t+1}$ |
| 跨期预算中初始资产贡献 | $RB_0$ | $A_0$；若教材也叫它 $B_0$，则写 $B_0$ |

<!-- bilingual-en:start -->
Under convention A, $B_t$ records principal before its date-$t$ return is paid; available resources are $RB_t+Y_t$. Under convention B, $A_t$ is already available wealth, and the return on newly saved resources is included in $A_{t+1}$. The initial-asset contribution to a present-value budget is consequently $RB_0$ in A and $A_0$ in B. Some textbooks call the second variable $B_t$ as well.
<!-- bilingual-en:end -->

这里用 $A_t$ 暂时区分第二种口径。**若要表达同一经济状态，两者满足 $A_t=RB_t$。** 所以不能把两个式子的初始资产都设成 100，就认为它们代表同样的可用财富。正式推导始终用讲义的 $B_t$。

<!-- bilingual-en:start -->
The temporary symbol $A_t$ distinguishes the second timing convention. **To represent the same economic state, set $A_t=RB_t$.** Giving both variables the value 100 does not give the household the same available wealth. All formal derivations here retain the lecture's $B_t$ convention.
<!-- bilingual-en:end -->

> [!example] 理解辅助：核对两个记账例子
> 取 $R=1.1$、$Y_0=50$、$C_0=30$。A 中若 $B_0=100$，本期资源为 160，$B_1=130$。B 中若 $A_0=100$，本期资源为 150，$A_1=132$。数字 130 和 132 的代数差是 2，但资产时点与初始可用财富都不同，不能将其理解为同一状态下凭空多赚 2。
>
> <!-- bilingual-en:start -->
> Let $R=1.1$, $Y_0=50$, and $C_0=30$. With $B_0=100$ in A, available resources are 160 and $B_1=130$. With $A_0=100$ in B, available resources are 150 and $A_1=132$. The numerical difference is 2, but both timing and initial available wealth differ; this is not an extra gain from the same economic state.
> <!-- bilingual-en:end -->
>
> 要逐项对应，应取 $A_0=RB_0=110$，此时 $A_1=1.1(110+50-30)=143=RB_1$。
>
> <!-- bilingual-en:start -->
> For a like-for-like comparison, use $A_0=RB_0=110$. Then $A_1=1.1(110+50-30)=143=RB_1$.
> <!-- bilingual-en:end -->

你在辅导中完成的验算使用 A：$R=1.1$、$B_0=0$、$Y_0=100$、$Y_1=0$、$C_0=20$、$C_1=0$，得到 $B_1=80$、$B_2=88$。这是一条**用于核对记账的路径**；关键是 $B_1=80$ 为留存本金，而 $RB_1=88$ 才是下一期可用资源。

<!-- bilingual-en:start -->
The calculation completed during tutoring uses convention A: with $R=1.1$, $B_0=0$, $Y_0=100$, $Y_1=0$, $C_0=20$, and $C_1=0$, it gives $B_1=80$ and $B_2=88$. This is an **accounting check**: $B_1=80$ is the saved principal, while $RB_1=88$ is available at the next date.
<!-- bilingual-en:end -->

## 六、λ 是什么：先理解报价，再写拉格朗日函数

<!-- bilingual-en:start -->
*Understanding the multiplier through a resource price*
<!-- bilingual-en:end -->

### 1. 为什么要引入 λ

<!-- bilingual-en:start -->
*Why introduce a multiplier?*
<!-- bilingual-en:end -->

你在第二页手写笔记里已经抓住了方向：λ 是「另一层面上每花一元的代价」。要再补清楚的是，**它把资源成本换算成效用单位，才能与多消费一点的边际收益比较。**

<!-- bilingual-en:start -->
The second handwritten page describes the multiplier as the cost of spending one more unit, measured on another level. More precisely, **it expresses the resource cost in utility units, so that it can be compared with the marginal benefit of consumption.**
<!-- bilingual-en:end -->

> [!example] 理解辅助：管家报价
> 假设不贴现、不计息，预算为 $W$。想象管家为每用一单位资源收取 $p$ 单位效用的账单。给定报价 $p$，你选择消费来最大化「消费效用减去资源账单」。
>
> <!-- bilingual-en:start -->
> With no discounting or interest and a budget of $W$, imagine a steward charging $p$ utility units for each unit of resources used. Given the price $p$, choose consumption to maximize utility minus the resource bill.
> <!-- bilingual-en:end -->
>
> $$
> \max_{C_0,C_1}\ u(C_0)+u(C_1)-p(C_0+C_1).
> $$
>
> 对每一期，最优购买量满足 $u'(C_i)=p$：多消费一点的收益恰等于报价。若报价太低，你想买的消费总量会超出 $W$；若太高，就会剩下资源。管家要找到让消费需求恰好等于 $W$ 的报价，这就是最优乘子 $\lambda^*$。
>
> <!-- bilingual-en:start -->
> Each date's consumption satisfies $u'(C_i)=p$: marginal benefit equals price. Too low a price generates demand above $W$; too high a price leaves resources unused. The price that makes total demand equal $W$ is the optimal multiplier $\lambda^*$.
> <!-- bilingual-en:end -->

把 $p$ 改记为 λ，加上对消费求导时不变的常数 $\lambda W$，便得到拉格朗日函数（Lagrangian）：

<!-- bilingual-en:start -->
Rename the price as $\lambda$ and add $\lambda W$, which is constant when differentiating with respect to consumption. This gives the Lagrangian:
<!-- bilingual-en:end -->

$$
\mathcal L
=u(C_0)+u(C_1)+\lambda(W-C_0-C_1).
$$

**λ 是让资源约束与个人边际选择对得上的报价。** 给定任意 λ，单独最大化消费部分还不够；必须同时找到让预算成立的 λ。

<!-- bilingual-en:start -->
**The multiplier is the price that aligns marginal consumption choices with resource feasibility.** Maximizing over consumption at an arbitrary multiplier is not enough; the multiplier must also make the budget hold.
<!-- bilingual-en:end -->

### 2. 为什么要写三条 FOC

<!-- bilingual-en:start -->
*Why are there three first-order conditions?*
<!-- bilingual-en:end -->

在受约束问题里，最优不是原目标对每个消费的偏导都等于零；这里 $u'(C)>0$。正确要求是：**沿任何允许的小调整方向，都没有一阶改进空间。** 拉格朗日法用资源报价把这个要求表达成对消费的驻点条件，同时用对 λ 的偏导恢复约束。

<!-- bilingual-en:start -->
In a constrained problem, optimality does not mean that the original objective has zero partial derivative with respect to every consumption choice; here $u'(C)>0$. It means that **no permitted small adjustment offers a first-order improvement**. The Lagrangian expresses this through stationarity with respect to consumption, while differentiation with respect to the multiplier restores feasibility.
<!-- bilingual-en:end -->

对带有 $\beta,R$ 的两期问题：

<!-- bilingual-en:start -->
For the two-period problem with discounting and interest:
<!-- bilingual-en:end -->

$$
\mathcal L=u(C_0)+\beta u(C_1)
+\lambda\left(W-C_0-\frac{C_1}{R}\right).
$$

| 条件 | 数学式 | 它在解决什么 |
|---|---|---|
| 对 $C_0$ 求偏导 | $u'(C_0)=\lambda$ | 今天消费的边际收益是否等于资源成本？ |
| 对 $C_1$ 求偏导 | $\beta u'(C_1)=\lambda/R$ | 明天消费的折现收益是否等于其现值资源成本？ |
| 对 $\lambda$ 求偏导 | $W-C_0-C_1/R=0$ | 这个报价下选出的消费是否恰好符合预算？ |

<!-- bilingual-en:start -->
The $C_0$ condition equates today's marginal benefit to its resource cost. The $C_1$ condition equates tomorrow's discounted marginal benefit to its present-value resource cost. The multiplier condition checks whether the chosen consumption bundle exactly satisfies the budget. There are three unknowns, $C_0,C_1,\lambda$, and three corresponding equations.
<!-- bilingual-en:end -->

第二条右端是 $\lambda/R$，因为一单位明天消费只占用 $1/R$ 单位今天资源。当 $R>1$ 时，明天消费的现值价格更低。用第二条乘以 $R$，再与第一条比较，便消去 λ 得到 EE。

<!-- bilingual-en:start -->
The cost in the second condition is $\lambda/R$ because one unit of tomorrow's consumption uses $1/R$ units of today's resources. When $R>1$, its present-value price is lower. Multiply that condition by $R$ and compare it with the first condition to eliminate the multiplier and obtain the Euler equation.
<!-- bilingual-en:end -->

以 $u(C)=\sqrt C$、$W=100$ 为例，完整的三条条件是：

<!-- bilingual-en:start -->
For square-root utility and $W=100$, the full set of conditions is:
<!-- bilingual-en:end -->

$$
\frac{1}{2\sqrt{C_0}}=\lambda,
\qquad
\frac{\beta}{2\sqrt{C_1}}=\frac{\lambda}{R},
\qquad
C_0+\frac{C_1}{R}=100.
$$

前两条给出 $C_1=\beta^2R^2C_0$，第三条给出 $C_0=100/(1+\beta^2R)$，与代入法完全一致。乘子法在这里是另一条解题路径；多期时，它能更清楚地保留每期资源的作用。

<!-- bilingual-en:start -->
The first two conditions give $C_1=\beta^2R^2C_0$, and the third gives $C_0=100/(1+\beta^2R)$, exactly as substitution did. The multiplier method is an alternative solution route here; with many periods, it makes each date's resource constraint easier to track.
<!-- bilingual-en:end -->

### 3. 影子价格：多给一点预算，最优效用能提高多少

<!-- bilingual-en:start -->
*Shadow price: how much does extra wealth raise optimal utility?*
<!-- bilingual-en:end -->

把财富为 $W$ 时、已经重新优化后的最高效用记为 $V(W)$。在本例最优值可微时：

<!-- bilingual-en:start -->
Let $V(W)$ be the highest utility attainable after optimizing at wealth $W$. Where the optimal value is differentiable:
<!-- bilingual-en:end -->

$$
\boxed{\lambda^*(W)=V'(W)},
\qquad
V(W+\Delta W)-V(W)\approx\lambda^*(W)\Delta W.
$$

这就是影子价格（shadow price）：放松一单位资源约束的**边际价值**。它是局部导数；预算一次增加 1 所得到的实际效用增量，与 λ 只是近似相等。

<!-- bilingual-en:start -->
The shadow price is the **marginal value** of relaxing the resource constraint. It is a local derivative, so the actual utility gain from a finite one-unit increase in wealth is approximately, rather than exactly, the multiplier.
<!-- bilingual-en:end -->

> [!example] 理解辅助：用数字验证 λ
> 回到 $u(C)=\sqrt C$、$\beta=R=1$ 的例子。最优分配为每期 $W/2$，因此：
>
> <!-- bilingual-en:start -->
> Return to square-root utility with $\beta=R=1$. Each period consumes $W/2$, so:
> <!-- bilingual-en:end -->
>
> $$
> V(W)=2\sqrt{W/2}=\sqrt{2W},
> \qquad \lambda^*(W)=\frac{1}{\sqrt{2W}}.
> $$
>
> $V(100)\approx14.14214$，$V(101)\approx14.21267$，实际增加约 $0.07053$；$\lambda^*(100)\approx0.07071$，很好地近似了这次增量。
>
> <!-- bilingual-en:start -->
> Utility rises from about $14.14214$ at wealth 100 to $14.21267$ at wealth 101, a gain of about $0.07053$. The initial multiplier, about $0.07071$, closely approximates that gain.
> <!-- bilingual-en:end -->

| 财富 / Wealth $W$ | 100 | 400 | 10000 |
|---|---|---|---|
| $\lambda^*(W)$ | 0.07071 | 0.03536 | 0.00707 |

财富越多，额外一单位财富的效用价值越小。这里的 λ 以效用计价，其数值依赖效用函数的尺度。

<!-- bilingual-en:start -->
More wealth reduces the utility value of one extra unit of resources. The multiplier is measured in utility units, so its numerical value depends on the scale of the utility function.
<!-- bilingual-en:end -->

### 4. λ 的两顶帽子，以及「最大化 L」的准确含义

<!-- bilingual-en:start -->
*The multiplier's two roles and what optimizing the Lagrangian means*
<!-- bilingual-en:end -->

| 看问题的层次 / Perspective | λ 的角色 / Role of the multiplier |
|---|---|
| 对 $C_0,C_1$ 求偏导 / Differentiating with respect to consumption | 暂时按住不动的报价 / A price held fixed |
| 联立求整个系统 / Solving the full system | 由最优条件和约束共同确定的未知数 / An endogenous unknown determined jointly with choices |

这没有矛盾：偏导数本来就是「改变一个变量，其余按住不动」。因此，对消费求导时把 λ 当常数，对 λ 求导时把消费当常数。

<!-- bilingual-en:start -->
There is no contradiction: a partial derivative varies one argument while holding the others fixed. When differentiating with respect to consumption, hold the multiplier fixed; when differentiating with respect to the multiplier, hold consumption fixed.
<!-- bilingual-en:end -->

也不能把 $\mathcal L$ 对消费和 λ 一起当普通函数求最大值。**λ 的条件负责使约束成立。** 在本次凹问题中，可以把合适的最优组合看作鞍点（saddle point）；对等式约束的 λ，$\mathcal L$ 是线性的，约束成立时更是对 λ 完全不变，不存在「λ 越大越好」或严格的 λ 极小点。

<!-- bilingual-en:start -->
Do not jointly maximize the Lagrangian over consumption and the multiplier as an ordinary unconstrained objective. **The multiplier condition enforces feasibility.** In this concave setting the optimal combination has a saddle-point interpretation. The Lagrangian is affine in an equality multiplier and is constant in that multiplier at a feasible allocation, so there is neither a benefit from choosing an ever-larger multiplier nor a strict minimum in the multiplier direction.
<!-- bilingual-en:end -->

## 七、三期模型：同一笔储蓄为何产生两项导数

<!-- bilingual-en:start -->
*Three periods: why the same saving decision contributes two derivative terms*
<!-- bilingual-en:end -->

### 1. 先确定已知量、选择变量与约束

<!-- bilingual-en:start -->
*Identify given quantities, choices, and constraints*
<!-- bilingual-en:end -->

三期是 $t=0,1,2$，初始资产 $B_0$ 给定，收入 $Y_0,Y_1,Y_2$ 给定。原始终端限制是 $B_3\ge0$，即最后不能把债务留给模型之外。

<!-- bilingual-en:start -->
The three dates are $t=0,1,2$, with initial assets $B_0$ and incomes $Y_0,Y_1,Y_2$ given. The original terminal restriction is $B_3\ge0$: unpaid debt cannot be left beyond the model's final date.
<!-- bilingual-en:end -->

**在无遗赠收益、最后一期效用权重为正且 $u'>0$ 时，最优选择 $B_3=0$。** 理由是：如果 $B_3>0$，保持前面选择不动，把它的一部分转成 $C_2$，仍能满足 $B_3\ge0$，同时提高效用。于是本次先把最优终端资产代入为零，集中学习内部各期的条件。

<!-- bilingual-en:start -->
**With no bequest benefit, a positive weight on final-period utility, and $u'>0$, the optimum has $B_3=0$.** If terminal assets were positive, part of them could instead finance additional $C_2$, holding earlier choices fixed, without violating $B_3\ge0$. Utility would rise. This session therefore substitutes zero terminal assets before studying the conditions for the interior dates.
<!-- bilingual-en:end -->

> [!note] 当前终点处理
> 这里使用你已经理解的「最后留下可用财富会浪费效用」论证。讲义通过终端不等式及其乘子给出正式推导；那部分留待下一次学习。
>
> <!-- bilingual-en:start -->
> This uses the resource-exhaustion argument already covered in tutoring. The lecture's formal derivation through the terminal inequality and its multiplier is reserved for the next session.
> <!-- bilingual-en:end -->

$$
\max_{C_0,C_1,C_2,B_1,B_2}
u(C_0)+\beta u(C_1)+\beta^2u(C_2)
$$

$$
\begin{aligned}
B_1&=RB_0+Y_0-C_0,\\
B_2&=RB_1+Y_1-C_1,\\
0&=RB_2+Y_2-C_2.
\end{aligned}
$$

选择变量共 5 个：$C_0,C_1,C_2,B_1,B_2$。每条预算约束各配一个乘子 $\lambda_0,\lambda_1,\lambda_2$。

<!-- bilingual-en:start -->
There are five choice variables: $C_0,C_1,C_2,B_1,B_2$. Attach a separate multiplier, $\lambda_0,\lambda_1,\lambda_2$, to each date's budget constraint.
<!-- bilingual-en:end -->

### 2. 写出 L：给每期资源分别报价

<!-- bilingual-en:start -->
*Write the Lagrangian with a price for each date's resources*
<!-- bilingual-en:end -->

$$
\begin{aligned}
\mathcal L={}&u(C_0)+\beta u(C_1)+\beta^2u(C_2)\\
&+\lambda_0(RB_0+Y_0-C_0-B_1)\\
&+\lambda_1(RB_1+Y_1-C_1-B_2)\\
&+\lambda_2(RB_2+Y_2-C_2).
\end{aligned}
$$

每个括号都按「资源 − 用途」写成等于零的形式。这让 λ 的正号具有直接含义：多给这一期一点资源，会提高最优目标值。

<!-- bilingual-en:start -->
Each bracket is resources minus uses, set equal to zero. This sign convention makes a positive multiplier intuitive: additional resources at that date raise the optimal objective value.
<!-- bilingual-en:end -->

### 3. 对消费求导：每期资源值多少效用

<!-- bilingual-en:start -->
*Differentiate with respect to consumption: value resources in utility units*
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial C_0}
&=u'(C_0)-\lambda_0=0,\\
\frac{\partial\mathcal L}{\partial C_1}
&=\beta u'(C_1)-\lambda_1=0,\\
\frac{\partial\mathcal L}{\partial C_2}
&=\beta^2u'(C_2)-\lambda_2=0.
\end{aligned}
$$

$$
\boxed{\lambda_t=\beta^tu'(C_t)}.
$$

$\lambda_t$ 是第 $t$ 期多一单位资源所带来的**第 0 期现值效用**。因为目标函数把第 $t$ 期效用乘以 $\beta^t$，这个报价也必须包含相同的折现权重。

<!-- bilingual-en:start -->
The multiplier is the **date-0 utility value** of one extra unit of resources at date $t$. Since the objective weights date-$t$ utility by $\beta^t$, the resource price contains the same discount weight.
<!-- bilingual-en:end -->

### 4. 对资产求导：先找本期出口，再找下期入口

<!-- bilingual-en:start -->
*Differentiate with respect to assets: locate today's use and tomorrow's resource*
<!-- bilingual-en:end -->

看 $B_1$：它是第 0 期「存出去」的本金，也是第 1 期产生 $RB_1$ 的本金。因此它在 $\mathcal L$ 中出现两次：

<!-- bilingual-en:start -->
The asset $B_1$ is principal set aside at date 0 and principal paying $RB_1$ at date 1. It therefore appears twice in the Lagrangian:
<!-- bilingual-en:end -->

$$
\underbrace{\lambda_0(\cdots-B_1)}_{\text{第 0 期用途}}
\quad+\quad
\underbrace{\lambda_1(RB_1+\cdots)}_{\text{第 1 期资源}}.
$$

| 同一变量的角色 | 所在的项 | 求导结果 |
|---|---|---|
| 本期出口：多存一单位，少用一单位本期资源 | $-\lambda_t B_{t+1}$ | $-\lambda_t$ |
| 下期入口：这单位本金带来 $R$ 单位下期资源 | $R\lambda_{t+1} B_{t+1}$ | $+R\lambda_{t+1}$ |

<!-- bilingual-en:start -->
One extra unit saved costs one unit of current resources, contributing $-\lambda_t$. It supplies $R$ units of next-period resources, contributing $+R\lambda_{t+1}$. Both appearances must be included in the derivative.
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial B_1}
&=-\lambda_0+R\lambda_1=0,\\
\frac{\partial\mathcal L}{\partial B_2}
&=-\lambda_1+R\lambda_2=0.
\end{aligned}
$$

所以一般形式是：

<!-- bilingual-en:start -->
Thus, in general:
<!-- bilingual-en:end -->

$$
\boxed{\lambda_t=R\lambda_{t+1}}.
$$

经济直觉是边际价值平衡，也可用无套利（no-arbitrage）类比来记：放弃一单位本期资源，损失 $\lambda_t$；将它存起来，获得的下期资源价值是 $R\lambda_{t+1}$。内点最优时二者相等。

<!-- bilingual-en:start -->
The intuition is a balance of marginal values, which can be remembered through a no-arbitrage analogy. Giving up one current resource unit costs $\lambda_t$; saving it creates next-period resources worth $R\lambda_{t+1}$. At an interior optimum these values are equal.
<!-- bilingual-en:end -->

> [!warning] 「有改进空间」不能直接推出「最优解不存在」
> 若当前路径上 $\lambda_t<R\lambda_{t+1}$，且增加储蓄可行，小幅增加储蓄可以提高效用。路径调整后消费变了，边际效用和 λ 也会变；当前资源和消费非负性还会限制能存多少。因此，这个不等式说明当前路径不是内点最优，不能据此断言可以无限储蓄。
>
> <!-- bilingual-en:start -->
> If $\lambda_t<R\lambda_{t+1}$ at the current path and extra saving is feasible, a small increase in saving improves utility. But changing the path also changes consumption, marginal utilities, and the multipliers; resources and nonnegative consumption limit saving. The inequality rules out interior optimality of that path, not the existence of an optimum.
> <!-- bilingual-en:end -->

### 5. 消去乘子，得到两条 EE

<!-- bilingual-en:start -->
*Eliminate the multipliers to obtain two Euler equations*
<!-- bilingual-en:end -->

$$
\lambda_0=R\lambda_1
\quad\Longrightarrow\quad
\boxed{u'(C_0)=\beta R u'(C_1)},
$$

$$
\lambda_1=R\lambda_2
\quad\Longrightarrow\quad
\beta u'(C_1)=R\beta^2u'(C_2)
\quad\Longrightarrow\quad
\boxed{u'(C_1)=\beta R u'(C_2)}.
$$

这两类 FOC 各做一件事：消费条件将「钱」与「效用」联系起来；资产条件将「本期的钱」与「下期的钱」联系起来。合起来就得到消费的跨期最优条件。

<!-- bilingual-en:start -->
The consumption conditions connect resources to utility. The asset conditions connect resources at adjacent dates. Combining them produces the intertemporal condition for consumption.
<!-- bilingual-en:end -->

### 6. 三期问题的方程检查

<!-- bilingual-en:start -->
*Check the equation count for three periods*
<!-- bilingual-en:end -->

| 条件来源 / Source | 条数 / Count |
|---|---|
| 对 $C_0,C_1,C_2$ 求导 / Consumption stationarity | 3 |
| 对 $B_1,B_2$ 求导 / Asset stationarity | 2 |
| 对三个乘子求导，恢复预算 / Budget constraints | 3 |

若连同乘子一起求，8 个未知数对应 8 条条件。消去乘子后，剩 **2 条 EE + 3 条预算 = 5 条方程**，对应 5 个选择变量。练习里说的「5 条 FOC」是前两行的驻点条件；真正解题还必须保留三条预算。数量匹配是防漏检查，不能单靠计数证明方程独立或解唯一。

<!-- bilingual-en:start -->
Including multipliers gives eight unknowns and eight conditions. Eliminating them leaves **two Euler equations plus three budgets for five choices**. The “five FOCs” in the practice instruction are the five stationarity conditions; the three budget equations are still required. Matching counts helps detect omissions but does not prove independence or uniqueness.
<!-- bilingual-en:end -->

## 八、推广下标，并分清现值与当期值乘子

<!-- bilingual-en:start -->
*Generalize the indices and distinguish present-value from current-value multipliers*
<!-- bilingual-en:end -->

当日期为 $t=0,1,\ldots,T$ 时，一共有 $T+1$ 个消费日期。相邻两期的内部推导保持不变：

<!-- bilingual-en:start -->
Dates $t=0,1,\ldots,T$ give $T+1$ consumption dates. The interior derivation for each adjacent pair is unchanged:
<!-- bilingual-en:end -->

$$
\begin{aligned}
\frac{\partial\mathcal L}{\partial C_t}=0
&\quad\Longrightarrow\quad \lambda_t=\beta^tu'(C_t),
&&t=0,\ldots,T,\\
\frac{\partial\mathcal L}{\partial B_{t+1}}=0
&\quad\Longrightarrow\quad \lambda_t=R\lambda_{t+1},
&&t=0,\ldots,T-1.
\end{aligned}
$$

$$
\beta^tu'(C_t)=R\beta^{t+1}u'(C_{t+1})
\quad\Longrightarrow\quad
\boxed{u'(C_t)=\beta R u'(C_{t+1}),\qquad t=0,\ldots,T-1}.
$$

**$T+1$ 期只有 $T$ 个相邻对，因此有 $T$ 条 EE。** 最后一期的 $B_{T+1}$ 没有下一期预算可接，不能套用同一条资产递推条件；它需要单独的终端处理。

<!-- bilingual-en:start -->
**There are $T$ adjacent pairs among $T+1$ dates, hence $T$ Euler equations.** Terminal assets $B_{T+1}$ do not enter a next-period budget, so their condition cannot use the same recursion; they require separate terminal treatment.
<!-- bilingual-en:end -->

### 为什么 λ 的递推中看不到 β

<!-- bilingual-en:start -->
*Why beta is absent from the multiplier recursion*
<!-- bilingual-en:end -->

| 乘子类型 / Type | 本模型中的表达式 / Expression | 相邻期关系 / Recursion |
|---|---|---|
| 现值乘子 / Present-value multiplier | $\lambda_t=\beta^t u'(C_t)$ | $\lambda_t=R\lambda_{t+1}$ |
| 当期值乘子 / Current-value multiplier | $\tilde\lambda_t=\lambda_t/\beta^t=u'(C_t)$ | $\tilde\lambda_t=\beta R\tilde\lambda_{t+1}$ |

现值乘子已经把时间贴现包含在报价中，所以递推式里不再显式写 β。当期值乘子去掉了这个折现权重，跨期比较时就要重新乘上 β。看见 β「不见了」，先检查目标函数和乘子的写法。

<!-- bilingual-en:start -->
Present-value multipliers already include discounting, so their recursion has no explicit beta. Current-value multipliers remove that weight, which must therefore reappear when comparing adjacent dates. If beta seems to be missing, check the objective and multiplier convention first.
<!-- bilingual-en:end -->

## 九、易错点清单

<!-- bilingual-en:start -->
*Error checklist*
<!-- bilingual-en:end -->

**1. 对 $B_{t+1}$ 求导漏掉一项。** 先圈出它出现的两行：本期 $-B_{t+1}$、下期 $+RB_{t+1}$。结果是 $-\lambda_t+R\lambda_{t+1}$。

<!-- bilingual-en:start -->

&nbsp;
**1.** **Missing an asset term.** Locate both appearances first: $-B_{t+1}$ today and $+RB_{t+1}$ tomorrow. Their derivative is $-\lambda_t+R\lambda_{t+1}$.<br>
<!-- bilingual-en:end -->

**2. 认为 $\lambda_t=R\lambda_{t+1}$ 漏了 β。** 先确认这里用现值乘子 $\lambda_t=\beta^tu'(C_t)$；β 已含在乘子中。

<!-- bilingual-en:start -->

&nbsp;
**2.** **Thinking beta was omitted.** With present-value multipliers, $\lambda_t=\beta^tu'(C_t)$ already incorporates discounting.<br>
<!-- bilingual-en:end -->

**3. 移项后留下错误的负号。** $-\lambda_t+R\lambda_{t+1}=0$ 应整理为 $\lambda_t=R\lambda_{t+1}$。五秒自查：在本笔记的「资源 − 用途」符号约定及 $\beta>0,u'>0$ 下，$\lambda_t>0$。

<!-- bilingual-en:start -->

&nbsp;
**3.** **Retaining a minus sign after rearranging.** The correct equation is $\lambda_t=R\lambda_{t+1}$. Under this note's resources-minus-uses convention with $\beta>0$ and $u'>0$, multipliers are positive.<br>
<!-- bilingual-en:end -->

**4. 只用 EE 就想得到消费水平。** EE 给相邻消费关系；预算与边界条件决定可负担的水平。「财富没出现在 EE 中」不意味着消费不受财富影响。

<!-- bilingual-en:start -->

&nbsp;
**4.** **Expecting the Euler equation alone to determine levels.** It relates adjacent choices; budgets and boundary conditions determine affordable levels. The absence of wealth from the equation does not imply that consumption is independent of wealth.<br>
<!-- bilingual-en:end -->

**5. 把 β 打在错误的一期上。** 在标准写法中，今天边际效用等于 **β × R × 明天边际效用**。从目标函数的 $\beta^t$、$\beta^{t+1}$ 出发检查，而不是仅靠背位置。

<!-- bilingual-en:start -->

&nbsp;
**5.** **Discounting the wrong date.** In the standard form, today's marginal utility equals **beta times the gross return times tomorrow's marginal utility**. Check it against the objective's adjacent discount weights.<br>
<!-- bilingual-en:end -->

**6. 链式法则漏外层，或错判 R 的幂次。** 对 $\beta\sqrt{R(100-C_0)}$ 求导为 $-\beta R/[2\sqrt{R(100-C_0)}]$，净剩 $\sqrt R$；对 $\beta\log(R(100-C_0))$ 求导，$R$ 完全约掉。通用 EE 仍保留 $\beta R u'(C_1)$。

<!-- bilingual-en:start -->

&nbsp;
**6.** **Missing the outer derivative or misreading the power of R.** Square-root utility leaves $\sqrt R$ after simplification; log utility cancels $R$ completely after budget substitution. The general Euler equation still contains $\beta R u'(C_1)$.<br>
<!-- bilingual-en:end -->

**7. 心里想 $C_1$，笔下却写 $100-C_0$。** 当 $R\ne1$ 时，$C_1=R(100-C_0)$。完成计算后，整理回 $u'$ 对 $u'$ 的形式，才能清楚地区分消费量与剩余本金。

<!-- bilingual-en:start -->

&nbsp;
**7.** **Confusing future consumption with unspent principal.** When $R\ne1$, $C_1=R(100-C_0)$, not $100-C_0$. Return to the marginal-utility form after calculating.<br>
<!-- bilingual-en:end -->

**8. 约束没有写完整的「等于零」形式。** 写 $\lambda(W-C_0-C_1/R)$；不能省掉 $W$ 后还对 λ 求导。若整体反号，乘子的符号解释也相应反转；等式约束的乘子并非脱离约定后永远为正。

<!-- bilingual-en:start -->

&nbsp;
**8.** **Writing an incomplete zero-form constraint.** Include the full expression $\lambda(W-C_0-C_1/R)$. Omitting $W$ changes the multiplier equation. Reversing the entire constraint reverses the multiplier's sign convention; equality multipliers are not universally positive.<br>
<!-- bilingual-en:end -->

**9. 忘记恢复预算约束。** 对消费和资产求导后，仍需对每个乘子求导，或直接把每条原预算列入方程组。否则找到的边际关系没有保证资源可行。

<!-- bilingual-en:start -->

&nbsp;
**9.** **Forgetting feasibility.** Include every multiplier condition or, equivalently, every original budget. Stationarity alone does not guarantee an affordable allocation.<br>
<!-- bilingual-en:end -->

**10. 没确认资产的记账时点。** 先看是 $RB_t+Y_t-C_t$，还是 $R(B_t+Y_t-C_t)$。符号可能同名但定义不同；本讲所有正式推导采用前者。

<!-- bilingual-en:start -->

&nbsp;
**10.** **Ignoring asset timing.** Check whether the budget uses $RB_t+Y_t-C_t$ or $R(B_t+Y_t-C_t)$. Identically named asset variables may have different definitions; this lecture uses the former.<br>
<!-- bilingual-en:end -->

**11. 把 L 当成对全部变量一起最大化。** 消费方向的驻点条件与 λ 方向恢复约束承担不同作用。满足约束时 $\mathcal L$ 对等式乘子不变；「鞍点」不意味着 λ 方向有一个严格谷底。

<!-- bilingual-en:start -->

&nbsp;
**11.** **Jointly maximizing over choices and multipliers.** Choice stationarity and multiplier feasibility have different roles. At a feasible choice, the Lagrangian is constant in an equality multiplier; its saddle-point interpretation does not imply a strict minimum in that direction.<br>
<!-- bilingual-en:end -->

**12. 混淆 $\sigma$ 与 EIS。** CRRA 下，$\sigma$ 是相对风险厌恶／曲率系数，**$1/\sigma$ 才是跨期替代弹性**。$\sigma$ 越大，EIS 越小。

<!-- bilingual-en:start -->

&nbsp;
**12.** **Confusing sigma with the EIS.** Under CRRA, sigma is the relative-risk-aversion or curvature coefficient; **the EIS is its reciprocal**. A larger sigma means a smaller EIS.<br>
<!-- bilingual-en:end -->

## 十、本次复习的完成条件

<!-- bilingual-en:start -->
*A concrete completion check for this session*
<!-- bilingual-en:end -->

**合上材料，在白纸上从头写出三期问题。** 给定 $B_0,Y_0,Y_1,Y_2$，使用本次已经理解的 $B_3=0$，写出目标、三条预算和 $\mathcal L$。随后写出 5 条消费／资产驻点条件，消去 λ，得到两条 EE。

<!-- bilingual-en:start -->
**Close the materials and reconstruct the three-period problem on a blank page.** Given $B_0,Y_0,Y_1,Y_2$ and using the covered result $B_3=0$, write the objective, three budgets, and Lagrangian. Then write the five consumption/asset stationarity conditions and eliminate the multipliers to obtain two Euler equations.
<!-- bilingual-en:end -->

完成标准是：能不看答案写对符号，并用自己的话解释 **λ 的效用报价、$B_1$ 为什么出现两次、β 为什么在现值乘子递推中没有显式出现**。若卡住，只回看对应的小节，再把该步独立重写一次。

<!-- bilingual-en:start -->
Completion means getting the signs right without consulting the answer and explaining **the multiplier's utility price, the two appearances of $B_1$, and why beta is not explicit in the present-value multiplier recursion**. If stuck, review only the relevant subsection and then reproduce that step independently.
<!-- bilingual-en:end -->

## 十一、术语与本次停止位置

<!-- bilingual-en:start -->
*Terminology and the stopping point*
<!-- bilingual-en:end -->

| 缩写 | 英文 | 中文 |
|---|---|---|
| FOC | First-order condition | 一阶条件 |
| SOC | Second-order condition | 二阶条件 |
| BC | Budget constraint | 预算约束 |
| EE | Euler equation | 欧拉方程 |
| MRS | Marginal rate of substitution | 边际替代率 |
| MRT | Marginal rate of transformation | 边际转换率 |
| CRRA | Constant relative risk aversion | 不变相对风险厌恶 |
| EIS | Elasticity of intertemporal substitution | 跨期替代弹性 |
| — | Lagrange multiplier; shadow price | 拉格朗日乘子；影子价格 |
| — | State variable; control variable | 状态变量；控制变量 |
| — | Present-value; current-value multiplier | 现值乘子；当期值乘子 |
| — | Consumption smoothing | 消费平滑 |
| — | Substitution effect; income effect | 替代效应；收入效应 |
| — | Perturbation; no arbitrage; saddle point | 扰动；无套利；鞍点 |

**学习停在有限期内部各期的欧拉方程。** 下一次从终端条件继续；下表只标位置：

<!-- bilingual-en:start -->
**The stopping point is the Euler equation for adjacent dates within the finite horizon.** Continue next time with terminal conditions; the table only locates the remaining material:
<!-- bilingual-en:end -->

| 后续内容 / Later material | 当前位置 / Current status |
|---|---|
| TVC：Transversality condition，横截性条件 | 只理解了本模型最后花完财富的直觉；正式推导尚未学习 / Terminal-resource intuition covered; formal derivation pending |
| IBC：Intertemporal budget constraint，跨期预算约束 | 已用两期现值预算；多期前向迭代尚未学习 / Two-period budget used; multi-period forward iteration pending |
| 有限期闭式解与一般完整方程计数 / Finite-horizon closed form and full general equation count | 尚未学习 / Pending |
| 无限期、NPGC（No-Ponzi-game condition）及其与 TVC 的区分 / Infinite horizon, the no-Ponzi-game condition, and its distinction from the TVC | 尚未学习 / Pending |
| 连续时间极限、存量与流量 / Continuous-time limit, stocks, and flows | 尚未学习 / Pending |
| Hamiltonian、maximum principle、co-state variable 与连续时间解 / Hamiltonian, maximum principle, co-state variables, and continuous-time solutions | 尚未学习 / Pending |

## 来源与核验

本笔记保留本次「两期直觉 → 乘子 → 三期推广」的学习顺序。正式符号与已学条件以讲义为准；教学例子按原始记录逐式重算。学习状态来自你提供的辅导记录与手写标记；本笔记没有据此判定整讲已经完成。

<!-- bilingual-en:start -->
These notes preserve the session's progression from two-period intuition to multipliers and then three periods. The lecture determines formal notation and the covered conditions; tutoring calculations were checked directly. Learning-state information comes from the supplied record and handwritten annotations and does not establish completion of the whole lecture.
<!-- bilingual-en:end -->

- [[Notes 1 - Dynamic Optimization.pdf#page=1|Dmitry Mukhin — Notes 1, p. 1]]：方法分类、有限期目标、逐期预算、终端限制与贴现约定。
- [[Notes 1 - Dynamic Optimization.pdf#page=2|Notes 1, p. 2]]：效用假设、现值乘子、消费／资产 FOC、EE 与扰动思路。
- [[Notes 1 - Dynamic Optimization.pdf#page=3|Notes 1, p. 3 上半页]]：扰动的一阶变化、MRS = MRT 与消费路径方向；本笔记的推导范围止于终端条件正式推导之前。
- [[Slides 1 - Dynamic Optimization.pdf#page=15|Slides 1，幻灯片 7 的 EE 首次展示（PDF 第 15 页）]]：交叉核对讲授中的欧拉方程；slides 含逐步展示，幻灯片编号与 PDF 页码不同。
- [[DPDE Lecture 1 - AI辅导原始记录|AI 辅导原始记录]]：两期练习、管家报价、三期演示、复习重点与学习边界；其中结论经过本笔记的条件核对。
- [[DPDE Lecture 1 手写笔记 1.jpg|手写第 1 页]]：凹性、β、R、链式法则、预算线与 EIS。
- [[DPDE Lecture 1 手写笔记 2.jpg|手写第 2 页]]：从代入法到拉格朗日法，以及 λ 的理解难点。
- [[DPDE Lecture 1 手写笔记 3.jpg|手写第 3 页]]：多期目标、预算、终端限制、EE 与「按期给钱报价」的理解。

<!-- bilingual-en:start -->
The lecture notes support the setup and method map on p. 1, assumptions and interior first-order conditions on p. 2, and the perturbation and economic interpretation at the top of p. 3. The slide reference cross-checks the Euler equation. The tutoring record supplies the exercises, analogy, and learning boundary; the three handwritten pages establish the learner's sequence and highlighted difficulties.
<!-- bilingual-en:end -->
