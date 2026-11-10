---
aliases:
  - "CTMC shared-rate estimation"
  - "Pooled CTMC rate MLE"
  - "共同生成率估计"
student_os: knowledge-atom
atom_id: PROB-CTMC-031
atom_set: continuous-time-markov-chains
atom_type: method
status: source-checked
requires:
  - "[[CTMC完整路径估计]]"
  - "[[生成矩阵约束]]"
related:
  - "[[Poisson是常速纯生链]]"
part_of:
  - "[[连续时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Estimating a shared CTMC rate
<!-- bilingual-en:end -->

*CTMC 共同率估计*

<!-- bilingual-en:start -->
Sometimes several transitions are governed by the same unknown clock speed. Estimating each edge separately would ignore that model restriction. **Shared-rate estimation** first expresses all affected rates using the common parameter, then combines the relevant jumps and exposure times into one likelihood. The exposure is weighted by how strongly each edge uses that parameter.
<!-- bilingual-en:end -->

有时，多种转移由同一个未知时钟速度控制。分别估计每条边，会忽略模型的约束。**共同率估计**先把相关速率写成共同参数的函数，再把对应跳数和暴露时间合并进一个似然。每条边的暴露需要按它使用该参数的程度加权。

<!-- bilingual-en:start -->
## Two routes share one clock
<!-- bilingual-en:end -->

*两条路线共享一个时钟*

<!-- bilingual-en:start -->
Suppose a system can move from $A$ to $B$, from $B$ back to $A$, and from $A$ to an absorbing state $C$. The model fixes $q_{AB}=\theta$ and $q_{AC}=2\theta$, with unknown $\theta\ge0$; the return rate $q_{BA}=\eta$ is a separate free parameter. During a complete four-hour record, start in $A$, jump to $B$ at time $1$, return to $A$ at time $3$, and then remain in $A$ until observation stops.
<!-- bilingual-en:end -->

设系统可以从 $A$ 到 $B$、从 $B$ 返回 $A$，还可从 $A$ 到吸收状态 $C$。模型规定 $q_{AB}=\theta$、$q_{AC}=2\theta$，未知参数 $\theta\ge0$；返回速率 $q_{BA}=\eta$ 是另一个独立参数。完整记录四小时：从 $A$ 出发，时刻 $1$ 跳到 $B$，时刻 $3$ 回到 $A$，之后一直留在 $A$，直到观察结束。

<!-- bilingual-en:start -->
The exposure at $A$ is $S_A=1+(4-3)=2$ hours, including the unfinished final wait. We see $N_{AB}=1$ and $N_{AC}=0$. While the system occupies $A$, both outgoing routes are available, so the total exit rate contributed by $\theta$ is $3\theta$. The part of the [[似然函数|likelihood]] involving $\theta$ is shown below; $\propto$ omits a multiplicative factor that does not depend on $\theta$.
<!-- bilingual-en:end -->

在 $A$ 的暴露为 $S_A=1+(4-3)=2$ 小时，包含最后未完成的等待。观察到 $N_{AB}=1$、$N_{AC}=0$。只要处在 $A$，两个出口都可能发生，所以 $\theta$ 对总出口率的贡献为 $3\theta$。[[似然函数]]中涉及 $\theta$ 的部分如下；$\propto$ 表示省去了不依赖 $\theta$ 的乘法因子。

$$L(\theta)\propto\theta^1(2\theta)^0e^{-(\theta+2\theta)2}=\theta e^{-6\theta},\qquad \widehat\theta=\frac16\ \text{per hour}.$$

<!-- bilingual-en:start -->
Thus the constrained fitted rates are $\widehat q_{AB}=1/6$ and $\widehat q_{AC}=1/3$ per hour. A positive fitted rate on the unobserved $A\to C$ edge is intentional: the observed $A\to B$ jump supports a positive shared parameter. Fitting edges freely would instead give $1/2$ and $0$, which do not satisfy the specified ratio.
<!-- bilingual-en:end -->

因此约束后的拟合速率为 $\widehat q_{AB}=1/6$、$\widehat q_{AC}=1/3$ 每小时。没有观察到的 $A\to C$ 边仍得到正速率，是约束的正常结果：$A\to B$ 的跳跃支持正的共同参数。自由逐边拟合则会给出 $1/2$ 和 $0$，不满足指定比例。

<!-- bilingual-en:start -->
## Pool weighted exposure before maximizing
<!-- bilingual-en:end -->

*先汇总加权暴露，再最大化*

<!-- bilingual-en:start -->
Use the [[CTMC完整路径估计|complete-path observation setup]]: a finite time-homogeneous CTMC, a fixed window $[0,T]$ with every genuine jump observed, exposure including the terminal wait, and likelihood conditional on the initial state. Let $E$ be a nonempty set of directed edges $(i,j)$ with $i\ne j$. For these edges assume
<!-- bilingual-en:end -->

采用[[CTMC完整路径估计|完整路径观察设定]]：有限时间齐次 CTMC、固定窗口 $[0,T]$、观察到每次真实跳跃，暴露包含终点前等待，并使用条件于初态的似然。令 $E$ 为非空的有向边集合，各边 $(i,j)$ 满足 $i\ne j$。对这些边假设

$$q_{ij}=c_{ij}\theta,\qquad c_{ij}>0\ \text{known},\qquad \theta\ge0.$$

<!-- bilingual-en:start -->
The multipliers $c_{ij}$ are fixed, known and dimensionless here, so $\theta$ has units of inverse time. Other off-diagonal rates do not depend on $\theta$; the [[生成矩阵约束|diagonals]] still adjust to make each row sum to zero. Let $N_{ij}$ be the observed jump count and $S_i$ the total time in state $i$. Define
<!-- bilingual-en:end -->

这里 $c_{ij}$ 为固定、已知且无量纲的比例系数，所以 $\theta$ 的单位是时间的倒数。其他非对角速率不依赖 $\theta$；[[生成矩阵约束|对角元]]仍须调整为使每行和为零。令 $N_{ij}$ 为观察跳数，$S_i$ 为在 $i$ 的总停留时间，定义

$$N_E=\sum_{(i,j)\in E}N_{ij},\qquad D_E=\sum_{(i,j)\in E}c_{ij}S_i.$$

<!-- bilingual-en:start -->
$N_E$ counts events on the constrained edges. $D_E$ is their weighted exposure: if two edges leave the same state, both use its exposure, each with its own multiplier. It need not equal the observation length $T$. In the example $E=\{(A,B),(A,C)\}$, giving $N_E=1$ and $D_E=(1+2)\cdot2=6$ hours.
<!-- bilingual-en:end -->

其中 $N_E$ 计数约束边上的事件，$D_E$ 是对应的加权暴露。同一状态若有两条约束出口，两条边都使用该状态的暴露，分别乘自己的系数；$D_E$ 不必等于观察长度 $T$。例子中 $E=\{(A,B),(A,C)\}$，所以 $N_E=1$，$D_E=(1+2)\cdot2=6$ 小时。

<!-- bilingual-en:start -->
Substituting the constraints into the path likelihood gives
<!-- bilingual-en:end -->

把约束代入路径似然，得到

$$
\begin{aligned}
L(\theta)&\propto\prod_{(i,j)\in E}(c_{ij}\theta)^{N_{ij}}e^{-c_{ij}\theta S_i}\\
&\propto\theta^{N_E}e^{-\theta D_E}.
\end{aligned}
$$

<!-- bilingual-en:start -->
The omitted factors do not depend on $\theta$. For $D_E>0$ and $N_E>0$, the log-likelihood's [[导数|derivative]] is $N_E/\theta-D_E$, and its second derivative is $-N_E/\theta^2<0$. Therefore
<!-- bilingual-en:end -->

略去的因子不依赖 $\theta$。若 $D_E>0$ 且 $N_E>0$，对数似然的[[导数|一阶导数]]为 $N_E/\theta-D_E$，二阶导数为 $-N_E/\theta^2<0$。所以

$$\widehat\theta=\frac{N_E}{D_E},\qquad \widehat q_{ij}=c_{ij}\widehat\theta\quad((i,j)\in E).$$

<!-- bilingual-en:start -->
For edges with $c_{ij}S_i>0$, their separate implied estimates of $\theta$ would be $N_{ij}/(c_{ij}S_i)$. The pooled estimate is a weighted average of these values with weights proportional to $c_{ij}S_i$, not their simple arithmetic mean. Edges with positive exposure and zero events still enter the denominator.
<!-- bilingual-en:end -->

对 $c_{ij}S_i>0$ 的边，单独推算的 $\theta$ 会是 $N_{ij}/(c_{ij}S_i)$。汇总估计相当于按 $c_{ij}S_i$ 成比例加权这些值，并不是简单算术平均。有正暴露却零事件的边仍然进入分母。

> [!warning] Zero data and structural zeros / 零记录与结构零
> <!-- bilingual-en:start -->
> If $D_E>0$ and $N_E=0$, the MLE is zero when the parameter domain includes zero; if it requires $\theta>0$, only a boundary supremum exists. If $D_E=0$, a valid record almost surely has $N_E=0$ too, and the likelihood is constant in $\theta$: the record does not identify it. An edge with $c_{ij}=0$ is a structural zero and is excluded from $E$. An observed jump on such an edge contradicts the parameterization rather than providing an alternative estimate.
> <!-- bilingual-en:end -->
>
> 若 $D_E>0$、$N_E=0$，参数域包含零时 MLE 为零；若强制 $\theta>0$，则只有边界上确界。若 $D_E=0$，合法记录几乎必然也有 $N_E=0$，似然不随 $\theta$ 改变，记录无法识别它。$c_{ij}=0$ 的边是结构零，不放入 $E$；若观察到该边跳跃，说明参数化与数据矛盾，并不是需要另换一个估计公式。

<!-- bilingual-en:start -->
## Poisson and population examples
<!-- bilingual-en:end -->

*Poisson 与种群例子*

<!-- bilingual-en:start -->
For a [[Poisson是常速纯生链|constant-rate pure-birth process]] started at zero, every $n\to n+1$ edge has rate $\lambda$. During a complete window with $T>0$, the jump count is $N(T)$ and the exposure times of all visited states sum to $T$. Thus $\widehat\lambda=N(T)/T$, including zero when no event is observed. Although the state set is infinite, this non-explosive model has only finitely many jumps on the window, so the same path calculation is finite.
<!-- bilingual-en:end -->

对从零开始的[[Poisson是常速纯生链|常速纯生过程]]，每条 $n\to n+1$ 边的速率均为 $\lambda$。完整观察窗中 $T>0$，跳数为 $N(T)$，所有访问状态的暴露时间之和为 $T$，因此 $\widehat\lambda=N(T)/T$；没有事件时包括零估计。虽然状态集无限，这个非爆炸模型在窗内只有有限次跳跃，所以路径计算仍是有限的。

<!-- bilingual-en:start -->
If a population of size $i$ has death rate $i\mu$, with known multiplier $i$ and common per-individual rate $\mu$, its contribution is different. The denominator is $\sum_{i\ge1}iS_i$, measured in individual-time units, and the numerator is the total observed deaths. For several independent paths with the same rate model, add counts and weighted exposures before taking the ratio.
<!-- bilingual-en:end -->

若种群规模为 $i$ 时死亡速率为 $i\mu$，已知系数为 $i$，共同参数为每个个体的死亡率 $\mu$，则分母不同：它是 $\sum_{i\ge1}iS_i$，单位为个体时间，分子为总观察死亡数。多个独立路径若共享同一速率模型，也应先相加计数与加权暴露，再求比率。

<!-- bilingual-en:start -->
The constraint is a modeling assumption, not something established merely by pooling observations. Unknown multipliers, other parameter-dependent rates, a parameter-dependent initial law included in the likelihood, or additional parameter restrictions can change the maximization problem. Discrete snapshots still require their own observation likelihood. The displayed kernel resembles a Poisson likelihood, but that algebra alone does not make $N_E$ conditionally Poisson given the random exposure, nor guarantee an unbiased estimator.
<!-- bilingual-en:end -->

共同约束是一项模型假设，并不是把观察记录合并就证明了它。如果比例系数未知、其他速率也依赖该参数、把依赖参数的初始律纳入似然，或另有参数限制，最大化问题都可能改变。离散快照仍须使用自己的观察似然。展示的核看起来像 Poisson 似然，但仅凭这个代数形式，不能断言给定随机暴露后 $N_E$ 就服从 Poisson，也不能保证估计无偏。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Oxford Statistical Lifetime-Models, §10.3, PDF pp.75–78 / printed pp.67–70, especially equations (10.14)–(10.15)](https://www.stats.ox.ac.uk/~winkel/bs3b10_14.pdf#page=75): known relationships between rates, inserting constraints into the likelihood, and the individual-time weighted denominator. §10.1.2, PDF pp.71–72, covers aggregation of paths.
- [Tataru and Hobolth (2011), Background, equation (1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3329461/): the complete-path likelihood from which the known $c_{ij}\theta$ substitution is derived. The finite example, boundary cases and weighted-average identity are calculated above.
<!-- bilingual-en:end -->

- Oxford §10.3，PDF75–78／印刷67–70页，尤其式(10.14)–(10.15)：核对已知速率关系、先代入约束，以及按个体时间加权的分母。§10.1.2 PDF71–72核对路径汇总。
- Tataru 与 Hobolth（2011）Background 式(1)：提供完整路径似然；正文直接代入已知 $c_{ij}\theta$，并计算有限例子、边界及加权平均恒等式。
