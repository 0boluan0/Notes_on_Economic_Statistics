---
aliases:
  - "Monte Carlo error taxonomy"
  - "模拟误差分层"
student_os: knowledge-atom
atom_id: PROB-MC-005
atom_set: monte-carlo-methods
atom_type: concept-distinction
status: source-checked
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[Monte Carlo估计]]"
related:
  - "[[Monte Carlo均值标准误]]"
  - "[[路径数不修模型]]"
  - "[[Monte Carlo实现验证]]"
  - "[[风险模型验证边界]]"
---

<!-- bilingual-en:start -->
# Layers of Monte Carlo error
<!-- bilingual-en:end -->

*Monte Carlo 误差分层*

<!-- bilingual-en:start -->
A simulation can give an inaccurate answer for several different reasons: too few random trials, a coarse numerical approximation, a programming mistake, or a model that does not describe the intended real system. **Error layers** help us identify which question to investigate. A smaller reported sampling error answers only one of these questions.
<!-- bilingual-en:end -->

模拟答案不准确，可能有几种不同原因：随机重复次数太少、数值近似太粗、程序写错，或者模型没有描述想研究的真实系统。**误差分层**帮助我们判断应该查哪一个问题。报告中的抽样误差变小，只回答了其中一个问题。

<!-- bilingual-en:start -->
Start with a simple [[Monte Carlo估计|Monte Carlo estimate]]. Let $U$ be [[连续均匀分布|uniform]] on $[0,1]$, so equal-length intervals have equal probabilities. We want the average squared value, denoted by $\mu$. Using [[LOTUS|expectation of a function]],
$$\mu=E[U^2]=\int_0^1 u^2\,du=\frac13.$$
Generate a fixed positive number $N$ of [[独立同分布|iid]] inputs $U_1,\ldots,U_N$ and average their squares. The calculation is simple enough that we can follow the different error sources separately.
<!-- bilingual-en:end -->

从一个简单的 [[Monte Carlo估计]]开始。令 $U$ 在 $[0,1]$ 上服从[[连续均匀分布]]，等长区间有相同概率。我们要计算平方值的平均数，记为 $\mu$。根据 [[LOTUS|随机变量函数的期望]]，$\mu=E[U^2]=1/3$。先固定正整数重复次数 $N$，生成[[独立同分布]]的 $U_1,\ldots,U_N$，再平均它们的平方。这个计算足够简单，可以分别追踪不同误差来源。

<!-- bilingual-en:start -->
## Finite random trials: variation around the right target
<!-- bilingual-en:end -->

*有限随机重复：围绕正确目标波动*

<!-- bilingual-en:start -->
Even correct code sampling from the correct model generally produces an average different from $1/3$. New random inputs give a new answer. This is **Monte Carlo sampling error**. Under iid sampling with finite output variance, the [[Monte Carlo均值标准误|standard error]] describes the repeated-run spread of the average and decreases in proportion to $1/\sqrt N$. Here the outputs lie between $0$ and $1$, so their variance is finite. The realised error in one run need not decrease whenever $N$ increases.
<!-- bilingual-en:end -->

即使代码和抽样模型都正确，有限次平方值的平均数通常也不等于 $1/3$；换一组随机输入，答案就可能改变。这是 **Monte Carlo 抽样误差**。在 iid 抽样、输出方差有限时，[[Monte Carlo均值标准误|标准误]]描述反复运行所得均值的波动尺度，并按 $1/\sqrt N$ 缩小。本例输出在 $0$ 到 $1$ 之间，所以方差有限；但单次已经实现的误差，不必每增加一次重复就变小。

<!-- bilingual-en:start -->
## Numerical approximation: averaging the approximated calculation
<!-- bilingual-en:end -->

*数值近似：反复计算的是近似对象*

<!-- bilingual-en:start -->
Suppose an algorithm replaces each input by the midpoint of its grid interval before squaring it. With two equal intervals, all inputs below $1/2$ become $1/4$, and the others become $3/4$. Each midpoint has probability $1/2$. This intentionally simplified computation has a different expected output:
$$\frac12\left(\frac14\right)^2+\frac12\left(\frac34\right)^2
=\frac5{16}=\frac13-\frac1{48}.$$
Increasing $N$ makes the average concentrate around $5/16$, not $1/3$. Refining to four equal intervals changes the midpoint expectation to $21/64=1/3-1/192$. The grid, rather than the number of random trials, controls this approximation error.
<!-- bilingual-en:end -->

假设算法在平方前，先把每个输入替换成它所在网格区间的中点。使用两个等长区间时，小于 $1/2$ 的输入都变成 $1/4$，其余变成 $3/4$；两个中点的概率各为 $1/2$。这个有意简化的计算，其期望是 $5/16=1/3-1/48$。增加 $N$ 会让均值集中到 $5/16$，而不是 $1/3$。改成四个等长区间后，中点平方的期望变成 $21/64=1/3-1/192$。控制这个近似误差的是网格精细度，而非随机重复次数。

<!-- bilingual-en:start -->
In a larger model, the approximation might be a time step, a truncated infinite sum, an iterative solver stopped at a tolerance, or finite-precision arithmetic. A correctly implemented approximation can still be inaccurate. Smaller time steps, tighter solver tolerances, or higher-accuracy benchmarks probe this layer; their success depends on the numerical method and problem, so refinement is evidence to inspect rather than an automatic guarantee.
<!-- bilingual-en:end -->

更大的模型中，近似可能来自时间步长、无穷和截断、达到容差就停止的迭代求解器，或者有限精度运算。即使近似方法实现正确，结果仍可能不够精确。缩小步长、收紧求解容差、比较更精确的基准，都是检查这一层的办法；效果取决于数值方法和具体问题，不能把“设置更细”自动当成正确性保证。

<!-- bilingual-en:start -->
## Implementation: computing a different rule by mistake
<!-- bilingual-en:end -->

*实现：程序误算成了另一条规则*

> [!example] Squaring before or after averaging · 平方与平均的顺序
> <!-- bilingual-en:start -->
> The intended rule averages $U_i^2$. A bug might instead square the average of the $U_i$. On the deterministic test inputs $0$ and $1$, the correct output is $(0^2+1^2)/2=1/2$, while the buggy output is $((0+1)/2)^2=1/4$. This discrepancy contains no sampling uncertainty: the inputs have already been fixed.
> <!-- bilingual-en:end -->
>
> 本应先计算各个 $U_i^2$ 再平均，程序却可能先平均 $U_i$ 再平方。人工固定输入 $0,1$ 时，正确结果为 $1/2$，错误结果为 $1/4$。这里不存在抽样不确定性，因为输入已经固定。
>
> <!-- bilingual-en:start -->
> For increasingly many iid uniform inputs, the buggy rule approaches $(E[U])^2=1/4$ by the [[弱大数定律|law of large numbers]] and continuity of squaring. More repetitions make this wrong answer stable. The distinction is [[期望与非线性变换不可交换|averaging a nonlinear transformation versus transforming an average]].
> <!-- bilingual-en:end -->
>
> 对越来越多 iid 均匀输入，由[[弱大数定律]]与平方函数的连续性，错误规则会趋向 $(E[U])^2=1/4$。更多重复让错误答案更稳定；根本区别见[[期望与非线性变换不可交换]]。

<!-- bilingual-en:start -->
Implementation mistakes also include sampling the wrong distribution, using the wrong units, or duplicating supposedly independent random streams. These can change the mean, dependence, or both. A small estimated standard error does not detect them by itself. [[Monte Carlo实现验证|Implementation verification]] compares code with the declared calculation using controlled test inputs and known answers.
<!-- bilingual-en:end -->

实现错误还包括抽错分布、用错单位，或复制本应独立的随机流。它们可能改变均值、依赖关系，或者同时改变二者。很小的估计标准误本身不能发现这些错误；[[Monte Carlo实现验证]]要用受控输入和已知答案，比较代码与原先声明的计算是否一致。

<!-- bilingual-en:start -->
## Model adequacy: whether this is the right real-world question
<!-- bilingual-en:end -->

*模型适用性：它是否回答了正确的现实问题？*

<!-- bilingual-en:start -->
If $U$ represents a real demand fraction, assuming a uniform distribution is a modelling choice. Computing $E[U^2]$ accurately under that assumption does not establish that actual demand is uniform, or that squared demand is the relevant cost. **Model validation** needs evidence tied to the intended use, such as relevant observed data, comparisons with the represented system, and checks of assumptions. Parameters estimated from finite data also carry uncertainty that is not included when those estimates are held fixed during Monte Carlo sampling.
<!-- bilingual-en:end -->

如果 $U$ 代表实际需求比例，假设它均匀分布是一项建模选择。在这个假设下准确算出 $E[U^2]$，并不能说明真实需求确实均匀，也不能说明需求平方就是所关心的成本。**模型有效性验证**需要针对预定用途的证据，例如相关实测数据、与所代表系统的比较，以及对假设的检查。由有限数据估计的参数也带有不确定性；在模拟中把估计值固定后，这部分不确定性不会自动进入 Monte Carlo 抽样误差。

> [!important] A diagnostic map, not four independent error bars · 诊断分层不是四条独立误差棒
> <!-- bilingual-en:start -->
> These categories identify different questions; they are not automatically four independent random variables whose variances can be added. For example, changing the grid also changes the simulated output variance, and a coding error can corrupt the reported standard error. An additive error budget needs an explicit mathematical decomposition and justified dependence assumptions.
> <!-- bilingual-en:end -->
>
> 这些类别对应不同问题，并不自动构成四个可直接相加方差的独立随机变量。例如，改变网格也会改变输出方差，代码错误还可能破坏标准误的计算。要做可加的误差预算，需要明确的数学分解，并说明相应的依赖假设。

<!-- bilingual-en:start -->
An Euler time step is the step size of a particular approximation to continuous-time evolution. If halving that step produces a change larger than the assessed sampling fluctuations, investigate numerical approximation first. That pattern is a diagnostic clue, not proof that the other layers are correct. Likewise, a stable answer far from an analytic benchmark calls for controlled checks: apparent stability can also hide rare events or underestimated sampling uncertainty. [[路径数不修模型|Adding paths alone does not resolve these questions]].
<!-- bilingual-en:end -->

Euler 时间步长是近似连续时间演化的一种方法所用的步长。如果把它减半后，结果变化超过已经评估的抽样波动，先检查数值近似。这是诊断线索，不是其他层都正确的证明。同样，一个稳定但远离解析基准的答案，需要受控检查；表面稳定也可能掩盖尚未遇到的稀有事件或被低估的抽样不确定性。[[路径数不修模型|单纯增加路径数不能回答这些问题]]。

## 来源与核验

<!-- bilingual-en:start -->
- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2, §§2.1–2.3, pp. 15–21](https://artowen.su.domains/mc/Ch-intro.pdf#page=15): iid sampling error and standard error; p. 17 separates model assumptions and parameter errors from sampling error; p. 21 discusses numerical instability in variance computation.
- NASA, [*NASA-STD-7009B* (2024), §§4.2.3–4.2.7, pp. 28–30](https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf#page=28): distinguishes implementation/solution verification from validation against a represented system and records numerical errors and uncertainty. The four-layer organisation is a diagnostic synthesis, not a theorem stated by NASA. The uniform-input calculations here are direct worked examples.
<!-- bilingual-en:end -->

- Owen，同上第 2 章第 15–21 页：支持 iid 抽样误差与标准误；第 17 页区分模型假设、参数错误与抽样误差，第 21 页讨论方差计算的数值不稳定。
- NASA，同上 2024 年标准第 28–30 页：区分实现/解验证与面向所代表系统的有效性验证，并要求记录数值误差与不确定性。本卡四层结构是诊断性整理，不冒充 NASA 的分解定理；均匀输入的算例为直接计算。
