---
aliases:
- Monte Carlo square-root rule
- Monte Carlo square-root law
- Monte Carlo 平方根律
student_os: knowledge-atom
atom_id: PROB-MC-024
atom_set: monte-carlo-methods
atom_type: decision-rule
status: source-checked
part_of:
- '[[Monte Carlo.canvas|Monte Carlo]]'
requires:
- '[[Monte Carlo均值标准误]]'
leads_to:
- '[[方差缩减]]'
- '[[方差缩减比较口径]]'
related:
- '[[路径数不修模型]]'
---

<!-- bilingual-en:start -->
# The Monte Carlo square-root rule
<!-- bilingual-en:end -->

*Monte Carlo 平方根律*

<!-- bilingual-en:start -->
Suppose a simulation is still too noisy and you can afford more runs. How many more are needed to make the answer noticeably more precise? For the ordinary average of independent draws, halving the standard error takes **four times** as many draws. Making it one tenth as large takes **one hundred times** as many. This is the Monte Carlo square-root rule.
<!-- bilingual-en:end -->

假设模拟答案波动仍然太大，而你可以再多跑一些。究竟多跑多少，才能明显提高精度？对独立抽样后取普通平均的规则，把标准误减半，需要**四倍**抽样数；降到原来的十分之一，需要**一百倍**抽样数。这就是 Monte Carlo 平方根律。

<!-- bilingual-en:start -->
Here “precision” refers to [[标准误含义|standard error]], the standard deviation of the final average across repeated whole simulations. It does not promise that one particular rerun will be closer to the answer. The [[Monte Carlo估计|Monte Carlo method]] fixes a simulation model and output rule, draws numerical outputs, and averages them to estimate the model's expected output.
<!-- bilingual-en:end -->

这里的“精度”指[[标准误含义|标准误]]，即把整个模拟反复重做时，最终平均值的标准差；它不承诺某一次重跑就一定更接近答案。[[Monte Carlo估计|Monte Carlo 方法]]先固定模拟模型与输出规则，生成数值输出，再用平均值估计模型的期望输出。

<!-- bilingual-en:start -->
## The exchange between draws and precision
<!-- bilingual-en:end -->

*抽样数与精度怎样换算*

<!-- bilingual-en:start -->
Let $Y_1,\ldots,Y_N$ be [[独立同分布|iid]] outputs from the same fixed distribution, with mean $\mu$ and **positive finite variance** $\sigma^2$. The estimate is $\widehat\mu_N=N^{-1}\sum_iY_i$. The [[Monte Carlo均值标准误|standard-error formula for an iid mean]] gives $SE_N=\sigma/\sqrt N$. Independence makes the variance of the sum grow as $N\sigma^2$, while division by $N$ squares that divisor in the variance. Taking a square root leaves $N^{-1/2}$.
<!-- bilingual-en:end -->

令 $Y_1,\ldots,Y_N$ 来自同一固定分布且[[独立同分布]]，均值为 $\mu$，**方差正且有限**，记为 $\sigma^2$。估计值是 $\widehat\mu_N=N^{-1}\sum_iY_i$。由[[Monte Carlo均值标准误|iid 均值的标准误公式]]，$SE_N=\sigma/\sqrt N$。独立使求和方差为 $N\sigma^2$；再除以 $N$，会在方差中除以 $N^2$，最后开平方就留下 $N^{-1/2}$。

> [!important] Hold the output distribution fixed · 固定单次输出分布
> $$\frac{SE_M}{SE_N}=\sqrt{\frac NM},\qquad
> SE_M\le\frac{SE_N}{k}\ \Longleftrightarrow\ M\ge k^2N\quad(k>1).$$
> <!-- bilingual-en:start -->
> $N$ is the original number of draws and $M$ the new total, not the number added. Since counts are integers, use $M=\lceil k^2N\rceil$ to meet the theoretical target. The brackets $\lceil\cdot\rceil$ mean take the smallest integer no smaller than the argument. This exact variance calculation needs no normal approximation.
> <!-- bilingual-en:end -->
>
> $N$ 是原抽样数，$M$ 是新的总数，不是另外增加的数量。计数必须为整数，所以取 $M=\lceil k^2N\rceil$；$\lceil\cdot\rceil$ 表示向上取整。这个精确方差换算不需要正态近似。

> [!example] From 100 draws to a smaller error scale · 从 100 次抽样出发
> <!-- bilingual-en:start -->
> If the per-draw standard deviation is $\sigma=3$, then $N=100$ gives $SE=0.3$. The same model gives:
> <!-- bilingual-en:end -->
>
> 若每次输出的标准差为 $\sigma=3$，$N=100$ 时标准误是 $0.3$。保持同一模型：
>
> | Desired SE · 目标标准误 | New total · 新总数 | Additional draws · 需新增 |
> | --- | ---: | ---: |
> | $0.15$ | $400$ | $300$ |
> | $0.10$ | $900$ | $800$ |
> | $0.03$ | $10{,}000$ | $9{,}900$ |
>
> <!-- bilingual-en:start -->
> Twice as many draws would only change $0.3$ to $0.3/\sqrt2\approx0.212$. To reach a specified standard error $h>0$ when $\sigma$ is known, choose a positive integer $N\ge\sigma^2/h^2$.
> <!-- bilingual-en:end -->
>
> 仅把抽样数翻倍，只会把 $0.3$ 降为 $0.3/\sqrt2\approx0.212$。若已知 $\sigma$，要达到指定标准误 $h>0$，选择满足 $N\ge\sigma^2/h^2$ 的正整数即可。

<!-- bilingual-en:start -->
## What must stay the same
<!-- bilingual-en:end -->

*使用换算时，哪些东西必须不变*

<!-- bilingual-en:start -->
In practice $\sigma$ is usually estimated by a sample standard deviation $s$. A pilot estimate suggests a run count, but a new run has a new $s$; its reported estimated standard error need not shrink by exactly the planned factor. A [[Monte Carlo近似区间|normal confidence interval]] has a half-width equal to a normal cutoff times estimated standard error, so the same planning factor applies approximately when the confidence level and distribution remain fixed and the approximation is appropriate.
<!-- bilingual-en:end -->

实际中通常用样本标准差 $s$ 估计未知的 $\sigma$。初步模拟可以帮助规划抽样数，但新一轮会得到新的 $s$，报告出来的估计标准误不必恰好按计划比例缩小。[[Monte Carlo近似区间|正态近似置信区间]]的半宽等于正态临界值乘估计标准误；置信水平和分布不变、且近似合适时，才可作相同的近似规划。

<!-- bilingual-en:start -->
The formula counts independent outputs. Copying an existing output produces no new independent information. Correlated draws, MCMC, quasi-Monte Carlo, stratification, nested simulation or changes to the output rule need their own variance and cost analysis. If $\sigma=0$, the output is already constant almost surely and the standard error is zero; an improvement ratio dividing by it is undefined. With infinite variance, this finite-standard-error argument does not apply.
<!-- bilingual-en:end -->

公式数的是独立输出。复制已有输出不会增加新的独立信息。相关抽样、MCMC、quasi-Monte Carlo、分层抽样、嵌套模拟或改变输出规则，都需要另行分析方差与成本。若 $\sigma=0$，输出已几乎处处为常数，标准误为零，不能再除以它计算改善比；若方差无穷，这个有限标准误论证也不适用。

<!-- bilingual-en:start -->
If the cost of one draw stays approximately constant, $k^2$ times the draws means approximately $k^2$ times the total computational work. It need not mean the same change in elapsed time when setup costs or parallel execution matter. [[方差缩减|Variance reduction]] instead changes how the same target is estimated so the per-draw variability can fall; compare its variance together with its cost. More draws reduce sampling error under the specified model, but [[路径数不修模型|do not repair a wrong model or implementation]].
<!-- bilingual-en:end -->

若单次抽样成本近似不变，抽样数增加到 $k^2$ 倍，总计算工作量也约为 $k^2$ 倍；启动成本或并行执行显著时，经过的实际时间未必同比例变化。[[方差缩减]]则改变同一目标的估计方式，使单次输出波动可能降低，比较时要同时看方差和成本。增加抽样数减少指定模型下的抽样误差，但[[路径数不修模型|不会修好错误的模型或实现]]。

## 来源与核验

<!-- bilingual-en:start -->
- [Art B. Owen, Monte Carlo Theory, Methods and Examples, §2.1, pp.16–17](https://artowen.su.domains/mc/Ch-intro.pdf#page=16): iid-mean variance and RMSE, fixed-variance sample-count scaling, and the role of per-draw cost. Here the ratio and integer-count examples are derived directly from that variance identity.
- [[Monte Carlo均值标准误|Monte Carlo mean standard error]] supplies the exact moment argument; [[Monte Carlo近似区间|normal intervals]] separately supplies the additional distributional approximation needed for interval-width planning. Degenerate and infinite-variance boundaries follow from the stated assumptions.
<!-- bilingual-en:end -->

- Owen 第2.1节第16–17页：核对 iid 均值的方差/RMSE、固定方差下的抽样数换算及单次成本作用；比例与整数例子由方差恒等式直接计算。
- 均值标准误卡承接精确矩计算，近似区间卡另行承接区间宽度需要的分布近似；零方差与无穷方差边界由假设直接检查。
