---
aliases:
  - "Monte Carlo estimation"
  - "Simple Monte Carlo"
  - "Crude Monte Carlo"
  - "简单蒙特卡洛估计"
student_os: knowledge-atom
atom_id: PROB-MC-004
atom_set: monte-carlo-methods
atom_type: definition
status: source-checked
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[随机变量]]"
  - "[[期望线性性]]"
leads_to:
  - "[[Monte Carlo均值标准误]]"
  - "[[Monte Carlo误差分层]]"
  - "[[Monte Carlo实现验证]]"
  - "[[方差缩减]]"
related:
  - "[[伪随机数生成器]]"
  - "[[风险蒙特卡洛]]"
---

<!-- bilingual-en:start -->
# Monte Carlo estimation
<!-- bilingual-en:end -->

*Monte Carlo 估计*

<!-- bilingual-en:start -->
Monte Carlo estimation answers a question about a probability model by generating many possible outcomes from that model and averaging a number calculated from each outcome. For example, to estimate the chance that two fair dice total at least $10$, repeatedly simulate a pair of dice, record $1$ for a total of at least $10$ and $0$ otherwise, and average those records. The average is the fraction of simulated pairs that meet the condition.
<!-- bilingual-en:end -->

Monte Carlo 估计通过从概率模型生成许多可能结果、再平均每个结果对应的数值，回答关于该模型的问题。例如，要估计两枚公平骰子的点数和至少为 $10$ 的概率，可以反复模拟一对骰子：点数和达到 $10$ 就记 $1$，否则记 $0$。这些记录的平均数，就是满足条件的模拟比例。

> [!example] Six simulated pairs · 六次模拟的记录
> <!-- bilingual-en:start -->
> Assume the dice are independent, and each face is equally likely. One possible six-trial record is:
> <!-- bilingual-en:end -->
>
> 假设两枚骰子独立、各面等可能。六次模拟可能得到下面这份记录：
>
> | Pair · 骰子对 | $(1,6)$ | $(4,6)$ | $(3,5)$ | $(6,6)$ | $(2,4)$ | $(4,5)$ |
> |---|---|---|---|---|---|---|
> | Total · 点数和 | $7$ | $10$ | $8$ | $12$ | $6$ | $9$ |
> | Recorded output · 记录值 | $0$ | $1$ | $0$ | $1$ | $0$ | $0$ |
>
> <!-- bilingual-en:start -->
> The estimated probability is $2/6=1/3$. The exact answer is $6/36=1/6$: the favourable ordered pairs are $(4,6),(5,5),(6,4),(5,6),(6,5),(6,6)$ among $36$ equally likely pairs. The finite simulation need not equal the exact answer. This small example also supplies a known answer against which to [[Monte Carlo实现验证|check an implementation]].
> <!-- bilingual-en:end -->
>
> 这次估计为 $2/6=1/3$。精确答案则是 $6/36=1/6$：$36$ 个等可能的有序骰子对中，有利结果是上面列出的六对。有限次模拟不必恰好等于精确答案；这个小例子也提供了一个已知答案，可以用于[[Monte Carlo实现验证|核对程序实现]]。

<!-- bilingual-en:start -->
## Specify the input, the output, and the target
<!-- bilingual-en:end -->

*先说清楚模拟什么、记录什么、求什么*

<!-- bilingual-en:start -->
Write $X$ for one random input and $P$ for its probability law. The input can be a single [[随机变量|random variable]], a vector such as the two dice, or an entire simulated path. A fixed measurable rule $g$ turns that input into the real-valued output $Y=g(X)$ that we average. Here “measurable” ensures that events concerning the output have probabilities. The target is the model's [[期望|expected output]],
$$\mu=E_P[g(X)]=E_P[Y].$$
The subscript $P$ specifies which model supplies the probabilities. We assume $E_P[|Y|]<\infty$, so this expectation is a finite real number. For the dice, $X$ is the ordered pair and $g(X)$ is the [[事件指示变量|indicator]] that equals $1$ when the total is at least $10$ and $0$ otherwise. The [[指示变量期望|expectation of that indicator]] is exactly the desired probability.
<!-- bilingual-en:end -->

用 $X$ 表示一次随机输入、$P$ 表示它的概率分布。输入可以是单个[[随机变量]]、两个骰子组成的向量，也可以是整条模拟路径。固定的可测规则 $g$ 把输入变成要平均的实数输出 $Y=g(X)$；“可测”保证关于输出的事件具有概率。目标 $\mu$ 是这个模型下的[[期望|平均输出]]，下标 $P$ 说明概率来自哪个模型。要求 $E_P[|Y|]<\infty$，以保证期望是有限实数。骰子例子中，$X$ 是有序骰子对，$g$ 是点数和至少为 $10$ 的[[事件指示变量]]；[[指示变量期望]]恰好就是所求概率。

<!-- bilingual-en:start -->
For **simple Monte Carlo**, choose a positive integer number of trials $N$ in advance, then generate $X_1,\ldots,X_N$ [[独立同分布|independently from the same law $P$]]. Compute $Y_i=g(X_i)$ and use
<!-- bilingual-en:end -->

**简单 Monte Carlo** 先选定正整数重复次数 $N$，再从同一个分布 $P$ 生成[[独立同分布]]的 $X_1,\ldots,X_N$。逐次计算 $Y_i=g(X_i)$，最后使用

> [!important] The simulation average · 模拟均值
> $$\widehat\mu_N=\frac1N\sum_{i=1}^N Y_i
> =\frac1N\sum_{i=1}^N g(X_i).$$
> <!-- bilingual-en:start -->
> The hat marks an estimate, and the subscript $i$ identifies a trial. Before the random inputs are generated, this is a random estimation rule. After they are generated, it is a numerical answer such as $1/3$ in the table. Repeating the whole $N$-trial run can give a different answer.
> <!-- bilingual-en:end -->
>
> 帽子表示估计，下标 $i$ 指第几次实验。生成随机输入之前，它是一条随机估计规则；生成以后，它成为具体数值，例如表中的 $1/3$。重新做一整轮 $N$ 次实验，答案可能改变。

<!-- bilingual-en:start -->
The same construction estimates an average cost, a mean waiting time, or an integral that has been expressed as an expectation. The difficult part may be computing the expectation analytically even when it is easy to generate inputs and evaluate $g$. Both the sampling law and the output rule must match the intended question.
<!-- bilingual-en:end -->

同一构造也能估计平均成本、平均等待时间，或者已写成期望的积分。即使生成输入、计算 $g$ 很容易，直接求出期望仍可能很困难。抽样分布与输出规则都必须对应真正要回答的问题。

<!-- bilingual-en:start -->
## What repeated sampling guarantees
<!-- bilingual-en:end -->

*重复抽样为什么有用*

<!-- bilingual-en:start -->
Because every output has mean $\mu$, [[期望线性性|linearity of expectation]] gives $E_P[\widehat\mu_N]=\mu$. This is [[估计量无偏性|unbiasedness]] under the specified model. Independence is not needed for this expectation identity alone; it is part of the simple Monte Carlo sampling scheme used for the following convergence and error results.
<!-- bilingual-en:end -->

每个输出的均值都是 $\mu$，所以由[[期望线性性]]，$E_P[\widehat\mu_N]=\mu$，这就是针对指定模型的[[估计量无偏性|无偏性]]。单独这条期望恒等式不需要独立；独立是这里的简单 Monte Carlo 抽样方案的一部分，用来支持后面的收敛与误差结论。

<!-- bilingual-en:start -->
For an infinite iid sequence with $E_P|Y|<\infty$, the [[弱大数定律|weak law of large numbers]] says that the chance of a fixed noticeable error tends to zero:
$$P\bigl(|\widehat\mu_N-\mu|>\varepsilon\bigr)\longrightarrow0
\quad\text{for every fixed }\varepsilon>0.$$
This is a statement about increasingly long runs, not a promise that the realised error decreases every time another trial is appended. A finite [[方差|output variance]] is an additional condition for the familiar [[Monte Carlo均值标准误|standard-error formula]]. An integrable output may have infinite variance, so convergence of the average and a finite standard error are different requirements.
<!-- bilingual-en:end -->

对于满足 $E_P|Y|<\infty$ 的无限 iid 序列，[[弱大数定律]]说明：对每个固定 $\varepsilon>0$，模拟均值偏离目标超过 $\varepsilon$ 的概率趋于零。它比较的是越来越长的模拟，不承诺每增加一次实验，已经实现的误差都更小。熟悉的[[Monte Carlo均值标准误|标准误公式]]还额外要求输出的[[方差]]有限；可积输出仍可能具有无限方差，因此均值收敛与有限标准误是两项不同条件。

<!-- bilingual-en:start -->
## Which number of repetitions is increasing?
<!-- bilingual-en:end -->

*重复次数增加了什么信息？*

<!-- bilingual-en:start -->
If one trial generates a dataset of $n$ observations and computes an estimate from it, then $N$ counts how many such datasets are simulated. Changing $n$ changes the estimation experiment; changing $N$ improves the Monte Carlo description of that fixed experiment. This is the [[样本量与模拟重复次数|distinction between sample size and simulation repetitions]]. Averaging the $N$ simulated estimates creates a new Monte Carlo statistic; it does not add observations to the real dataset originally collected.
<!-- bilingual-en:end -->

如果一次实验先生成含 $n$ 个观察的数据集、再计算一个估计值，$N$ 数的是模拟了多少份这样的数据集。改变 $n$ 会改变估计实验本身；改变 $N$ 则改善对固定实验的 Monte Carlo 描述。这就是[[样本量与模拟重复次数]]的区别。把 $N$ 个模拟估计值再平均，会形成新的 Monte Carlo 统计量，却不会给原来实际收集的数据增加观察。

> [!warning] The model is part of the answer · 模型决定所求的答案
> <!-- bilingual-en:start -->
> These guarantees concern $E_P[g(X)]$. More trials cannot establish that $P$ represents the real system, or repair a wrong output rule. If model parameters were fitted from a finite dataset and then held fixed during simulation, this Monte Carlo average targets that fitted model's expectation; uncertainty from fitting is a separate question. See [[Monte Carlo误差分层|the layers of Monte Carlo error]] and [[路径数不修模型|why more paths do not repair a model]].
> <!-- bilingual-en:end -->
>
> 这些保证针对 $E_P[g(X)]$。更多重复不能证明 $P$ 代表真实系统，也不能修正错误的输出规则。如果模型参数由有限数据拟合、随后在模拟中固定，那么这个模拟均值求的是拟合模型的期望；参数拟合带来的不确定性是另一个问题。参见[[Monte Carlo误差分层]]与[[路径数不修模型]]。

<!-- bilingual-en:start -->
This card explains the independent repeated-sampling construction. MCMC, quasi-Monte Carlo, sequential methods, or nested simulation can change the sampling mechanism or estimator; their error analysis must match that construction. A reproducible [[伪随机数生成器|pseudorandom stream]] is an implementation tool, not by itself a proof of all the probabilistic assumptions.
<!-- bilingual-en:end -->

本卡解释独立重复抽样的构造。MCMC、quasi-Monte Carlo、序贯方法或嵌套模拟可能改变抽样机制或估计量，误差分析必须与具体构造匹配。可复现的[[伪随机数生成器|伪随机流]]是实现工具，本身并不证明全部概率假设。

## 来源与核验

<!-- bilingual-en:start -->
- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2, §2.1, pp. 15–17, and §2.4, p. 22](https://artowen.su.domains/mc/Ch-intro.pdf#page=15): the input/output expectation formulation, iid averaging, unbiasedness, laws of large numbers, and event probabilities as indicator expectations. Page 17 distinguishes Monte Carlo error from model error.
- [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec26.pdf#page=21|MIT 6.100L Lecture 26, slides 21–28]]: the course's define–repeat–record–summarise framework, dice frequencies, and averaging a transformed random flow rate. The two-dice table here is a direct teaching calculation, not a claimed quotation from the slides.
<!-- bilingual-en:end -->

- Owen 第 2 章第 15–17 页核对输入、输出、期望目标、iid 均值、无偏性与大数定律，第 22 页核对指标概率估计；第 17 页区分 Monte Carlo 抽样误差与模型误差。
- MIT 第 26 讲第 21–28 页保留课程的重复实验、骰子频率与变换后输出求平均的范围。本文两骰子例子直接枚举核算，未归称为讲义原例。
