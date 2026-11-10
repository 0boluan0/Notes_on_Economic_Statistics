---
aliases:
  - Markov switching model
  - Markov regime-switching model
  - 马尔可夫区制转换
student_os: knowledge-atom
atom_id: PROB-DTMC-023
atom_set: discrete-time-markov-chains
atom_type: definition
status: source-checked
requires:
  - "[[Markov性]]"
related:
  - "[[Markov充分状态]]"
  - "[[状态扩充]]"
  - "[[DTMC时间齐次转移核]]"
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
leads_to:
  - "[[观测序列不必Markov]]"
part_of:
  - "[[离散时间马尔可夫链.canvas]]"
---

<!-- bilingual-en:start -->
# Markov-switching models
<!-- bilingual-en:end -->

*Markov 区制转换模型*

<!-- bilingual-en:start -->
A machine may alternate between a quiet regime and a busy regime, while its recorded signal is noisy. A Markov-switching model describes both **how the hidden regime changes** and **how each regime produces observations**. The regime is not itself the recorded signal. In economics, regimes might represent expansion and recession, or low and high volatility; these labels are interpretations of a specified statistical model.
<!-- bilingual-en:end -->

一台机器可能在“安静”和“忙碌”两种状态之间切换，而仪表信号带有噪声。Markov switching 模型同时描述**潜在区制怎样变化**和**各区制怎样产生观测**。区制本身不等于记录下来的信号。在经济学中，区制可以解释为扩张与衰退，或低波动与高波动；这些名称是对某个具体统计模型的解释。

<!-- bilingual-en:start -->
## A hidden state and a noisy reading

Write $S_t$ for the hidden state at time $t$ and $Y_t$ for the observation. In the simplest version, $S_t$ is a finite-state [[Markov性|Markov chain]]: once its present state is known, earlier states add no information about its next state. We use the [[Markov矩阵左右约定|row convention]]: entry $p_{ij}$ is the probability of moving from state $i$ to state $j$, so each row sums to one.
<!-- bilingual-en:end -->

## 潜在状态与带噪读数

用 $S_t$ 表示时刻 $t$ 的潜在状态，$Y_t$ 表示观测。在最简单的版本中，$S_t$ 是有限状态的 [[Markov性|Markov 链]]：知道当前状态后，更早状态不再提供下一状态的信息。这里采用 [[Markov矩阵左右约定|行约定]]：元素 $p_{ij}$ 是从状态 $i$ 转到状态 $j$ 的概率，因此每行和为一。

> [!example] Two regimes / 两种区制
> <!-- bilingual-en:start -->
> Let the states be $0$ and $1$. Start with probability $1/2$ in each state. The transition matrix and observation probabilities are
> <!-- bilingual-en:end -->
>
> 设状态为 $0$、$1$，初始各有 $1/2$ 的概率。转移矩阵与观测概率为
> $$
> P=\begin{pmatrix}0.9&0.1\\0.1&0.9\end{pmatrix},\qquad
> \Pr(Y_t=1\mid S_t=0)=0.2,\qquad
> \Pr(Y_t=1\mid S_t=1)=0.8.
> $$
> <!-- bilingual-en:start -->
> The regime persists with probability $0.9$. A reading of $1$ is more likely in state $1$, but does not identify that state with certainty. Assume that, given the entire state path, the readings are independent and each reading uses only its current state. This last assumption specifies how the observation noise behaves over time; the two displayed conditional probabilities alone do not specify it.
> <!-- bilingual-en:end -->
>
> 区制以 $0.9$ 的概率保持不变。在状态 $1$ 中更容易读到 $1$，但读到 $1$ 并不意味着状态一定是 $1$。再假设：给定整个状态路径后，各次读数相互独立，而且每次读数只使用当期状态。这个额外假设规定了观测噪声的时间结构，仅有上面两个条件概率还不够。

<!-- bilingual-en:start -->
## What specifies the simple hidden Markov model?

For discrete observations, this version is a hidden Markov model. Its ingredients are an initial distribution $\pi_j=\Pr(S_1=j)$, transition probabilities $p_{ij}$, and an observation probability $b_j(y)=\Pr(Y_t=y\mid S_t=j)$ for each state $j$. The observation law is often called an *emission distribution*. In this version the transition and emission rules are fixed over time.

The observations are [[条件独立|conditionally independent]] given the whole state path: after that path is fixed, the joint observation probability factors into its separate terms. This is not a claim of independence before the hidden states are known. Let $T\ge1$ be the number of observations. Write $s_{1:T}=(s_1,\ldots,s_T)$ and $y_{1:T}=(y_1,\ldots,y_T)$ for specified paths. Their joint probability is
<!-- bilingual-en:end -->

## 简单隐 Markov 模型需要哪些部分？

对离散观测而言，这个版本就是隐 Markov 模型。它需要初始分布 $\pi_j=\Pr(S_1=j)$、转移概率 $p_{ij}$，以及每个状态 $j$ 的观测概率 $b_j(y)=\Pr(Y_t=y\mid S_t=j)$。观测规律也称为“发射分布”。在这个版本中，转移规律与发射规律都不随时间变化。

给定完整状态路径后，观测之间 [[条件独立]]：固定这条路径后，观测的联合概率可以写成各项之积。这不表示尚不知道潜在状态时，各次观测也独立。对正整数 $T$ 次观测，用 $s_{1:T}=(s_1,\ldots,s_T)$、$y_{1:T}=(y_1,\ldots,y_T)$ 表示两条指定路径，其联合概率为

$$
\Pr(S_{1:T}=s_{1:T},Y_{1:T}=y_{1:T})
=\pi_{s_1}
\left(\prod_{t=2}^{T}p_{s_{t-1},s_t}\right)
\left(\prod_{t=1}^{T}b_{s_t}(y_t)\right).
$$

<!-- bilingual-en:start -->
The three factors describe the starting state, subsequent state transitions, and observations along the path. For $T=1$, the transition product has no factors and equals $1$. Because only $Y_{1:T}$ is observed, its probability is obtained by summing this expression over all possible state paths. With the observed data held fixed, that probability as a function of model parameters is the [[似然函数|likelihood]]. Continuous observations use corresponding [[概率密度函数|probability densities]].

A transition matrix alone therefore cannot give the observation likelihood. For example, the same matrix $P$ produces very different readings if the emission probabilities are changed from $0.2,0.8$ to $0.49,0.51$. The initial distribution matters too; it need not be a [[Markov稳态分布|stationary distribution]].
<!-- bilingual-en:end -->

三个因子依次描述初始状态、随后各次状态转移，以及沿路径产生的观测。$T=1$ 时，转移乘积没有因子，约定为 $1$。由于实际只观察到 $Y_{1:T}$，需要对所有可能的状态路径求和，才能得到观测的概率。固定观测数据后，把这个概率看成模型参数的函数，就得到 [[似然函数]]。连续观测相应使用 [[概率密度函数]]。

因此，仅有转移矩阵不能得到观测数据的似然。例如，把发射概率从 $0.2,0.8$ 改为 $0.49,0.51$，即使矩阵 $P$ 不变，读数的性质也会明显变化。初始分布同样重要，而且不必取 [[Markov稳态分布|平稳分布]]。

<!-- bilingual-en:start -->
## Switching regression and autoregression

A switching model can also let the current regime select regression coefficients or a noise variance. One first-order autoregressive specification, conditional on a fixed starting observation $Y_0$, is
<!-- bilingual-en:end -->

## 区制转换回归与自回归

区制转换模型也可以让当前区制选择回归系数或噪声方差。给定一个固定的初始观测 $Y_0$，一种一阶自回归设定是

$$
Y_t=c_{S_t}+\phi_{S_t}Y_{t-1}+\sigma_{S_t}Z_t,
\qquad \sigma_j>0.
$$

<!-- bilingual-en:start -->
Here the real coefficients $c_j$ and $\phi_j$ are the regime's intercept and the multiplier of the previous observation, and $\sigma_j$ is the noise scale. Specify the initial state distribution and its transition law, and let the shocks $Z_t$ be [[独立同分布|independent and identically distributed]] [[正态分布|standard normal]] variables independent of the hidden state path. This gives a complete conditional model. If $Y_0$ is random instead, specify its joint initialization with the state, with the shocks independent of both the initial observation and the state path. The intercept $c_j$ is not generally the unconditional mean of $Y_t$.

Even after fixing the state path, $Y_t$ can now depend on $Y_{t-1}$. The observation part of the likelihood uses densities conditional on that past observation, rather than the simple product of $b_{s_t}(y_t)$ above. Thus “Markov-switching” describes a family of models; the observation dependence must be specified for the particular member.
<!-- bilingual-en:end -->

这里实系数 $c_j$ 是区制截距，实系数 $\phi_j$ 是上一期观测的系数，$\sigma_j$ 是噪声尺度。指定初始状态分布及转移规律，再令冲击 $Z_t$ 为 [[独立同分布]] 的 [[正态分布|标准正态变量]]，且与整个潜在状态路径独立，就得到一个完整的条件模型。若 $Y_0$ 本身随机，还需要指定它与状态的联合初始化方式，并令冲击独立于初始观测及状态路径。截距 $c_j$ 一般不等于 $Y_t$ 的无条件均值。

此时即使固定状态路径，$Y_t$ 仍可能依赖 $Y_{t-1}$。似然中的观测部分应采用给定过去观测的条件密度，而不能直接照搬上面的 $b_{s_t}(y_t)$ 乘积。因此，“Markov switching”是一类模型的名称，具体采用什么观测依赖结构需要另外说清。

> [!note] Changing transition probabilities / 转移概率也可以变化
> <!-- bilingual-en:start -->
> A fixed matrix is not required for every switching model. For a specified external covariate sequence $x_{1:T}$, one may assume
> <!-- bilingual-en:end -->
>
> 并非所有区制转换模型都必须使用固定矩阵。对指定的外部协变量序列 $x_{1:T}$，可以假设
> $$
> \Pr(S_t=j\mid S_{1:t-1},x_{1:T})=p_{S_{t-1},j}(x_t),\qquad t\ge2.
> $$
> <!-- bilingual-en:start -->
> Each row of $p(x_t)$ must still be a probability distribution. Under this declared conditional model, the state is first-order Markov **given the covariate sequence**, possibly with a different matrix at each time. If covariates are random, or include previous observations, ignoring them does not automatically leave a Markov process for $S_t$ alone. Their role and dependence must be part of the model.
> <!-- bilingual-en:end -->
>
> $p(x_t)$ 的每行仍须是概率分布。在这一明确声明的条件模型下，状态**在给定协变量序列后**满足一阶 Markov 性，但各时刻的矩阵可以不同。若协变量随机，或包含过去的观测，忽略这些变量后，$S_t$ 单独未必仍是 Markov 过程；它们的作用和依赖结构也是模型的一部分。

<!-- bilingual-en:start -->
Predicting the next observation usually requires inferring the current hidden state from the available readings. The latest reading alone need not contain all useful information: [[观测序列不必Markov|the observed sequence can retain memory even when the hidden state is Markov]].
<!-- bilingual-en:end -->

预测下一次观测通常需要用已有读数推断当前潜在状态。最新一次读数未必包含所有有用信息：[[观测序列不必Markov|即使潜在状态满足 Markov 性，观测序列也可能保留更早历史的影响]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [Rabiner (1989), A Tutorial on Hidden Markov Models](https://web.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf), printed pp. 259–262 (PDF pp. 3–6): hidden states, emission and initial distributions, and the state-path factorization and summation.
- [Hamilton (2005), Regime-Switching Models](https://econweb.ucsd.edu/~jhamilton/palgrav1.pdf), printed pp. 1–6 (PDF pp. 2–7): switching autoregression, initialization, filtering, and transition rules depending on external variables or past observations. The displayed autoregression applies these ideas in an explicitly specified example.
- [statsmodels, MarkovRegression](https://www.statsmodels.org/stable/generated/statsmodels.tsa.regime_switching.markov_regression.MarkovRegression.html), Parameters and Notes; [Markov autoregression examples](https://www.statsmodels.org/stable/examples/notebooks/generated/markov_autoregression.html), Filardo model section: implemented switching parameters and time-varying transition probabilities. Matrix orientation follows this card's stated row convention.
<!-- bilingual-en:end -->

- Rabiner，印刷第 259–262 页（PDF 第 3–6 页）：核对潜在状态、发射分布、初始分布，以及路径联合概率和对隐路径求和。
- Hamilton，印刷第 1–6 页（PDF 第 2–7 页）：核对区制转换自回归、初始化、滤波，以及依赖外部变量或过去观测的转移规律。本文自回归式把这些结构用于一个明确指定的例子。
- statsmodels 文档的 Parameters、Notes 与示例中的 Filardo 段：核对状态依赖参数及随时间变化的转移概率。矩阵方向以本文声明的行约定为准。
