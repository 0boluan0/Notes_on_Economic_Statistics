---
aliases:
  - "损失口径下，VaR 是给定置信水平的损失左分位点"
  - "损失口径下的 VaR 是给定置信水平的损失分位点而不是最大可能损失"
  - Value at Risk is a loss quantile
  - 损失分位型 VaR 定义
student_os: knowledge-atom
atom_id: RM-VAR-002
atom_set: var-es-backtesting
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[风险度量口径]]"
  - "[[累积分布函数]]"
related:
  - "[[EL与信用VaR]]"
leads_to:
  - "[[ES定义]]"
  - "[[绝对与相对VaR]]"
  - "[[VaR超越概率]]"
  - "[[VaR不是最大损失]]"
  - "[[VaR非次可加]]"
  - "[[VaR采样误差]]"
  - "[[VaR回测损益口径]]"
  - "[[方差协方差VaR]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 损失口径下，VaR 是给定置信水平的损失左分位点
<!-- bilingual-en:start -->
*Under a loss convention, VaR is the left quantile of loss at a stated confidence level*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 对损失随机变量 $L$，置信水平 $\alpha$ 下的 VaR 是损失分布的左分位点：从小到大累积概率时，累计概率第一次达到 $\alpha$ 的损失水平。
>
> <!-- bilingual-en:start -->
> For a loss random variable $L$, VaR at confidence level $\alpha$ is the left quantile of the loss distribution: the first loss level at which cumulative probability reaches $\alpha$.
> <!-- bilingual-en:end -->

令

$$
F_L(\ell)=P(L\le \ell).
$$

损失口径下定义

$$
\operatorname{VaR}_{\alpha}(L)
=q_\alpha(L)
=\inf\{\ell\in\mathbb R:F_L(\ell)\ge\alpha\}.
$$

这个定义同时适用于连续与离散分布。若 $L$ 在 $[0,100]$ 上均匀分布，则 $\operatorname{VaR}_{0.99}(L)=99$：它读取的是损失分布的 99% 概率位置。

<!-- bilingual-en:start -->
The definition applies to both continuous and discrete distributions. If $L$ is uniform on $[0,100]$, then $\operatorname{VaR}_{0.99}(L)=99$: VaR reads the 99th-percentile location of the loss distribution.
<!-- bilingual-en:end -->

VaR 的定义本身不把该位置解释为损失上限，见 [[VaR不是最大损失]]；分位点有概率质量时，严格超越概率也不必恰好等于 $1-\alpha$，见 [[VaR超越概率]]。有限样本排序、插值或参数模型得到的是 VaR 估计量，其不确定性由 [[VaR采样误差]] 承载。

<!-- bilingual-en:start -->
The definition does not make the quantile a maximum loss; see [[VaR不是最大损失|VaR is not a maximum loss]]. With probability mass at the quantile, the strict exceedance probability need not equal $1-\alpha$; see [[VaR超越概率|VaR exceedance probability]]. Sample sorting, interpolation, and parametric models produce estimators of VaR, whose uncertainty is handled in [[VaR采样误差|VaR sampling error]].
<!-- bilingual-en:end -->

> [!question]- 自检
> 若损失分布的累计概率第一次在 £1m 达到 99%，它的 99% VaR 是多少？
>
> **答案：** £1m。VaR 读取的是累计概率第一次达到给定置信水平的位置。
>
> <!-- bilingual-en:start -->
> **Self-check:** If cumulative loss probability first reaches 99% at £1m, what is the 99% VaR?
>
> **Answer:** £1m. VaR reads the first loss level at which cumulative probability reaches the stated confidence level.
> <!-- bilingual-en:end -->

## 不能越界

- 必须先固定损失正号；若随机变量是收益，尾部分位方向不同。
- VaR 是所指定损失分布的函数；改变持有期、组合、损益口径或估计模型会改变所讨论的对象。
- 一个总体分位点定义不证明所估计模型已经校准正确。

<!-- bilingual-en:start -->
- Fix the loss sign first; a return convention reverses the tail direction.
- VaR is a functional of the specified loss distribution. Changing the horizon, portfolio, P&L convention, or estimation model changes the object under discussion.
- A population quantile definition does not establish that an estimated model is correctly calibrated.
<!-- bilingual-en:end -->

## 来源与核验

- [Rockafellar & Uryasev, *Conditional Value-at-Risk for General Loss Distributions*](https://sites.math.washington.edu/~rtr/papers/rtr187-CVaR2.pdf)：核对一般分布中的分位定义。
- [Basel Framework MAR10](https://www.bis.org/basel_framework/chapter/MAR/10.htm?inforce=20230101&published=20200327&tldate=20520504)：核对监管语境中的 VaR 术语。
- [[累积分布函数]]：承载左分位点的基础定义。
