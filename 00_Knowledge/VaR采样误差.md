---
aliases:
  - "样本 VaR 的不确定性由尾部样本量与分位点附近密度共同决定，尾部稀疏或密度低都会放大标准误"
  - "The sampling error of VaR depends on tail sample size and density at the target quantile"
  - "VaR 分位数标准误"
  - "VaR standard error"
student_os: knowledge-atom
atom_id: RM-VAR-013
atom_set: var-es-backtesting
atom_type: sampling-uncertainty
status: source-checked
mastery_state: unassessed
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
requires:
  - "[[VaR定义]]"
  - "[[风险估计窗口]]"
  - "[[独立同分布]]"
  - "[[依分布收敛]]"
related:
  - "[[风险窗口权衡]]"
  - "[[经典中心极限定理]]"
  - "[[历史模拟法]]"
  - "[[风险蒙特卡洛]]"
  - "[[历史模拟Bootstrap]]"
  - "[[路径数不修模型]]"
  - "[[尾部外推不确定性]]"
  - "[[风险模型验证边界]]"
  - "[[Kupiec无条件覆盖]]"
---

# 样本 VaR 的不确定性由尾部样本量与分位点附近密度共同决定，尾部稀疏或密度低都会放大标准误
<!-- bilingual-en:start -->
*The sampling uncertainty of VaR is governed by tail sample size and the density around the target quantile*
<!-- bilingual-en:end -->

> [!summary] 同一个分位点估计值不等于同一精度
> VaR 是总体分位点，历史样本或模拟样本给出的只是估计量。高置信水平留下的有效尾部观测很少；目标分位点附近若分布很平，一点概率误差就会变成很大的金额误差。

固定 $\alpha\in(0,1)$，令 $L_1,\ldots,L_n$ 为独立同分布样本，
$$
q_\alpha=F_L^{-1}(\alpha),
$$
并假设 $F_L$ 在 $q_\alpha$ 附近有连续且严格为正的密度 $f_L(q_\alpha)$。样本分位数 $\hat q_\alpha$ 满足
$$
\sqrt n(\hat q_\alpha-q_\alpha)
\overset{d}{\longrightarrow}
N\!\left(
0,
\frac{\alpha(1-\alpha)}
{f_L(q_\alpha)^2}
\right).
$$
因此常见渐近标准误为
$$
\widehat{\operatorname{se}}(\hat q_\alpha)
\approx
\frac{\sqrt{\alpha(1-\alpha)}}
{\sqrt n\,\hat f_L(\hat q_\alpha)}.
$$

上述极限把 $\alpha$ 固定后再令 $n\to\infty$。若目标分位水平本身随样本量变化，即 $\alpha=\alpha_n\to1$，尤其当 $n(1-\alpha_n)$ 很小或不发散时，目标已经进入中间或极端次序统计量的范围，不能直接沿用这个固定 $\alpha$ 的正态标准误。

这个公式揭示两层稀缺：

1. 在固定的高 $\alpha$ 下，样本量 $n$ 小时，实际决定尾部分位的次序统计量很少；若 $\alpha_n$ 还随 $n$ 趋近 1，则要另换极端分位理论；
2. $\hat f_L(\hat q_\alpha)$ 小时，分位点附近概率质量稀薄，同样的概率位置误差会跨越更大的损失区间。

## 公式没有覆盖的风险

- **时间依赖：** 若损失序列相关，独立样本公式中的 $\alpha(1-\alpha)$ 要由分位命中过程的长期方差替代；普通随机重抽 bootstrap 也会破坏依赖，应使用与时间结构匹配的区块等方法。
- **离散或混合分布：** 分位点上有概率质量时，连续正密度的渐近公式可能失效；应使用匹配分布结构的次序统计、精确区间或重抽方法。
- **有限模拟误差：** 条件于一个固定模型，独立 Monte Carlo 损失样本也有分位采样误差；增加路径数可以降低它，却不能消除参数误差和模型误设。
- **重新估计误差：** 若 VaR 还依赖重新估计的波动率、相关性或尾部参数，区间估计就要传播这层参数不确定性。采用 bootstrap 或模拟重抽时，每个 replicate 应重跑完整估计管线，不能只重抽最终损失列；若正则条件允许，也可使用 delta method 或 sandwich 协方差等解析渐近方法传播参数误差。

VaR 的统计置信区间回答“这个分位点估计有多不确定”，不是把置信水平从 99% 改成 99.5%。风险置信水平与估计量置信区间是两个不同概率问题。

> [!question]- 自检
> 两个模型都给出 99% VaR 为 £10m，样本量也相同，标准误是否必然相同？
>
> **答案：** 不必然。目标分位点附近的密度、时间依赖、参数估计步骤和模拟方法都可能不同，从而产生不同的不确定性。
>
> <!-- bilingual-en:start -->
> **Self-check:** Two models report the same 99% VaR of £10m from samples of the same size. Must their standard errors be equal?
>
> **Answer:** No. Density near the target quantile, time dependence, parameter-estimation steps, and simulation methods can differ and produce different uncertainty.
> <!-- bilingual-en:end -->

## 来源与核验

- [Bahadur, *A Note on Quantiles in Large Samples*](https://doi.org/10.1214/aoms/1177699450)：核对固定分位水平、正密度条件下样本分位数的渐近表示与方差。
- [Gribkova & Helmers, *On the Bahadur–Kiefer Representation for Intermediate Sample Quantiles*](https://arxiv.org/abs/1106.2260)：核对分位水平可随样本量趋近 0 或 1 的中间样本分位问题，并与固定 $\alpha$ 的中心极限口径区分。
- [Kupiec, *Techniques for Verifying the Accuracy of Risk Measurement Models*](https://fedinprint.org/item/fedgfe/34596/original)：核对高置信水平尾部在有限样本中的低功效与不确定性边界。
- [[02_Economy/07_金融机构与风险管理/13_历史模拟法和极值理论]]：提供 VaR 标准误计算的课程语境。
