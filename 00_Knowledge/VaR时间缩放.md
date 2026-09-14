---
aliases:
  - "把一日 VaR 直接按平方根时间缩放并非定义；它在零均值独立同分布正态增量下精确成立，明确忽略均值时只是近似"
  - "把一日 VaR 直接按平方根时间缩放并非定义，只有零均值或明确忽略均值的独立同分布正态增量才给出该等式"
  - "把一日 VaR 按平方根时间缩放并非定义，其常见推导要求独立同分布的近正态增量、均值单独处理与头寸近线性"
  - Square-root-of-time VaR scaling has restrictive assumptions
  - VaR 持有期平方根缩放边界
student_os: knowledge-atom
atom_id: RM-VAR-004
atom_set: var-es-backtesting
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[风险度量口径]]"
  - "[[独立同分布]]"
  - "[[和的方差协方差项]]"
related:
  - "[[波动率聚集]]"
  - "[[动态Delta对冲]]"
  - "[[FRTB 市场风险]]"
  - "[[方差协方差VaR]]"
  - "[[VaR回测损益口径]]"
leads_to:
  - "[[历史模拟法]]"
  - "[[风险蒙特卡洛]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# 把一日 VaR 直接按平方根时间缩放并非定义；它在零均值独立同分布正态增量下精确成立，明确忽略均值时只是近似
<!-- bilingual-en:start -->
*Direct square-root-of-time scaling is not part of the VaR definition; it is exact with zero-mean iid normal increments and only approximate when the mean is deliberately neglected*
<!-- bilingual-en:end -->

> [!summary] 平方根时间是模型结论
> $h$ 日 VaR 并不由一日 VaR 自动决定。常见的 $\sqrt h$ 规则来自零均值、独立同分布、方差稳定的正态损失增量以及固定近线性头寸；任一关键条件失效，都应直接估计或模拟目标持有期损失。

若每日损失增量满足

$$
X_i\overset{iid}{\sim}N(\mu,\sigma^2),
\qquad
L_h=\sum_{i=1}^hX_i,
$$

则

$$
L_h\sim N(h\mu,h\sigma^2)
$$

且

$$
\operatorname{VaR}_\alpha(L_h)
=h\mu+z_\alpha\sigma\sqrt h.
$$

当 $\mu=0$ 时，完整 VaR 精确满足

$$
\operatorname{VaR}_\alpha(L_h)
=\sqrt h\,\operatorname{VaR}_\alpha(X_1).
$$

若 $\mu$ 并非严格为零，只是建模者明确判断它在目标持有期内可忽略，则只能写成近似式

$$
\operatorname{VaR}_\alpha(L_h)
\approx\sqrt h\,\operatorname{VaR}_\alpha(X_1).
$$

若 $\mu\ne0$ 且要保留均值，“单独处理”的正确式子是

$$
\operatorname{VaR}_\alpha(L_h)
=h\mu+
\sqrt h\bigl(\operatorname{VaR}_\alpha(X_1)-\mu\bigr),
$$

而不是 $\sqrt h\,\operatorname{VaR}_\alpha(X_1)$：日度 VaR 中的均值部分应按 $h$ 累积，只有标准差部分按 $\sqrt h$ 缩放。

若损失增量协方差平稳，令 $\gamma_k=\operatorname{Cov}(X_t,X_{t-k})$，则

$$
\operatorname{Var}\!\left(\sum_{t=1}^hX_t\right)
=h\gamma_0
+2\sum_{k=1}^{h-1}(h-k)\gamma_k.
$$

存在序列相关时，协方差项不能删除；存在波动率聚集时，无条件线性相关很弱也不保证多期尾部分布可由一日分布机械缩放。

即使 IID 成立，非正态分位点也不必按平方根缩放。令每日损失相互独立且

$$
P(X=0)=0.99,\qquad P(X=100)=0.01.
$$

一日 99% VaR 为 0。两日损失满足

$$
P(X_1+X_2=0)=0.99^2=0.9801<0.99,
$$

而

$$
P(X_1+X_2\le100)=0.9999,
$$

所以两日 99% VaR 为 100，不等于 $\sqrt2\times0$。这说明 IID 本身不足以给出精确的分位点缩放。

正态近似还可能被厚尾、跳跃、时变波动率、期权非线性、动态对冲和持有期内头寸变化破坏。中心极限定理主要描述标准化和在分布中心附近的渐近行为，不能自动保证有限 $h$ 下深尾分位准确。

> [!question]- 自检
> 已知一日 99% VaR 为 £1m，能否立即报告十日 VaR 为 $\sqrt{10}$m？
>
> **答案：** 不能。必须先说明损失增量、均值、依赖结构、尾部分布和头寸线性等条件；否则应直接构造十日损失分布。
>
> <!-- bilingual-en:start -->
> **Self-check:** Given a one-day 99% VaR of £1m, can ten-day VaR immediately be reported as $\sqrt{10}$m?
>
> **Answer:** No. The loss increments, mean, dependence structure, tail distribution, and position linearity must first justify the scaling; otherwise construct the ten-day loss distribution directly.
> <!-- bilingual-en:end -->

## 不能越界

- 标准差按 $\sqrt h$ 缩放，不等于 VaR 必然按 $\sqrt h$ 缩放。
- IID 本身不足以给出精确的 VaR 平方根缩放。
- 监管指定的持有期或缩放规则不是统计恒等式。
- 不得把未来持有期 $h$ 与历史估计窗口 $W$ 混为一谈。
- 不得忽略非零均值：均值按 $h$ 缩放，标准差才按 $\sqrt h$ 缩放。
- 头寸若在持有期内重平衡或具有明显非线性，必须重新定义多期损失，而不是只缩放一日数字。

## 来源与核验

- [BCBS Working Paper 19, Messages from the Academic Literature on Risk Measurement for the Trading Book](https://www.bis.org/publ/bcbs_wp19.pdf)：核对 GARCH、跳跃等数据生成过程下平方根缩放的失真方向。
- [Diebold et al., Converting 1-Day Volatility to h-Day Volatility](https://archive.nyu.edu/bitstream/2451/27078/2/wpa98080.pdf)：核对平方根时间规则的限制条件及直接拟合目标 horizon 的建议。
- [[和的方差协方差项]] 与 [[波动率聚集]]：分别承载多期方差展开和条件方差动态。
