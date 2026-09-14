---
aliases:
  - "Holt-Winters 加法与乘法季节分别描述差值和比率"
  - Additive versus multiplicative Holt-Winters
  - Holt-Winters seasonality
  - 加法季节与乘法季节
student_os: knowledge-atom
atom_id: TS-ETS-004
atom_set: exponential-smoothing-ets
atom_type: specification-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[Holt阻尼趋势]]"
related:
  - "[[加法平滑与加法误差]]"
  - "[[时间序列分解与预测]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# Holt-Winters 加法与乘法季节分别描述差值和比率
<!-- bilingual-en:start -->
*Additive and multiplicative Holt-Winters seasonality represent differences and ratios, respectively*
<!-- bilingual-en:end -->

> [!summary] 原子区别
> 加法季节把 $s_t$ 当作与水平同单位的季节差值，适合季节绝对振幅近似稳定；乘法季节把 $s_t$ 当作无量纲比率，适合季节振幅大致与水平成比例。
> <!-- bilingual-en:start -->
> Additive seasonality expresses seasonal effects as level-scale differences. Multiplicative seasonality expresses them as ratios and is appropriate only when proportional seasonal variation is meaningful.
> <!-- bilingual-en:end -->

以无阻尼趋势为例，加法 Holt–Winters 的点预测是
$$
\hat y_{t+h|t}=\ell_t+hb_t+s_{t+h-m(k+1)},
$$
其中 $m$ 是季节周期，$k=\lfloor(h-1)/m\rfloor$，所以预测始终调用样本末端最近一个完整周期中与目标期同季的状态。季节调整通过减去 $s_{t-m}$ 完成，一个完整周期的季节指标通常归一化为和约为零。若旺季总是比基础水平高约 20 个单位，不论总水平是 100 还是 200，这种加法描述较自然。

乘法版本则为
$$
\hat y_{t+h|t}=(\ell_t+hb_t)s_{t+h-m(k+1)},
$$
季节调整需要除以 $s_{t-m}$，一个周期的季节指标通常归一化为平均约 1（和约为 $m$）。若旺季稳定地约为基础水平的 1.2 倍，乘法描述较自然。

“季节尖峰看起来很大”不足以选择乘法形式；要看振幅是否随水平按比例变化。乘法状态更新含除法，通常要求序列与相关 level/seasonal factors 保持严格为正。零观测会破坏稳定的正比率表示，并可能把季节或水平因子推到零，使后续除法失效；负值则通常失去“季节比例”的含义并带来数值/解释问题。即使数据严格为正，接近零的分母仍可能使更新极不稳定。可考虑合适变换后的加法季节，但 log/Box–Cox 预测还涉及回到原尺度时的偏差与区间，不能把变换当作无成本替代。
<!-- bilingual-en:start -->
Additive seasonal indices sum approximately to zero and enter by addition or subtraction. Multiplicative indices average approximately one and enter by multiplication or division. Zeros make ratio updates undefined, and values near zero can be numerically unstable; large seasonal swings alone do not justify a multiplicative form.
<!-- bilingual-en:end -->

> [!question]- 自检
> 某产品淡季销量可能为 0，但旺季振幅随平均销量增长。能否仅凭“成比例”直接使用乘法 Holt–Winters？
>
> **答案：** 不能。零值破坏严格正的比率表示，并可能使后续按季节或水平因子做的除法失效；应重新考虑尺度、变换或能处理零值的加法/其他模型，并用外样本验证。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §8.3](https://otexts.com/fpp3/holt-winters.html)：核对两套递推、季节指标归一化及差值/比率解释。
- [Winters (1960), “Forecasting Sales by Exponentially Weighted Moving Averages”](https://doi.org/10.1287/mnsc.6.3.324)：Holt–Winters 季节方法的原始论文。
- [Hyndman et al. (2002), §3](https://www.monash.edu/business/ebs/research/publications/ebs/a_state_space_framework_for_automatic_forecasting_using_exponential_smoothing_methods.pdf)：核对含零观测时乘法季节状态空间形式的不适用边界。
