---
aliases:
  - "精确GPD尾部升高阈值后形状不变而尺度按阈值差线性调整"
  - GPD threshold stability
student_os: knowledge-atom
atom_id: RM-EVT-011
atom_type: theorem
status: source-checked
part_of:
  - "[[极值理论 EVT 与尾部风险.canvas]]"
requires:
  - "[[广义Pareto分布]]"
  - "[[超阈值法]]"
leads_to:
  - "[[均值超额图]]"
  - "[[POT阈值选择]]"
---

# 精确GPD尾部升高阈值后形状不变而尺度按阈值差线性调整
<!-- bilingual-en:start -->
*Raising the threshold of an exact GPD tail preserves shape and adjusts scale linearly with the threshold difference.*
<!-- bilingual-en:end -->

设损失在阈值 $u$ 以上的超额精确服从 $\operatorname{GPD}(\xi,\beta_u)$。对更高阈值 $v\ge u$，只要 $P(L>v)>0$，也就是这里的 $\beta_v=\beta_u+\xi(v-u)>0$，就有
<!-- bilingual-en:start -->
Suppose the excess loss above $u$ follows an exact $\operatorname{GPD}(\xi,\beta_u)$. For a higher threshold $v\ge u$ with $P(L>v)>0$, equivalently $\beta_v=\beta_u+\xi(v-u)>0$ in this model,
<!-- bilingual-en:end -->

$$
L-v\mid L>v\sim\operatorname{GPD}(\xi,\beta_v),
\qquad \beta_v=\beta_u+\xi(v-u).
$$

因此保持不变的是形状 $\xi$ 和修正尺度 $\beta_v-\xi v$，不是原尺度 $\beta_v$。$\xi>0$ 时尺度上升，$\xi<0$ 时下降，$\xi=0$ 时才不变；有限上端点本身不能作为条件事件有正概率的阈值。
<!-- bilingual-en:start -->
The invariants are shape $\xi$ and modified scale $\beta_v-\xi v$, not raw scale $\beta_v$. Scale increases when $\xi>0$, decreases when $\xi<0$, and stays constant when $\xi=0$. A finite upper endpoint itself cannot be used as a threshold with a positive exceedance probability.
<!-- bilingual-en:end -->

当 $\xi\ne0$ 时，直接取生存概率之比即可证明；$y\ge0$ 还须使对应概率位于支持内：
<!-- bilingual-en:start -->
For $\xi\ne0$, the result follows from a ratio of survival probabilities, with $y\ge0$ restricted to the support:
<!-- bilingual-en:end -->

$$
\begin{aligned}
P(L-v>y\mid L>v)
&=\frac{[1+\xi(v-u+y)/\beta_u]^{-1/\xi}}
{[1+\xi(v-u)/\beta_u]^{-1/\xi}}\\
&=[1+\xi y/\beta_v]^{-1/\xi}.
\end{aligned}
$$

$\xi=0$ 时同样相除得到 $e^{-y/\beta_u}$。例如 $u=100,\beta_u=10,\xi=0.2$，升到 $v=110$ 后 $\beta_v=12$；两处的修正尺度都等于 $-10$。修正尺度可为负，因为它不是分布的尺度参数。
<!-- bilingual-en:start -->
For $\xi=0$, the ratio is $e^{-y/\beta_u}$. For example, $u=100$, $\beta_u=10$, and $\xi=0.2$ imply $\beta_{110}=12$. Both modified scales equal $-10$. A modified scale may be negative because it is not itself a distributional scale parameter.
<!-- bilingual-en:end -->

在真实数据中，[[超阈值极限定理]]只给高阈值近似。跨阈值估计的 $\hat\xi_v$ 与 $\hat\beta_v-\hat\xi_vv$ 应结合误差带观察近似稳定性；抽样波动、相互重叠的超额样本与近似偏差都可能影响图形，不能要求估计值逐点相等。
<!-- bilingual-en:start -->
For real data, the [[超阈值极限定理|threshold-excess limit theorem]] supplies a high-threshold approximation. Examine approximate stability of $\hat\xi_v$ and $\hat\beta_v-\hat\xi_vv$ together with uncertainty bands. Sampling noise, overlapping excess samples, and approximation bias affect the plots; estimates need not agree point by point.
<!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [Haugh，Columbia《Extreme Value Theory》，PDF 第 21、25 页](https://martin-haugh.github.io/files/QRM/EVT_MasterSlides.pdf#page=25)：支持升阈值后的同形状与尺度转换。已重开公式并目视第 25 页；本卡将阈值严格限制在有限端点以下，并依公式区分尺度上升与下降。生存概率比及数字例为直接代数核验。
  <!-- bilingual-en:start -->
  Haugh, PDF pp. 21 and 25, supplies the shape and scale transformation. The formula and p. 25 were reopened visually. This card excludes the finite endpoint and distinguishes increasing from decreasing scale. The survival-ratio proof and example are direct algebraic checks.
  <!-- bilingual-en:end -->
- [Belzile，UNIL 2025《Choosing the threshold in extreme value analysis》，“Threshold stability and extrapolation”及“Graphical stability plots”](https://lbelzile.github.io/UNIL-2025-choosing-threshold/UNIL-choosing_threshold.html#threshold-stability-and-extrapolation)：支持 $\beta_v>0$、阈值相关尺度及稳定图的抽样解释。
  <!-- bilingual-en:start -->
  The cited sections support positive updated scale, threshold-dependent scale, and the sampling interpretation of stability plots.
  <!-- bilingual-en:end -->
