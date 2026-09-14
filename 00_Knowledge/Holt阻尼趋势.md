---
aliases:
  - "Holt 线性趋势无限外推而阻尼 phi 使远期趋势贡献封顶"
  - Holt damped trend
  - Damped trend horizon
  - 阻尼趋势
student_os: knowledge-atom
atom_id: TS-ETS-003
atom_set: exponential-smoothing-ets
atom_type: method-mechanism
status: source-checked
mastery_state: unassessed
requires:
  - "[[SES权重机制]]"
related:
  - "[[Holt-Winters季节形式]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# Holt 线性趋势无限外推而阻尼 phi 使远期趋势贡献封顶
<!-- bilingual-en:start -->
*Holt's linear trend extrapolates indefinitely, while damping caps the long-run trend contribution*
<!-- bilingual-en:end -->

> [!summary] 原子机制
> Holt 线性法用当前水平 $\ell_t$ 与斜率 $b_t$ 给出 $\hat y_{t+h|t}=\ell_t+hb_t$。阻尼法把 $h$ 换成 $\phi+\phi^2+\cdots+\phi^h$；当 $0<\phi<1$ 时，这个趋势乘数随 horizon 累积但最终封顶，贡献方向仍由 $b_t$ 的符号决定。
> <!-- bilingual-en:start -->
> Holt's linear method adds $h$ times the current slope to the level. A damped trend replaces $h$ with a geometric sum, so the trend contributes at short horizons but approaches a finite limit when $0<\phi<1$.
> <!-- bilingual-en:end -->

Holt 的更新同时维护水平与局部斜率：
$$
\ell_t=\alpha y_t+(1-\alpha)(\ell_{t-1}+b_{t-1}),
$$
$$
b_t=\beta^*(\ell_t-\ell_{t-1})+(1-\beta^*)b_{t-1}.
$$
因此 $b_t$ 是随数据更新的局部趋势，不是结构性增长率。直接使用 $hb_t$ 等于假定最后估计斜率在任意远期都继续存在；当 $h$ 很大时，这一假定常比短期更难辩护。

阻尼版本用
$$
\hat y_{t+h|t}=\ell_t+\phi_h b_t,\qquad
\phi_h=\sum_{j=1}^{h}\phi^j=\frac{\phi(1-\phi^h)}{1-\phi}.
$$
阻尼不只修改多步预测式；它也修改一步状态递推中对旧斜率的传递：
$$
\ell_t=\alpha y_t+(1-\alpha)(\ell_{t-1}+\phi b_{t-1}),
$$
$$
b_t=\beta^*(\ell_t-\ell_{t-1})+(1-\beta^*)\phi b_{t-1}.
$$
因此阻尼模型中的 $b_t$ 是按同一个 $\phi$ 更新出来的状态，不能先用无阻尼 Holt 估计 $b_t$，再只在预测阶段机械乘上几何和。

若 $0<\phi<1$，则 $\phi_h\to\phi/(1-\phi)$，所以预测趋向 $\ell_t+\phi b_t/(1-\phi)$，不是继续以固定斜率发散。$\phi=1$ 恢复 Holt 线性趋势；$\phi$ 很接近 1 时，有限 horizon 内阻尼与不阻尼可能几乎不可区分。$\phi$ 也不是“趋势最终必为零”的经济定律，只是对外推形状的统计约束。

阻尼不能自动保证预测合理：负水平、正值约束、结构性拐点或制度信息仍需单独处理。选择 $\phi$ 和是否阻尼，应结合估计稳定性、远期形状与相关 horizon 的 rolling-origin 表现，而不是只看训练期曲线更贴合。
<!-- bilingual-en:start -->
The local slope is an updated state, not a structural growth rate. Damped Holt uses $\phi b_{t-1}$ in both the level prediction and the carried-forward slope; damping cannot be added only after an undamped fit. For $0<\phi<1$, the cumulative multiplier converges to $\phi/(1-\phi)$; $\phi=1$ recovers Holt's undamped method. Damping controls extrapolation shape but does not guarantee positivity or protect against regime changes.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若 $\phi=0.9$，当 $h\to\infty$ 时，当前斜率 $b_t$ 对预测的累计贡献是多少？这是否表示斜率本身被估计为零？
>
> **答案：** 累计贡献趋于 $0.9/(1-0.9)b_t=9b_t$。它限制的是外推总贡献，不是把当前估计斜率改成零。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §8.2](https://otexts.com/fpp3/holt.html)：核对 Holt 与 damped Holt 的状态更新、$\phi_h$ 及远期极限。
- [Gardner & McKenzie (1985), “Forecasting Trends in Time Series”](https://doi.org/10.1287/mnsc.31.10.1237)：原始阻尼趋势论文；核对长期无阻尼外推的风险与长 lead time 的设计动机。
