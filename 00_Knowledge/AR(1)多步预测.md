---
aliases:
  - "AR(1) 的 h 步最佳线性预测回归均值且无条件 MSE 累积到过程方差"
  - AR(1) h-step forecast
  - AR(1) forecast error variance
  - AR(1) mean reversion forecast
student_os: knowledge-atom
atom_id: TS-ARMA-022
atom_set: arma-modeling
atom_type: calculation
status: source-checked
mastery_state: unassessed
requires:
  - "[[AR(p)模型]]"
  - "[[AR因果根条件]]"
  - "[[ARMA多步预测]]"
related:
  - "[[ARMA预测区间]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# AR(1) 的 h 步最佳线性预测回归均值且无条件 MSE 累积到过程方差
<!-- bilingual-en:start -->
*The best linear h-step forecast for AR(1) reverts to the mean while unconditional MSE accumulates to process variance*
<!-- bilingual-en:end -->

> [!summary] 原子计算
> 对稳定 AR(1)
> $$y_t-\mu=\phi(y_{t-1}-\mu)+\varepsilon_t,\qquad |\phi|<1,$$
> 在参数已知、$\varepsilon_t$ 是方差 $\sigma^2$ 的二阶白噪声时，最佳线性预测与它的无条件均方误差为
> $$\hat y_{t+h|t}=\mu+\phi^h(y_t-\mu),$$
> $$E(e_{t+h|t}^2)=\operatorname{Var}(e_{t+h|t})=
> \sigma^2\sum_{j=0}^{h-1}\phi^{2j}
> =\sigma^2\frac{1-\phi^{2h}}{1-\phi^2}.$$
> 若创新进一步是条件方差恒为 $\sigma^2$ 的 MDS（i.i.d. 是一个充分条件），上述点预测才同时是条件均值，上式也才可相应解释为 $\operatorname{Var}(e_{t+h|t}\mid\mathcal F_t)$。
> <!-- bilingual-en:start -->
> For a stable known-parameter AR(1) driven by second-order white noise, the displayed recursion is the best linear forecast and the geometric sum is its unconditional MSE. If the innovations are additionally an MDS with constant conditional variance $\sigma^2$ (i.i.d. is sufficient), the same forecast is the conditional mean and the same sum is its conditional forecast-error variance.
> <!-- bilingual-en:end -->

因为 $\phi^h\to0$，远期最佳线性预测只剩 $\mu$。同时无条件 MSE 随 $h$ 非减并趋于 $\sigma^2/(1-\phi^2)=\operatorname{Var}(y_t)$。这不是“越远的真实值越稳定”，而是当前的线性预测信息逐渐消失，误差反而扩大到过程的无条件波动水平。在上述 MDS 条件下，才可把同一句话也说成“条件均值回归到 $\mu$”。

$\phi<0$ 时点预测会在均值两侧交替回归；方差仍只含 $\phi^{2j}$。若 $|\phi|\ge1$，上述稳定均值与有限无条件方差极限不成立。
<!-- bilingual-en:start -->
Negative $\phi$ produces alternating mean reversion, but the variance still accumulates through squared coefficients. The long-horizon variance limit is the unconditional variance, not evidence that distant outcomes are more stable. The formulas' stationary limits require $|\phi|<1$.
<!-- bilingual-en:end -->

> [!question]- 自检
> 若只知道创新是二阶白噪声，上面的几何和最稳妥地应该叫什么？什么时候才能叫条件方差？
>
> **答案：** 它是最佳线性预测的无条件 MSE（也是无条件预测误差方差）。只有再加上创新对当时信息的条件均值为 0、条件方差恒为 $\sigma^2$ 等条件，才能把同一公式称为条件误差方差。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=124|课程讲义 pp. 124–126]]：核对 AR(1) 点预测、误差展开与方差极限。
- [Hyndman & Athanasopoulos, FPP3 Chapter 9](https://otexts.com/fpp3/arima.html)：核对稳定 ARIMA/ARMA 多步预测解释。
