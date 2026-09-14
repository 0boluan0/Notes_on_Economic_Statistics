---
aliases:
  - "ETS 创新残差是按误差形式标准化的一步预测误差"
  - ETS innovation residual
  - ETS relative innovation
  - ETS 一步创新
student_os: knowledge-atom
atom_id: TS-ETS-011
atom_set: exponential-smoothing-ets
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[ETS三轴模型]]"
related:
  - "[[加法平滑与加法误差]]"
  - "[[ETS残差诊断边界]]"
  - "[[ARMA残差诊断]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# ETS 创新残差是按误差形式标准化的一步预测误差
<!-- bilingual-en:start -->
*An ETS innovation residual is a one-step forecast error normalised according to the model's error form*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 令 $\mu_t=\hat y_{t|t-1}$ 为模型在看到 $y_t$ 之前给出的一步条件位置。additive-error ETS 的创新残差是 $y_t-\mu_t$；multiplicative-error ETS 的创新残差是 $(y_t-\mu_t)/\mu_t$。它是观测方程中真正驱动状态更新的估计创新，不应一概等同于原尺度预测误差。
> <!-- bilingual-en:start -->
> Let $\mu_t=\hat y_{t|t-1}$ denote the one-step conditional location available before observing $y_t$. An additive-error ETS innovation is $y_t-\mu_t$, whereas a multiplicative-error ETS innovation is $(y_t-\mu_t)/\mu_t$. This is the estimated shock that drives the state update, not always the raw forecast error on the observation scale.
> <!-- bilingual-en:end -->

additive error 的观测方程为
$$
y_t=\mu_t+\varepsilon_t,
$$
所以 innovation residual 与 regular residual
$$
e_t=y_t-\hat y_{t|t-1}
$$
数值相同、单位也与 $y_t$ 相同。multiplicative error 则写成
$$
y_t=\mu_t(1+\varepsilon_t),
$$
于是
$$
\hat\varepsilon_t=\frac{y_t-\mu_t}{\mu_t},
\qquad
e_t=\mu_t\hat\varepsilon_t.
$$
此时 innovation 是无量纲相对误差，regular residual 仍是原尺度差值；二者只由 $\mu_t$ 联系，并不相等。

这个区别让跨水平比较有意义。若两期的预测位置分别是 200 和 50，而观测分别是 220 和 55，原尺度误差是 20 和 5，但 multiplicative innovations 都是 $0.10$。模型表达的是“都比条件位置高 10%”，不是“第一期的冲击是第二期四倍”。反过来，若误差的绝对幅度大致稳定，强行除以水平会制造并不存在的尺度关系。

multiplicative innovation 要求 $\mu_t$ 能作稳定分母；零、负值或接近零的条件位置会使比率无定义、失去自然解释或数值爆炸。即使模型可以计算，innovation 是否近似零均值、无剩余相关并符合所假定尺度，仍要另做 [[ETS残差诊断边界|残差诊断]]。定义一种残差，并不等于它已经表现良好。
<!-- bilingual-en:start -->
Under additive errors, the innovation and the regular one-step residual coincide on the observation scale. Under multiplicative errors, the innovation is dimensionless and the regular residual equals the conditional location times that innovation. Equal percentage surprises can therefore have different raw magnitudes. The relative form requires a stable nonzero conditional location, and defining the innovation does not establish that the fitted innovations satisfy the model assumptions.
<!-- bilingual-en:end -->

> [!question]- 自检
> 某 multiplicative-error ETS 在一期给出 $\mu_t=80$，实际观测为 92。它的 regular residual 与 innovation residual 分别是多少？
>
> **答案：** regular residual 是 $92-80=12$；innovation residual 是 $12/80=0.15$。前者有观测单位，后者表示高出条件位置 15%。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §8.5](https://otexts.com/fpp3/ets.html)：核对 additive-error 与 multiplicative-error ETS 的观测方程、相对创新定义，以及同一创新如何驱动状态更新。
- [Hyndman & Athanasopoulos, FPP3 §8.6](https://otexts.com/fpp3/ets-estimation.html)：核对 multiplicative-error ETS 中 innovation residual 与 regular one-step residual 不相等。
