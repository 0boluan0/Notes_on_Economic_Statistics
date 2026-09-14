---
aliases:
  - "SES 的加法创新状态空间形式在对齐符号和初值时等价于 ARIMA(0,1,1)"
  - SES ARIMA(0,1,1) equivalence
  - ETS(A,N,N) ARIMA mapping
  - 简单指数平滑与 ARIMA
student_os: knowledge-atom
atom_id: TS-ETS-010
atom_set: exponential-smoothing-ets
atom_type: algebraic-equivalence
status: source-checked
mastery_state: unassessed
requires:
  - "[[SES权重机制]]"
  - "[[ETS三轴模型]]"
related:
  - "[[ETS与ARIMA边界]]"
  - "[[ARMA似然初值处理]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# SES 的加法创新状态空间形式在对齐符号和初值时等价于 ARIMA(0,1,1)
<!-- bilingual-en:start -->
*The additive-innovation state-space form of SES equals ARIMA(0,1,1) only under aligned signs and initialisation*
<!-- bilingual-en:end -->

> [!summary] 原子等价
> ETS$(A,N,N)$ 的状态方程推出
> $$\Delta y_t=\varepsilon_t+(\alpha-1)\varepsilon_{t-1},$$
> 所以在 ARIMA 约定 $(1-B)y_t=(1+\theta B)\varepsilon_t$ 下，$\theta=\alpha-1$。这是带明确误差符号、参数映射和初始状态的模型等价，不是任意软件拟合结果必然相同。
> <!-- bilingual-en:start -->
> ETS(A,N,N) implies an ARIMA(0,1,1) equation with $\theta=\alpha-1$ under the plus-sign MA convention. Matching finite-sample forecasts also requires aligned innovations and initial-state treatment.
> <!-- bilingual-en:end -->

从
$$
y_t=\ell_{t-1}+\varepsilon_t,\qquad
\ell_t=\ell_{t-1}+\alpha\varepsilon_t
$$
出发，有 $y_{t-1}=\ell_{t-2}+\varepsilon_{t-1}$ 与 $\ell_{t-1}=\ell_{t-2}+\alpha\varepsilon_{t-1}$。相减即得
$$
y_t-y_{t-1}=\varepsilon_t+(\alpha-1)\varepsilon_{t-1}.
$$
因此 SES 的 flat forecast 与这个受限 ARIMA 模型使用同一 innovations recursion。

符号约定必须先写明。有些教材/软件把 MA 写成 $(1-\theta B)\varepsilon_t$，此时同一模型报告的参数会是 $\theta=1-\alpha$，符号相反。传统 $0<\alpha<1$ 映射为 plus convention 下 $-1<\theta<0$，只占 invertible MA(1) 参数空间的一部分；$\alpha=1$ 给出 $\theta=0$ 的随机游走/naïve 边界。$\alpha=0$ 则给出 $\theta=-1$，MA 因子 $(1-B)$ 与左侧差分因子完全抵消；这是非最小、不可逆边界表示，其原 ETS 模型只是固定水平加白噪声，而不是随机趋势。

有限样本还依赖初始化。ETS 要选择或估计 $\ell_0$；ARIMA 程序可能使用 diffuse/Kalman、conditional sum of squares 或不同 pre-sample innovations。只有把初始水平、创新定义、likelihood 条件化方式和参数映射对齐，才应期待逐期 fitted values 与 forecasts 一致。单独拟合两个默认软件对象后出现小差异，并不推翻代数等价。

还要区分“SES 点预测方法”与“ETS$(A,N,N)$ 随机模型”：前者只给递推，后者另指定 additive innovations，因而才有上述 ARIMA 数据生成方程与 likelihood 对应。
<!-- bilingual-en:start -->
The parameter sign changes under the alternative MA convention $(1-\theta B)\varepsilon_t$. Traditional SES occupies only a restricted invertible MA(1) region. At $\alpha=0$, the MA and differencing factors cancel, so the ARIMA expression is non-minimal and the underlying ETS model is a fixed level plus white noise. Default ETS and ARIMA software may differ because they initialise levels and pre-sample innovations differently; equation-level equivalence requires those conventions to be aligned.
<!-- bilingual-en:end -->

> [!question]- 自检
> 某软件把 ARIMA 写成 $(1-B)y_t=(1-\theta B)\varepsilon_t$。SES 的 $\alpha=0.3$ 对应报告的 $\theta$ 是多少？
>
> **答案：** $\theta=1-\alpha=0.7$。若使用 plus-sign convention，则同一模型写成 $\theta=\alpha-1=-0.7$。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §9.10](https://otexts.com/fpp3/arima-ets.html)：核对 ETS$(A,N,N)$ ↔ ARIMA$(0,1,1)$ 与 $\theta=\alpha-1$ 的 plus-sign 映射。
- [Hyndman & Athanasopoulos, FPP3 §8.5](https://otexts.com/fpp3/ets.html)：核对 ETS$(A,N,N)$ 的 measurement/state equations。
- [[ARMA似然初值处理]]：复用不同 likelihood 初始化口径不能机械混同的通用边界。
