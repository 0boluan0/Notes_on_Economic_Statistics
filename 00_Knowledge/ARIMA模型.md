---
aliases:
  - "ARIMA(p,d,q) 是对 d 阶差分序列拟合 ARMA(p,q) 的模型"
  - "ARIMA 在适当差分后的序列上拟合 ARMA 而过度差分会制造动态"
  - ARIMA差分边界
  - ARIMA
  - ARIMA model
  - ARIMA as differenced ARMA
  - 差分后的 ARMA
student_os: knowledge-atom
atom_id: TS-ARMA-020
atom_set: arma-modeling
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARMA(p,q)模型]]"
  - "[[滞后算子约定]]"
  - "[[I(d)定义]]"
related:
  - "[[最小差分原则]]"
  - "[[ACF-PACF阶数识别]]"
  - "[[ARMA信息准则]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
  - "[[趋势、单位根与差分.canvas]]"
---

# ARIMA(p,d,q) 是对 d 阶差分序列拟合 ARMA(p,q) 的模型
<!-- bilingual-en:start -->
*An ARIMA(p,d,q) model fits ARMA(p,q) dynamics to a series differenced d times*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 非季节 ARIMA($p,d,q$) 写成
> $$\phi(B)(1-B)^dy_t=c+\theta(B)\varepsilon_t.$$
> 先令 $w_t=(1-B)^dy_t$；若 $w_t$ 是适合平稳建模的对象，再用 ARMA($p,q$) 描述它的短期线性依赖。因此 $d$ 决定**建模哪个序列**，$p$ 与 $q$ 决定该序列怎样依赖过去观测和创新。
> <!-- bilingual-en:start -->
> A non-seasonal ARIMA($p,d,q$) first forms $w_t=(1-B)^dy_t$ and then models the stationary short-run dependence in $w_t$ with an ARMA($p,q$). Thus $d$ chooses the response being modelled, while $p$ and $q$ describe its AR and MA dynamics.
> <!-- bilingual-en:end -->

“I” 是 integrated：差分是积分的逆运算。若 $d=0$，ARIMA($p,0,q$) 就是 ARMA($p,q$)；若 $(p,d,q)=(0,1,0)$ 且 $c=0$，则
$$\Delta y_t=\varepsilon_t$$
对应无漂移随机游走。恢复水平预测时，要把预测的差分逐期累加回去；不能把 $w_t$ 的预测直接当作 $y_t$ 的预测。
<!-- bilingual-en:start -->
The “I” stands for integrated because differencing reverses accumulation. ARIMA($p,0,q$) is ARMA($p,q$), while ARIMA($0,1,0$) without drift is a random walk. Forecasts for the differenced target must be accumulated back to the level scale.
<!-- bilingual-en:end -->

ARIMA 的定义不替研究者选择 $d$。差分不足会留下随机趋势，差分过多会制造短期相关并可能把 MA 根推到不可逆边界；决策与精确反例见[[最小差分原则]]。不同 $d$ 还会改变 likelihood 所使用的数据，所以不能把跨差分阶的 AIC 当作无条件可比的自动裁判。
<!-- bilingual-en:start -->
The model definition does not select $d$. Underdifferencing leaves stochastic trend; overdifferencing can manufacture short-run dependence and reach an MA noninvertibility boundary. Different differencing orders also change the data entering the likelihood, so AIC should not be treated as an automatic cross-$d$ referee.
<!-- bilingual-en:end -->

季节 ARIMA 可再加入 $(1-B^s)^D$ 以及季节 AR、MA 多项式。普通差分与季节差分仍应由可解释的建模对象和[[最小差分原则]]约束，而不是机械叠加。
<!-- bilingual-en:start -->
Seasonal ARIMA adds seasonal differencing and seasonal AR/MA polynomials. Ordinary and seasonal differences must still be justified by an interpretable target and the minimum-differencing principle.
<!-- bilingual-en:end -->

> [!question]- 自检
> 在 ARIMA($2,1,1$) 中，ARMA($2,1$) 描述的是 $y_t$ 还是 $\Delta y_t$？
>
> **答案：** 描述 $\Delta y_t$。$d=1$ 先改变建模对象，之后 $p=2,q=1$ 才描述这个差分序列的短期依赖。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=258|课程讲义 pp. 258–259]]：核对 ARIMA($p,d,q$) 以 $d$ 阶差分序列作为平稳 ARMA 对象。
- [MIT OCW 18.S096, Lecture 8, p. 25](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/1926c83ecd7ea700f7cb63914c6d7c0f_MIT18_S096F13_lecnote8.pdf)：核对 integrated ARMA 的定义与 $(p,d,q)$ 三个阶数。
- [Hyndman & Athanasopoulos, FPP3 §9.5](https://otexts.com/fpp3/non-seasonal-arima.html)：核对非季节 ARIMA 多项式、阶数含义与季节扩展。
