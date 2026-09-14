---
aliases:
  - "GARCH 一步方差预测使用当期平方创新并决定条件区间宽度"
  - One-step GARCH variance forecast
  - GARCH one-step prediction interval
  - 一步波动率预测
student_os: knowledge-atom
atom_id: TS-VOL-016
atom_set: conditional-volatility
atom_type: forecasting-rule
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH(p,q)模型]]"
  - "[[ARMA预测区间]]"
related:
  - "[[GARCH多步方差预测]]"
  - "[[Student-t GARCH标准化]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH 一步方差预测使用当期平方创新并决定条件区间宽度
<!-- bilingual-en:start -->
*A one-step GARCH variance forecast uses the current squared innovation and sets the conditional interval width*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> 对 GARCH(1,1)
> $$h_{t+1}=\omega+\alpha\varepsilon_t^2+\beta h_t,$$
> 时点 $t$ 已知 $\varepsilon_t$ 与滤波方差 $h_t$，所以
> $$\hat h_{t+1\mid t}=\omega+\alpha\hat\varepsilon_t^2+\beta\hat h_t.$$
> 一步预测必须使用刚观察到的 **$t$ 期**平方创新；写成 $\varepsilon_{t-1}^2$ 会平白落后一格。

若 $\hat\mu_{t+1\mid t}$ 是条件均值预测，且标准化未来创新的条件 $p$ 分位数为 $q_p$，则已知参数下观测的 plug-in 条件分位数是
$$Q_p(y_{t+1}\mid\mathcal F_t)=\hat\mu_{t+1\mid t}+q_p\sqrt{\hat h_{t+1\mid t}}.$$
相应的 $1-\eta$ 双侧预测区间为
$$\left[\hat\mu_{t+1\mid t}+q_{\eta/2}\sqrt{\hat h_{t+1\mid t}},\ \hat\mu_{t+1\mid t}+q_{1-\eta/2}\sqrt{\hat h_{t+1\mid t}}\right].$$
Gaussian 95% 对称近似才简化为 $\hat\mu_{t+1\mid t}\pm1.96\sqrt{\hat h_{t+1\mid t}}$；$h$ 只给尺度，Student-$t$、偏态或经验分布仍需要各自的分位数。

这一区间把参数、均值模型和滤波状态当作已知。估计不确定性、尾部错设和断点会改变覆盖率。对多步均值预测，误差还累积多个未来创新，不能只拿单个 $h_{t+h\mid t}$ 开平方。

> [!question]- 自检
> 今日冲击异常大但昨日冲击很小，正确的一步方差更新应立即反应还是再等一期？
>
> **答案：** 立即反应；$h_{t+1\mid t}$ 使用今日已实现的 $\varepsilon_t^2$。

## 来源与核验

- [Bollerslev (2023), “GARCH Musings”](https://public.econ.duke.edu/~boller/Papers/GARCH_Musings.pdf)：核对 $\sigma_{t+1}^2=\omega+\alpha\varepsilon_t^2+\beta\sigma_t^2$ 的索引。
- [[01_Math/06_时间序列分析/lecture.pdf#page=173|课程讲义 p. 173]]：原幻灯片把一步项错印为 $\varepsilon_{t-1}^2$；此卡按模型递推纠正。
- [[ARMA预测区间]]：复用预测区间的分布与 plug-in 边界。
