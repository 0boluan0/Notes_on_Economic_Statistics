---
aliases:
  - "稳定 ADL 的长期阶跃乘数汇总直接滞后与自回归传播"
  - "The long-run step multiplier of a stable ADL combines distributed effects with autoregressive propagation"
  - "ADL long-run multiplier"
  - "ARDL long-run multiplier"
student_os: knowledge-atom
atom_id: TS-DYN-008
atom_set: dynamic-regression-intervention
atom_type: proposition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[回归预测与动态回归.canvas]]"
  - "[[差分方程与滞后算子.canvas]]"
requires:
  - "[[自回归分布滞后]]"
  - "[[分布滞后乘数]]"
related:
  - "[[干预函数类型]]"
  - "[[一阶仿射递推]]"
leads_to:
  - "[[动态回归规格流程]]"
---

# 稳定 ADL 的长期阶跃乘数汇总直接滞后与自回归传播
<!-- bilingual-en:start -->
*The long-run step multiplier of a stable ADL combines distributed effects with autoregressive propagation*
<!-- bilingual-en:end -->

> [!summary] 核心命题
> 对
> $$
> y_t=\alpha+\rho y_{t-1}+\beta_0x_t+\beta_1x_{t-1}+u_t,
> $$
> 一次性单位脉冲的响应为 $\delta_0=\beta_0$、$\delta_1=\rho\beta_0+\beta_1$，以后按 $\delta_h=\rho\delta_{h-1}$ 传播。若 $x$ 从某期起永久增加一个单位且 $|\rho|<1$，长期水平响应是脉冲响应之和：
> $$
> \sum_{h=0}^{\infty}\delta_h
> =\frac{\beta_0+\beta_1}{1-\rho}.
> $$
> <!-- bilingual-en:start -->
> In an ADL(1,1), a one-period unit pulse has impact response $\beta_0$ and next-period response $\rho\beta_0+\beta_1$, followed by autoregressive decay. A permanent unit step sums all pulse responses, giving $(\beta_0+\beta_1)/(1-\rho)$ when $|\rho|<1$.
> <!-- bilingual-en:end -->

这里必须把“脉冲在下一期的响应”和“阶跃在下一期的总水平响应”分开。脉冲下一期只有 $\rho\beta_0+\beta_1$；永久阶跃在下一期还有新到达的当期输入 $\beta_0$，所以总响应是 $\beta_0+\beta_1+\rho\beta_0$。长期公式之所以是脉冲响应的总和，正是因为阶跃可以看作每期叠加一个新脉冲。
<!-- bilingual-en:start -->
The next-horizon response to a pulse is not the next-horizon level response to a step. A step also supplies a fresh contemporaneous unit at the next date, so its horizon-one response is $\beta_0+\beta_1+\rho\beta_0$. A permanent step is a sequence of pulses, which is why its long-run effect sums the pulse responses.
<!-- bilingual-en:end -->

也可以直接令系统到达新稳态。永久变化前后相减得到
$$
(1-\rho)\Delta\bar y=(\beta_0+\beta_1)\Delta\bar x.
$$
这条稳态法与响应求和法应给出同一个结果。若 $|\rho|\ge1$，初始效应不衰减，有限长期乘数一般不存在；若参数随制度变化，旧样本乘数也不能无条件外推。
<!-- bilingual-en:start -->
The same result follows by comparing steady states: $(1-\rho)\Delta\bar y=(\beta_0+\beta_1)\Delta\bar x$. If the autoregressive part is not stable, a finite long-run multiplier generally does not exist. Structural change also prevents an automatic extrapolation of the historical multiplier.
<!-- bilingual-en:end -->

> [!example] 数值锚点
> 若 $(\rho,\beta_0,\beta_1)=(0.5,2,1)$，脉冲响应从 $2,2,1,0.5,\ldots$ 开始，总和为 $6$。永久阶跃的水平响应从 $2,4,5,5.5,\ldots$ 逐步趋近 $6$。
> <!-- bilingual-en:start -->
> With $(\rho,\beta_0,\beta_1)=(0.5,2,1)$, the pulse responses begin $2,2,1,0.5,\ldots$ and sum to $6$. The step responses begin $2,4,5,5.5,\ldots$ and converge to $6$.
> <!-- bilingual-en:end -->

> [!question]- 自检
> 为什么 $\rho\beta_0+\beta_1$ 不是永久阶跃在下一期的完整响应？
>
> **答案：** 它只是原始脉冲传播到下一期的响应；永久阶跃在下一期又新增一个当期输入，其直接效应为 $\beta_0$。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=197|课程讲义 pp. 197–199]]：核对 AR(1) 干预模型的脉冲传播、阶跃部分和及 $c_0/(1-a_1)$ 长期极限。
- [[分布滞后乘数]] 与 [[自回归分布滞后]]：本卡在课程的 $\beta_1=0$ 特例上加入一个解释变量滞后，并逐期推导一般 ADL(1,1) 结果。
