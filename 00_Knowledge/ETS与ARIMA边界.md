---
aliases:
  - "ETS 与 ARIMA 只在受限的线性加法子类中重叠"
  - ETS versus ARIMA overlap
  - ETS ARIMA partial equivalence
  - ETS 与 ARIMA 部分等价
student_os: knowledge-atom
atom_id: TS-ETS-009
atom_set: exponential-smoothing-ets
atom_type: model-family-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[ETS三轴模型]]"
  - "[[ARIMA模型]]"
related:
  - "[[SES-ARIMA受限等价]]"
  - "[[滚动起点评估]]"
part_of:
  - "[[指数平滑与 ETS.canvas]]"
---

# ETS 与 ARIMA 只在受限的线性加法子类中重叠
<!-- bilingual-en:start -->
*ETS and ARIMA overlap only in a restricted linear additive subclass*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 部分 additive-error、线性状态更新的 ETS 可写成带参数限制的 ARIMA；含 multiplicative error、multiplicative seasonality 等非线性 ETS 没有普通线性 ARIMA 对应。反向也不成立：许多 ARIMA，尤其平稳模型，没有 ETS 对应。
> <!-- bilingual-en:start -->
> Some linear additive-error ETS models have restricted ARIMA representations. Nonlinear ETS models lack ordinary linear ARIMA counterparts, while many ARIMA models have no ETS counterpart either.
> <!-- bilingual-en:end -->

FPP3 给出的典型重叠包括：

- ETS$(A,N,N)$ 对应受限 ARIMA$(0,1,1)$；
- ETS$(A,A,N)$ 对应受限 ARIMA$(0,2,2)$；
- ETS$(A,A_d,N)$ 对应受限 ARIMA$(1,1,2)$；
- additive-error、additive-seasonal ETS 也有带季节差分和大量参数限制的 ARIMA 表示。

“对应”不是说两个完整模型族相同。ETS 以 level/trend/seasonal states 组织模型，ARIMA 以差分后的 AR/MA 多项式组织模型；映射只覆盖特定状态方程和参数子空间。ARIMA 可描述很多平稳自相关结构，而 multiplicative ETS 的误差尺度或季节更新是非线性的，普通线性 ARIMA 无法复制。

因此不能因为一个 SES 例子可化为 ARIMA 就宣布“ETS 只是 ARIMA”，也不能说“ARIMA 总比 ETS 更一般”。软件报告的跨族 AICc 只有在 observed-data likelihood 的样本、变换、常数与初始化口径确实一致时才可比较；FPP3 的 ETS/ARIMA 实现不满足这一直接可比条件。实务选择应在共同 forecast target、horizon、样本与损失下使用 rolling-origin，并同时检查预测形状和残差。
<!-- bilingual-en:start -->
The equivalence map covers particular state equations and restricted parameter subspaces, not the full model families. ETS organises dynamics through evolving components; ARIMA organises them through differenced autoregressive and moving-average polynomials. Neither family contains the other.
<!-- bilingual-en:end -->

> [!question]- 自检
> 发现 ETS$(A,N,N)$ 有 ARIMA 表示后，能否推断 ETS$(M,A,M)$ 也必有普通 ARIMA 表示？
>
> **答案：** 不能。后者含乘法误差与乘法季节的非线性结构，不在已知线性加法重叠子类中。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §9.10](https://otexts.com/fpp3/arima-ets.html)：核对两族重叠图、各典型等价映射、非线性 ETS 与平稳 ARIMA 的双向非包含边界。
- Hyndman et al. (2008), *Forecasting with Exponential Smoothing: The State Space Approach*, Springer, Ch. 11：核对 ETS 的 ARIMA reduced forms 与参数限制。
