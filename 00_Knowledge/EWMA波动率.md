---
aliases:
  - "在零条件均值或使用去均值创新时，EWMA 用统一衰减因子递推方差与协方差；它响应新冲击但没有固定长期方差目标"
student_os: knowledge-atom
atom_id: RM-VOL-004
atom_set: volatility-measurement
atom_type: forecasting-rule
status: source-checked
mastery_state: unassessed
requires:
  - "[[条件尺度与标准化冲击]]"
related:
  - "[[协方差矩阵半正定性]]"
  - "[[风险窗口权衡]]"
  - "[[GARCH一步方差预测]]"
  - "[[IGARCH平稳与矩边界]]"
  - "[[历史波动率]]"
leads_to:
  - "[[条件异方差：ARCH 与 GARCH.canvas|条件异方差：ARCH 与 GARCH]]"
part_of:
  - "[[波动率度量.canvas|波动率度量]]"
---

# 在零条件均值或使用去均值创新时，EWMA 用统一衰减因子递推方差与协方差；它响应新冲击但没有固定长期方差目标
<!-- bilingual-en:start -->
*With a zero conditional mean or demeaned innovations, EWMA recursively updates variances and covariances using one decay factor; it responds to new shocks but has no fixed long-run variance target*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> EWMA 对最近平方创新和交叉乘积给更高权重。若直接代入 raw return，就已经假设短期条件均值为零；否则应先形成创新。多资产应用必须用同一个 $\lambda$ 更新整张协方差矩阵，才能由半正定初值保留半正定性。

令收益向量的条件均值为 $\mu_t$，创新为

$$
\varepsilon_t=r_t-\mu_t.
$$

多资产 EWMA 一步更新是

$$
\Sigma_t
=\lambda\Sigma_{t-1}
+(1-\lambda)\varepsilon_{t-1}\varepsilon_{t-1}^{\mathsf T},
\qquad 0<\lambda<1.
$$

对单资产，这退化为

$$
h_t=\lambda h_{t-1}+(1-\lambda)\varepsilon_{t-1}^2.
$$

当条件均值明确固定为零时才可写 $\varepsilon_{t-1}=r_{t-1}$。若均值可预测却仍把 raw return 平方，方向性均值会被错误计入方差。

使用同一个 $\lambda$ 时，若 $\Sigma_{t-1}\succeq0$，则 $\Sigma_t$ 是两个半正定矩阵的凸组合，仍满足 $\Sigma_t\succeq0$。逐资产、逐 pair 任意选择不同衰减因子，通常不再等价于一次矩阵凸组合，可能得到不是合法协方差矩阵的结果。

递推展开后，最近一期创新外积的权重是 $1-\lambda$，再早一期是 $(1-\lambda)\lambda$。某个冲击权重降为原来一半所需的 half-life 为

$$
h_{1/2}=\frac{\log(1/2)}{\log\lambda}.
$$

当 $\lambda=0.94$ 时，$h_{1/2}\approx11.2$ 个交易日。不要把 $1/(1-\lambda)=16.67$ 无说明地叫作唯一“有效历史长度”：平均 lag、Kish effective sample size 和把残余权重降到 1% 的 tolerance horizon 是不同定义；RiskMetrics 在 1% tolerance 下给 $\lambda=0.94$ 约 74 天。

EWMA 递推没有常数项，且平方冲击与滞后方差的系数和为

$$
(1-\lambda)+\lambda=1.
$$

因此它位于 integrated GARCH 的持久性边界，没有单独的有限长期目标 $\bar h$ 把预测拉回去。这里说的是递推结构，不等于把 [[IGARCH平稳与矩边界]] 中关于严格平稳和矩存在性的条件全部省略；更一般的 ARCH/GARCH 模型条件仍由 [[条件异方差：ARCH 与 GARCH.canvas|条件异方差主题图]] 承载。

> [!question]- 最小数值自检
> $\lambda=0.94$、昨日方差 $h_{t-1}=0.0001$、昨日去均值创新 $\varepsilon_{t-1}=2\%$。今日 EWMA 方差和波动率是多少？这一结果会自动向某个固定长期方差回归吗？
>
> **答案：** $h_t=0.94(0.0001)+0.06(0.02^2)=0.000118$，$\sqrt{h_t}\approx1.0863\%$。不会；该零常数、系数和为一的递推没有单独长期方差目标。

> [!warning] 真实应用仍需验证
> 必须验证收益与均值口径、初始化、$\lambda$、缺失与异步数据处理、协方差矩阵维度、目标 horizon 和预测损失。$0.94$ 是 RiskMetrics 对其日频数据和当时准则得到的历史选择，不是所有资产、频率与业务目标的通用最优值。本轮独立审计已核对递推、半正定边界、数值锚点、来源定位与适用范围，状态为 `source-checked`；它没有验证具体生产数据或预测效果，掌握度仍为 `unassessed`。

## 来源与核验

- J.P. Morgan/Reuters, [*RiskMetrics—Technical Document*, 4th ed. (1996), Chapter 5, especially pp. 80–93 and 103–112](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a)：原始方法文档核对 EWMA 方差/协方差递推、日频均值设零、整张协方差矩阵的一致衰减因子、$\lambda=0.94$ 的历史选择及 1% tolerance 下约 74 天。
- [[协方差矩阵半正定性]]：复用 $a^{\mathsf T}\Sigma a\ge0$ 的合法协方差边界；本卡给出同 $\lambda$ 凸组合保留半正定性的直接证明。
- [[IGARCH平稳与矩边界]] 与 [[GARCH一步方差预测]]：复用 integrated boundary、矩条件和一般 GARCH 一步预测，不在本卡复制 TS-VOL 原子。
