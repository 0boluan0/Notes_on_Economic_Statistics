---
aliases:
  - "理论 ACF 与 PACF 的截尾拖尾只能生成候选阶数"
  - ACF and PACF identification
  - ACF PACF cutoff and tailoff
  - Partial Autocorrelation Function
  - ARMA order identification from ACF and PACF
student_os: knowledge-atom
atom_id: TS-ARMA-012
atom_set: arma-modeling
atom_type: diagnostic-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[AR(p)模型]]"
  - "[[MA(q)模型]]"
  - "[[自协方差与ACF]]"
  - "[[偏自相关函数]]"
related:
  - "[[ARMA(p,q)模型]]"
  - "[[Box-Jenkins流程]]"
  - "[[ACF信息边界]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# 理论 ACF 与 PACF 的截尾拖尾只能生成候选阶数
<!-- bilingual-en:start -->
*Population ACF/PACF cutoff patterns generate candidate orders; sample plots do not identify a model by themselves*
<!-- bilingual-en:end -->

> [!summary] 原子诊断
> 对因果、可逆且最小的理想总体模型：
> - AR($p$)：ACF 拖尾，PACF 在 $p$ 阶后截尾；
> - MA($q$)：ACF 在 $q$ 阶后截尾，PACF 拖尾；
> - 一般 ARMA($p,q$)：两者通常都拖尾。
> <!-- bilingual-en:start -->
> In ideal population models, an AR($p$) has a tailing ACF and a PACF cutoff after $p$; an MA($q$) has an ACF cutoff after $q$ and a tailing PACF; a mixed ARMA generally has two tails.
> <!-- bilingual-en:end -->

[[自协方差与ACF|ACF]] 保留本期与第 $k$ 阶滞后之间全部线性关系；[[偏自相关函数|PACF]] 先剔除中间滞后的线性传递。理论“截尾”说的是总体值在某阶之后恰为零，不是样本图上每个点都落在零线上。

有限样本 ACF/PACF 有抽样误差，多个滞后同时查看还会产生偶然显著点。近单位根、季节性、混合 ARMA、结构突变和前处理也会模糊形状。因此图形只用于提出少量候选；候选仍要经过可比信息准则、参数边界、残差联合诊断与时间顺序外样本评估。一个漂亮的截尾图不是数据生成方程的唯一证书。
<!-- bilingual-en:start -->
The ACF retains all linear dependence transmitted across a lag, while the [[偏自相关函数|PACF]] removes the intervening lags. Population cutoff means exact zeros; sample estimates fluctuate. Near-unit roots, seasonality, mixed models, breaks, preprocessing, and multiple visual comparisons blur the patterns. Use the plots to propose candidates, then estimate and diagnose them.
<!-- bilingual-en:end -->

> [!question]- 自检
> 样本 PACF 在第 4 阶越过置信带一次，是否已经证明真实过程是 AR(4)？
>
> **答案：** 没有。单个样本尖峰可能是抽样波动；还要比较相邻候选、检查稳定性与残差，并做外样本评估。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=97|课程讲义 pp. 97–110]]：核对 AR、MA、ARMA 的 ACF/PACF 理论形状与样本图置信带。
- [Hyndman & Athanasopoulos, FPP3 §9.5](https://otexts.com/fpp3/non-seasonal-arima.html)：核对 PACF 定义、典型截尾模式与用图选择候选阶数的边界。
- [[ACF信息边界]]：承载二阶线性图形不能识别完整过程的边界。
