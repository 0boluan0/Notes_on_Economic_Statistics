---
aliases:
  - "预测区间依赖创新分布或近似且通常低估参数不确定性"
  - ARMA prediction intervals
  - Forecast distribution uncertainty
  - Parameter uncertainty in forecasts
  - ARMA 预测区间
student_os: knowledge-atom
atom_id: TS-ARMA-023
atom_set: arma-modeling
atom_type: uncertainty-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARMA多步预测]]"
related:
  - "[[AR(1)多步预测]]"
  - "[[ARMA似然初值处理]]"
  - "[[ARMA残差诊断]]"
  - "[[波动率聚集]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# 预测区间依赖创新分布或近似且通常低估参数不确定性
<!-- bilingual-en:start -->
*Prediction intervals depend on an innovation distribution or approximation and often understate parameter uncertainty*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 已知参数、线性 ARMA 且创新 jointly Gaussian 时，预测误差也是 Gaussian，可用
> $$\hat y_{t+h|t}\pm z_{1-\alpha/2}\sqrt{\operatorname{Var}(e_{t+h|t})}$$
> 构造条件预测区间。若创新非 Gaussian，这个形状只是一种近似，不能从“残差不相关”自动推出。
> <!-- bilingual-en:start -->
> With known parameters and jointly Gaussian innovations, the linear ARMA forecast error is Gaussian and a symmetric normal interval follows from its variance. Without that distributional assumption, the same interval is only an approximation.
> <!-- bilingual-en:end -->

区间至少有三类不确定性：未来创新、参数估计、模型/结构变化。教科书递推方差通常只计算第一类，并把 $\phi,\theta,\sigma^2$ 当作已知；实际 plug-in 区间因此常比完整参数不确定性下更窄。小样本、高阶模型、近单位根时这项遗漏尤其重要。

厚尾、偏态或条件异方差会让对称 Gaussian 区间失准。可以依据明确的创新分布、残差 bootstrap 或模拟产生分布预测：只有当 fitted innovations 可合理视为 i.i.d./exchangeable 时，独立残差重采样才合适；若绝对值、平方或其他非线性依赖仍存在，就必须显式建模或用能保留该结构的方案。无论方法为何，区间覆盖率应在 rolling-origin 中按 horizon 校准，而不是只看样本内残差图。
<!-- bilingual-en:start -->
Textbook forecast variances usually include future-innovation uncertainty while treating parameters as known. Plug-in intervals can therefore be too narrow, especially near unit roots or in small samples. Independent residual resampling needs approximately i.i.d./exchangeable fitted innovations; remaining magnitude or nonlinear dependence must instead be modelled or preserved. Any simulation or bootstrap interval should be checked chronologically for coverage by horizon.
<!-- bilingual-en:end -->

> [!question]- 自检
> 残差 ACF 合格后直接使用 $\pm1.96$ 标准误，缺少哪项关键论证？
>
> **答案：** 缺少 forecast error 近似 Gaussian 的分布依据，还通常忽略参数估计与模型不确定性。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=126|课程讲义 pp. 126, 128–129]]：核对 Gaussian 区间公式与 coefficient uncertainty。
- [Hyndman & Athanasopoulos, FPP3 §5.5](https://otexts.com/fpp3/prediction-intervals.html)：核对分布预测、区间假设与非正态 residual 的 simulation/bootstrapping 处理。
- [Hyndman & Athanasopoulos, FPP3 §5.4](https://otexts.com/fpp3/diagnostics.html)：核对残差不相关与残差正态/同方差是不同诊断层次。
