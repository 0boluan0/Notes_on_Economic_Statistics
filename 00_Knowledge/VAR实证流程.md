---
aliases:
  - "VAR 实证工作流从目标规格估计诊断走向识别解释"
  - VAR workflow
student_os: knowledge-atom
atom_id: TS-VAR-021
atom_set: vector-autoregression
atom_type: workflow
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR规格选择]]"
  - "[[VAR逐方程OLS]]"
related:
  - "[[协整与差分边界]]"
  - "[[结构VAR]]"
  - "[[SVAR识别条件]]"
  - "[[结构脉冲响应]]"
  - "[[Granger因果]]"
  - "[[Granger因果边界]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# VAR 实证工作流从目标规格估计诊断走向识别解释
<!-- bilingual-en:start -->
*A VAR workflow proceeds from objective and specification through estimation and diagnostics to identification and interpretation*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> VAR 实证分析应按“目标 → 规格 → 估计 → 诊断 → 必要时识别 → 有边界地解释”推进；预测任务不必强行结构化，结构 IRF 与 FEVD 却不能跳过识别。

1. **先定目标。** 区分联合预测、描述动态、检验 Granger 非因果，还是估计某个经济冲击的因果路径。不同目标决定变量集、损失函数与所需识别强度。
2. **构造规格。** 说明样本频率、变量变换、确定项、外生项和候选滞后。检查单位根与协整；存在稳定长期组合时，不要只因单序列为 $I(1)$ 就把全部水平信息机械删除。
3. **估计简约型。** 在正交性、秩和动态正则条件下逐方程 OLS，并保存 $\hat\Sigma_u$、参数不确定性与样本定义。VAR 在此阶段仍只是简约型联合动态模型。
4. **做模型诊断。** 检查剩余序列相关、稳定根、异方差或结构变化、异常值，以及关键结果对变量集和滞后阶数的稳健性。信息准则和未拒绝检验都不能单独证明规格正确。
5. **只在问题需要时识别。** 纯预测、总预测误差协方差或线性 Granger 检验可停留在简约型；经济结构 IRF、结构 FEVD 和历史冲击解释必须选择并论证短期、长期、符号或外部工具等方案。
6. **按对象解释。** 报告预测期限与误差；Granger 结论写明信息集和联合原假设；IRF/FEVD 写明冲击尺度、变量单位、累计口径、期限、区间及识别敏感性。

任一步失败都应回到前一步修改，而不是用更强的因果语言掩盖诊断问题。尤其不能因为模型名叫 VAR，就把它自动称为“结构模型”或“解决内生性”。

> [!question]- 自检
> 若任务只是提高四季度联合预测，是否必须先用 Cholesky 把所有创新命名为经济冲击？
>
> **答案：** 不必须。简约型 VAR 已能产生联合预测和总预测误差协方差；只有结构解释或按经济冲击分解时才需要识别。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)：核对规格、估计、诊断、预测与结构分析的完整路线。
- [Stock & Watson (2001), *Vector Autoregressions*](https://www.aeaweb.org/articles?id=10.1257/jep.15.4.101)：核对 VAR 在宏观实证中的预测、数据描述、结构分析与政策边界。
- [Sims (1986), *Are Forecasting Models Usable for Policy Analysis?*](https://fedinprint.org/item/fedmqr/41636)：核对从预测系统跨到政策分析时必须明确结构识别假设，而不能把简约型动态直接当政策干预。
- [Kilian & Lütkepohl (2017), *Structural Vector Autoregressive Analysis*](https://doi.org/10.1017/9781108164818)：核对从简约型到结构识别、IRF 与推断的工作流。
