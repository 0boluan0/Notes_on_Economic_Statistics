---
aliases:
  - "VAR 的变量集合与滞后阶数应由目标和诊断共同决定"
  - VAR specification
  - VAR lag selection
student_os: knowledge-atom
atom_id: TS-VAR-004
atom_set: vector-autoregression
atom_type: workflow
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR(p)模型]]"
related:
  - "[[ARMA信息准则]]"
  - "[[ARMA残差诊断]]"
  - "[[VAR参数量]]"
  - "[[VAR实证流程]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# VAR 的变量集合与滞后阶数应由目标和诊断共同决定
<!-- bilingual-en:start -->
*The variables and lag order of a VAR should be chosen jointly from the objective and diagnostics*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> VAR 没有脱离问题背景的“正确变量表”和“正确滞后阶数”。预测目标、结构问题、数据频率、样本长度、整合性质、信息准则与残差诊断必须共同约束规格。

变量选择先服务于研究对象。若目标是预测，保留能稳定改善样本外预测的信息可能最重要；若目标是结构响应，信息集还必须足以让所命名冲击和排除限制可信。遗漏共同驱动变量可能制造虚假的 Granger 关系或污染冲击，机械加入大量变量又会因 $K^2p$ 的参数增长损害估计精度。

滞后阶数反映数据频率与动态记忆。AIC、AICc、BIC 等只能在相同因变量、样本区间和可比似然的候选规格之间排序，不能独立裁决。选定候选阶数后，还应检查残差是否仍有可检测的序列相关、模型是否稳定、参数是否过多，以及关键结论对相邻阶数是否稳健。未拒绝残差检验只表示在所检范围内没有发现剩余线性相关，不证明模型是真实的数据生成过程。

变量若有单位根或协整，还要先决定水平 VAR、差分 VAR 或 VECM 哪个表示保留了问题所需的信息。确定项、季节项、结构突变和外生控制也会改变似然、自由度和动态解释，不能在选完 $p$ 后当作无关附加项。

> [!question]- 自检
> BIC 在 VAR(1) 到 VAR(8) 中选择 VAR(2)，是否足以宣布 VAR(2) 是最终规格？
>
> **答案：** 不足。还需确认候选模型可比，并结合目标、残差序列相关、稳定性、变量变换和结论稳健性诊断。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 4 章：核对 VAR 阶数选择与模型充分性检查。
- [[ARMA信息准则]]、[[ARMA残差诊断]]：复用信息准则与残差诊断的边界。
