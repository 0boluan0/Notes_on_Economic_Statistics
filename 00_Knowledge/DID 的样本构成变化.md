---
aliases:
  - "DID 中的样本构成变化可以伪装成处理效应"
  - DID composition changes
  - Repeated cross-section composition in DID
  - 双重差分样本构成
student_os: knowledge-atom
atom_id: ECON-DID-007
atom_type: threat
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[双重差分法]]"
leads_to:
  - "[[2×2 DID 识别 ATT]]"
  - "[[DID 协变量调整]]"
---

# DID 中的样本构成变化可以伪装成处理效应

<!-- bilingual-en:start -->
*Changes in sample composition can masquerade as treatment effects in DID*
<!-- bilingual-en:end -->

> [!summary] 核心威胁
> DID 比较的是四个样本均值。若政策前后进入这些均值的人、企业或地区发生系统变化，均值的差可能来自“换了一批观察对象”，而不是同一总体的结果发生变化。
>
> <!-- bilingual-en:start -->
> DID compares four sample means. If the people, firms, or places entering those means change systematically across periods, the contrast may reflect a different set of observed units rather than a change in outcomes for the same target population.
> <!-- bilingual-en:end -->

面板数据跟踪同一单位，构成问题主要表现为退出、缺失或处理诱发的存活选择。重复截面每期重新抽样，本来就不是同一批个体；它依赖的是各组在每期都代表同一个稳定总体，或者研究者已明确把人口构成变化纳入目标。两种数据都能做 DID，但它们需要的抽样与构成论证不同。
<!-- bilingual-en:start -->
Panel data follow the same units, so composition problems usually appear as attrition, missingness, or treatment-induced survival. Repeated cross-sections draw new observations each period, so they require each group-period sample to represent the same stable population—or an explicit estimand that includes composition change. Both data structures can support DID, but their sampling and composition arguments are different.
<!-- bilingual-en:end -->

## 一个不会被固定效应修好的例子

一项城市补贴吸引高收入家庭迁入处理城市。若结果是平均消费，政策后均值上升可能来自原居民消费提高，也可能只是新居民收入更高。地区固定效应只能吸收不变的地区差异；时间固定效应只能吸收共同冲击，二者都不会把“谁住在这里”恢复到政策前。
<!-- bilingual-en:start -->
Suppose a city subsidy attracts high-income households into the treated city. If the outcome is average consumption, the post-policy increase may reflect higher consumption among original residents or simply the arrival of richer residents. Place fixed effects remove stable place differences, and time fixed effects remove common shocks; neither restores the pretreatment population composition.
<!-- bilingual-en:end -->

## 先决定问题，再决定处理方式

若研究问题是对政策前居民的影响，需要追踪原有单位、处理迁出缺失并说明选择假设。若问题是对处理城市当期人口平均结果的影响，迁入可能就是政策总效应的一部分。不能在看到结果后才在两种 estimand 之间切换。原始人数、进入退出率和组别构成趋势应与结果趋势一起报告。
<!-- bilingual-en:start -->
If the question concerns effects on pretreatment residents, the design must track those units, address attrition, and state the relevant selection assumptions. If the question concerns the average outcome among the treated city's current population, migration may be part of the policy's total effect. The estimand cannot be switched after seeing the result. Counts, entry and exit rates, and composition trends should be reported alongside outcome trends.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 学校改革后，成绩均值提高，但低成绩学生同时大量转出。为什么这不能直接视为学生学习效果？
>
> **答案：** 因为政策前后均值可能来自不同学生构成。要区分原学生的结果变化、转出选择和改革后在校总体的变化，并据此明确 estimand。

## 来源与核验

- Callaway & Sant’Anna (2021), [*Difference-in-Differences with Multiple Time Periods*](https://psantanna.com/files/Callaway_SantAnna_2020.pdf)，Appendix B：核验多期 DID 向重复截面数据的扩展及其抽样结构。
- Baker et al. (2026), [*Difference-in-Differences Designs: A Practitioner’s Guide*](https://www.aeaweb.org/articles?id=10.1257/jel.20251650)：交叉核验设计类型、目标参数、协变量与样本结构必须配套解释。
