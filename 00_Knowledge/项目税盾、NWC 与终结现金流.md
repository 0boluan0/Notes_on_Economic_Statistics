---
aliases:
  - "折旧税盾营运资本与终结回收必须按真实税法和回收时点进入项目现金流"
  - "Depreciation tax shield working capital and terminal cash flow"
  - "Project cash-flow timing components"
  - "折旧税盾与终结现金流"
student_os: knowledge-atom
atom_id: CORP-CB-003
atom_set: capital-budgeting-investment-decisions
atom_type: accounting-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[项目增量现金流]]"
related:
  - "[[WACC 与无杠杆 FCF]]"
  - "[[名义与实际贴现]]"
part_of:
  - "[[资本预算与投资决策.canvas]]"
---

# 折旧税盾营运资本与终结回收必须按真实税法和回收时点进入项目现金流
<!-- bilingual-en:start -->
*Depreciation tax shields, working capital, and terminal recovery enter project cash flow only under the actual tax and recovery timing*
<!-- bilingual-en:end -->

> [!summary] 三个时点边界
> 折旧不是现金流入；它只有在税法允许扣除且企业能在相应时点利用扣除时，才通过少缴税形成现金税盾。净营运资本在真正占用现金时流出、在真正释放时流入；项目终结出售资产则按售价扣除基于税基的税款后计入。
>
> <!-- bilingual-en:start -->
> Depreciation is not a cash inflow. It creates a cash tax shield only when the deduction is legally available and usable at that date. NWC is an outflow when cash is committed and an inflow when actually released; terminal sale proceeds enter after tax based on the applicable tax basis.
> <!-- bilingual-en:end -->

在税率为 $T$、折旧扣除当期可用的简化情形，折旧税盾为 $Dep_tT$。若企业没有足够应税所得、亏损结转规则推迟使用，或税务折旧与会计折旧不同，税盾金额和日期都必须随实际规则调整，不能机械套 $Dep\times T$。

净营运资本通常写作经营性流动资产减经营性流动负债。增加存货和应收占用现金，增加应付可部分融资；因此项目现金流扣除的是 $\Delta NWC$，而不是销售额本身。期末“收回全部营运资本”只是预测假设：坏账、滞销存货或清偿义务会使实际回收低于账面数。

在处置损益适用同一税率、且当期可以使用相关税收收益或承担税款的简化情形，资产处置的税后终结流为

$$
\text{after-tax sale proceeds}=S-T(S-B),
$$

其中 $S$ 是售价、$B$ 是适用税基；当 $S<B$ 且损失确实可抵税时，后一项成为税收节省。若税率分档、损失利用受限或纳税时点不同，就要改用实际税收现金流。还需另外纳入拆除、弃置、环境修复等真实终结现金流。

> [!question]- 自检
> 设备账面折旧 30、税务折旧 20，税率 25%，且税务扣除当期可用。折旧税盾应按多少计算？
>
> **答案：** 按税务允许扣除的 20 计算，为 5；会计折旧 30 不能直接决定现金税盾。

## 来源与核验

- [[02_Economy/05_财务管理/2023年注册会计师全国统一考试辅导教材---财务成本管理 (中国注册会计师协会) (Z-Library).pdf#page=154|CPA《财务成本管理》第五章第三节，第 146–155 页]]：核对建设/经营/终结现金流、营运资本、折旧抵税及售价相对税基的终结税收。
- [MIT 15.414 Financial Management, Capital Budgeting, slides 19–23 and 28–32](https://ocw.mit.edu/courses/15-414-financial-management-summer-2003/db7e4784cad8bd727b087f077cab0eae_lec3_capital_budgeting.pdf)：核对 $FCF$ 构造、$\Delta NWC$、资产出售和折旧税收边界。
