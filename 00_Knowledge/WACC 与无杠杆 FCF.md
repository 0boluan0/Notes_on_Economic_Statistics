---
aliases:
  - "用 WACC 贴现时项目自由现金流应排除利息本金与股利以免重复计融资成本"
  - "Unlevered project free cash flow"
  - "Separate project cash flow from financing flow"
  - "项目自由现金流与融资流分离"
student_os: knowledge-atom
atom_id: CORP-CB-002
atom_set: capital-budgeting-investment-decisions
atom_type: accounting-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[项目增量现金流]]"
  - "[[贴现率口径匹配]]"
related:
  - "[[项目税盾、NWC 与终结现金流]]"
  - "[[净现值决策]]"
part_of:
  - "[[资本预算与投资决策.canvas]]"
---

# 用 WACC 贴现时项目自由现金流应排除利息本金与股利以免重复计融资成本
<!-- bilingual-en:start -->
*When WACC discounts project free cash flow, exclude interest, principal, and dividends to avoid counting financing twice*
<!-- bilingual-en:end -->

> [!summary] 口径配对
> 若用加权平均资本成本（WACC）折现企业口径的项目自由现金流，分子应先按项目**不含融资方式**的经营与投资现金流来写；债务利息、本金偿还和股利不再从中扣除，因为债权人与股东的必要回报已经进入 WACC。
>
> <!-- bilingual-en:start -->
> When WACC discounts firm-level project FCF, construct the numerator from operating and investment cash flows before financing. Do not deduct debt interest, principal repayment, or dividends again, because required returns to debt and equity are already represented in WACC.
> <!-- bilingual-en:end -->

一种常见的简化写法是

$$
FCF_t=EBIT_t(1-T)+Dep_t-CapEx_t-\Delta NWC_t.
$$

它从扣除折旧后的息税前经营利润出发，让折旧先通过税基影响现金税，再把非现金折旧全额加回，并扣除资本开支和净营运资本占用。若先从收入扣利息得到税后利润、再从现金流扣还本，最后仍用 WACC 折现，就会把融资成本分别放进分子和分母，通常造成重复计算。

这不是“融资永远与项目无关”。若使用股权现金流，就应纳入净借款并配股权资本成本；若用调整现值法（APV），可把无杠杆项目价值与融资副作用分开估值。关键不是永远删除融资流，而是让**现金流索取权、税收处理与贴现率属于同一个估值框架**。

> [!question]- 自检
> 项目预测已经从经营现金流中扣除了利息和偿还本金，分析师仍用 WACC 折现。最小修正是什么？
>
> **答案：** 要么恢复不含融资流的项目 FCF 并继续用 WACC，要么明确改用与含融资现金流相匹配的股权/APV 口径；不能混用。

## 来源与核验

- [[02_Economy/05_财务管理/2023年注册会计师全国统一考试辅导教材---财务成本管理 (中国注册会计师协会) (Z-Library).pdf#page=154|CPA《财务成本管理》第五章第三节，第 146 页]]：核对项目经营期现金流排除利息、本金与股利，因为折现率已包含融资成本。
- [MIT 15.401 Capital Budgeting, slides 5–10](https://ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008/06b2ce70ad26ae3d3be8d94145495be6_MIT15_401F08_lec18.pdf)：核对现金流而非会计利润、税后口径及项目风险与贴现率匹配。
- [MIT 15.414 Financial Management, Capital Budgeting, slides 30–32](https://ocw.mit.edu/courses/15-414-financial-management-summer-2003/db7e4784cad8bd727b087f077cab0eae_lec3_capital_budgeting.pdf)：核对经营利润、折旧、资本开支和营运资本构成的项目自由现金流口径。
