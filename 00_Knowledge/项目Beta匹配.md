---
aliases:
  - 项目 Beta 应匹配项目经营风险而不是机械继承公司权益 Beta
  - Project beta matching
  - Pure-play beta method
student_os: knowledge-atom
atom_id: INV-CAPM-009
atom_set: capm-systematic-risk
atom_type: valuation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[权益Beta杠杆调整]]"
  - "[[贴现率口径匹配]]"
related:
  - "[[WACC 与无杠杆 FCF]]"
  - "[[净现值决策]]"
part_of:
  - "[[CAPM、系统风险与资本成本.canvas]]"
---

# 项目 Beta 应匹配项目经营风险而不是机械继承公司权益 Beta
<!-- bilingual-en:start -->
*A project's beta should match its operating risk rather than mechanically inherit the firm's equity beta*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 公司历史权益 beta 是现有业务组合、经营杠杆和融资结构共同形成的。只有当新项目与公司平均业务风险相近，并准备维持相近融资政策时，公司 WACC 才是自然起点。项目若进入不同产品、客户、成本结构或经济周期，必须估计项目自己的系统风险。
> <!-- bilingual-en:start -->
> A firm's historical equity beta combines its existing business mix, operating leverage, and financing. Firm-wide WACC is a natural starting point only when a new project has similar average business risk and a similar financing policy. A project entering different products, customers, cost structures, or economic cycles needs its own systematic-risk estimate.
> <!-- bilingual-en:end -->

常用 pure-play（纯业务可比）方法是：选择主要经营同类业务的上市公司；估计其权益 beta；按一致口径去杠杆并剔除非经营现金影响；汇总经营资产 beta；最后按项目目标资本结构重新加杠杆。每一步都在建立可比性，不是把同一行业标签自动当作相同风险。
<!-- bilingual-en:start -->
The pure-play method selects listed firms focused on comparable operations, estimates their equity betas, unleverages them consistently, adjusts for non-operating cash, aggregates operating-asset betas, and finally releverages to the project's target financing. Each step constructs comparability; a shared industry label does not automatically imply equal risk.
<!-- bilingual-en:end -->

beta 只处理相对所选市场因子的协方差风险。币种、期限、国家违约或其他定价因子仍要与现金流和风险溢价口径一致；不能为了“更保守”而在 beta、市场溢价和现金流中重复加入同一风险。
<!-- bilingual-en:start -->
Beta addresses covariance risk relative to the selected market factor. Currency, horizon, sovereign default, and other priced factors must still be matched consistently to cash flows and premia. The same risk should not be added repeatedly to beta, the market premium, and cash flows under the label of conservatism.
<!-- bilingual-en:end -->

> [!question]- 自检
> 一家公用事业公司准备投资高波动的软件创业项目，为什么不能直接用现有公用事业 WACC？
>
> **答案：** 现有 WACC 反映公用事业业务与资本结构；软件项目的收入周期、成本刚性和市场协方差可能完全不同。应从纯业务可比公司的资产 beta 与项目目标融资出发。

## 来源与核验

- [MIT 15.401, Capital Budgeting, slides 5–10](https://ocw.mit.edu/courses/15-401-finance-theory-i-fall-2008/06b2ce70ad26ae3d3be8d94145495be6_MIT15_401F08_lec18.pdf)：核对项目风险与折现率匹配以及公司平均资本成本的适用边界。
- [Aswath Damodaran, Valuation Online, Session 5](https://www.stern.nyu.edu/~adamodar/pdfiles/valonlineslides/session5.pdf)：核对 bottom-up/pure-play beta、现金调整和目标杠杆流程。
- [[贴现率口径匹配]] 与 [[WACC 与无杠杆 FCF]]：复用现金流风险、币种、索取权和融资口径匹配原则。
