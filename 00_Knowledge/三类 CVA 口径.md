---
aliases:
  - "CVA 风险资本、对手方违约资本与会计 CVA 是三个不同口径"
  - CVA risk capital versus CCR default capital and accounting CVA
  - CVA 三种口径
student_os: knowledge-atom
atom_id: MB-BAS-017
atom_set: basel-capital-liquidity-regulation
atom_type: regulatory-distinction
status: source-checked
mastery_state: unassessed
requires:
  - "[[SA-CCR]]"
  - "[[单边CVA]]"
related:
  - "[[QCCP资本处理]]"
  - "[[RWA 聚合与 12.5 换算]]"
  - "[[DVA与双边估值]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# CVA 风险资本、对手方违约资本与会计 CVA 是三个不同口径
*CVA risk capital, counterparty-default capital and accounting CVA are three distinct measures*

> [!summary] 价值调整与两类资本
> 会计 CVA 是把对手方信用风险纳入衍生品公允价值的调整；CCR 违约资本覆盖对手方违约导致的损失；监管 CVA 风险资本覆盖信用利差及相关市场因子变化使 CVA 本身波动的风险。三者有关联，但损失事件、计量目标和规则不同。

一名对手尚未违约，其信用利差上升就可能使 CVA 扣减的绝对值增加、衍生品公允价值下降并产生会计损失；若采用把 CVA 本身记为负数的符号约定，也可说“CVA 变得更负”。这正说明 CVA 风险不等于违约风险。反之，CCR EAD、PD、LGD 资本关注违约损失，不应被“已经记了会计 CVA”自动抵消。

当前 Basel CVA 框架包括基础法（BA-CVA）、在获批条件下的标准法（SA-CVA），以及对不重要 CVA 风险交易使用 100% 对手方信用风险 RWA 的有限重要性处理。课程中以信用利差平行移动、VaR 和增量风险描述的 2010 年高级 CVA 方法应标为历史；现行框架更系统地处理 CVA 对信用利差、利率、汇率等风险因子的敏感度，不能再写成“其他市场因子一律不计 CVA 资本”。

## 边界

- DVA 是自身信用风险对公允价值的调整，不能与监管 CVA 资本或对手方违约资本当成同义词。
- 会计准则、Basel 资本与内部经济资本可使用不同估值、净额、对冲和期限口径。
- 具体交易是否豁免、采用 BA-CVA 还是 SA-CVA，须查适用规则和批准状态。

> [!question]- 自检
> 对手方没有违约，银行为什么仍可能发生 CVA 损失？
>
> **答案：** 对手信用利差或其他影响未来敞口/折现的市场因子变化，会使 CVA 公允价值重估。

## 来源与核验

- [Basel Committee, Targeted revisions to the CVA risk framework](https://www.bis.org/bcbs/publ/d507.htm)：核对当前 BA-CVA、SA-CVA 及市场风险因子处理。
- [FSI, Counterparty credit risk in Basel III](https://www.bis.org/fsi/fsisummaries/ccr_in_b3.htm)：核对 CCR 默认资本与 CVA 风险资本的区别。
- [[单边CVA]] 与 [[DVA与双边估值]]：复用会计/经济 CVA、DVA 与双边 close-out 的稳定定义。
- 口径核验日：2026-08-29。
