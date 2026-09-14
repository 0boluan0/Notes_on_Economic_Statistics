---
aliases:
  - "Basel I 五档风险权重与 Cooke 比率是历史口径而非当前通用权重表"
  - Basel I risk weights and Cooke ratio
  - Basel I 五档权重
student_os: knowledge-atom
atom_id: MB-BAS-009
atom_set: basel-capital-liquidity-regulation
atom_type: historical-method
status: source-checked
mastery_state: unassessed
requires:
  - "[[Basel 框架演进]]"
  - "[[风险加权资本率]]"
related:
  - "[[RWA 聚合与 12.5 换算]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# Basel I 五档风险权重与 Cooke 比率是历史口径而非当前通用权重表
*Basel I's five risk-weight buckets and Cooke ratio are historical conventions, not today's universal risk-weight table*

> [!summary] 历史计算入口
> 1988 年 Basel I 使用 0%、10%、20%、50%、100% 五个风险权重；其中 10% 属于国内公共部门实体等项目的国家裁量，所以许多课堂简表只列更常见的 0%、20%、50%、100% 四档。表外项目先用信用转换因子折成信用等价额；Cooke 总资本率以合格资本除以信用风险加权资产，历史最低值为 8%。

设资产为主权债 100（题设权重 0%）、银行债权 100（20%）、住宅按揭 100（50%）、企业贷款 100（100%），则历史 Basel I RWA 为 $0+20+50+100=170$；8% 总资本基线对应 13.6。这个算例只在题设明确采用 Basel I 权重时成立。

五档设计易计算、提高了跨国可比性，却把同一档内的大量风险差异压平。它没有“消灭监管套利”，反而可能鼓励银行在同一权重档内转向更高收益、更高真实风险的资产，并推动表外化和证券化结构优化。Basel II 以后方法更复杂，当前权重必须查适用框架。

## 边界

- 8% 是历史 Cooke 总资本最低值，不表示 Tier 2 另有独立 4% 最低要求。
- Basel I 的类别与权重有更细的条件和国家裁量；“所有主权 0%、所有按揭 50%”是错误泛化。
- 历史权重表不能用于判断 2026 年某银行合规。

> [!question]- 自检
> 为什么同为 100% 权重的两笔企业贷款仍可能具有非常不同的真实信用风险？
>
> **答案：** Basel I 桶位粗，不能反映每个借款人的 PD、LGD、期限、抵押和集中差异。

## 来源与核验

- [Basel Committee, International convergence of capital measurement and capital standards (1988)](https://www.bis.org/publ/bcbs04.pdf)：核对五个历史风险权重、国家裁量、表外转换和 Cooke 比率。
- [[02_Economy/07_金融机构与风险管理/15_《巴塞尔协议I II》和 偿付能力法案II.md#2. basel I|课程：Basel I 历史计算]]：保留课堂算例，但不外推为当前规则。
- 当前整合框架对照日：2026-08-29；本卡数值仍明确限定在 1988 年历史口径。
