---
aliases:
  - "银行账簿利率风险必须同时用 NII 与 EVE 视角并纳入行为选择权"
  - Interest Rate Risk in the Banking Book
  - IRRBB
  - 银行账簿利率风险
  - IRRBB NII and EVE entry
  - 银行账簿利率风险双视角
student_os: knowledge-atom
atom_id: MB-ALM-013
atom_set: commercial-bank-alm
atom_type: risk-entry
status: source-checked
mastery_state: unassessed
requires:
  - "[[合同期限、重定价期限与行为期限]]"
related:
  - "[[重定价缺口]]"
  - "[[久期缺口]]"
  - "[[IRRBB基差]]"
  - "[[无到期存款行为]]"
  - "[[提前还款选择权]]"
  - "[[银行利润归因]]"
part_of:
  - "[[商业银行资产负债管理.canvas]]"
  - "[[银行账簿利率风险.canvas|银行账簿利率风险]]"
---

# 银行账簿利率风险必须同时用 NII 与 EVE 视角并纳入行为选择权
*IRRBB requires both NII and EVE perspectives with behavioural optionality*

> [!summary] 风险入口
> 银行账簿利率风险（IRRBB）至少要分两种视角：NII 观察规划期内资产收益与负债成本怎样重定价，EVE 观察全部相关资产、负债与表外现金流现值怎样随利率变化。两者时间窗和对象不同，冲击结果不必同号。

长期固定利率贷款由快速重定价存款融资时，加息可能先抬高负债成本、压缩 NII；若资产端久期明显长于负债端且其他项目不足以抵消，资产价值降幅更大，EVE 也会下降。若资产是浮息、负债利率调整较慢，短期 NII 又可能改善。只报告“一年 gap”无法推出全期限经济价值。

无到期存款的稳定余额、存款 beta、客户转向高息产品、贷款提前还款和取款都含行为选择权；不同参考利率的变化还产生基差风险。管理层必须把这些假设写明并压力测试；从 [[银行账簿利率风险.canvas|银行账簿利率风险总图]] 可进入各项测量和行为模型。

## 边界

- IRRBB 不是交易账簿市场风险的同义词；账簿边界和资本处理按适用规则确定。
- NII 是收益视角，不应直接解释为银行经济价值；EVE 也不是未来会计利润的逐期预测。
- 本卡是 ALM 入口，不复制 Duration Gap、曲线冲击和行为模型细节。

> [!question]- 自检
> 为什么浮息资产快速重定价可能改善短期 NII，却仍不能保证 EVE 不受损？
>
> **答案：** NII 看短期收入重定价；EVE 折现全部未来现金流，长期固定现金流或负债结构仍可能产生价值损失。

## 来源与核验

- [Basel Committee, Interest rate risk in the banking book](https://www.bis.org/bcbs/publ/d368.htm)：核对 NII/EVE、gap、basis、无到期存款和提前还款的风险边界。
- [[02_Economy/07_金融机构与风险管理/09_利率风险.md|课程：利率风险]]：核对课程中的 NII、期限错配和再融资案例；计算细节保留在独立 IRRBB 主题。
