---
aliases:
  - "LCR 与 NSFR 因子必须按资产质量、交易对手、剩余期限和受限状态分类"
  - LCR and NSFR factor classification
  - LCR NSFR 因子分类
student_os: knowledge-atom
atom_id: MB-BAS-018
atom_set: basel-capital-liquidity-regulation
atom_type: regulatory-classification
status: source-checked
mastery_state: unassessed
requires:
  - "[[LCR 与 NSFR]]"
  - "[[银行流动性资源边界]]"
related:
  - "[[资产出售、抵押与无损即时变现]]"
  - "[[批发融资流动性风险]]"
  - "[[流动性压力管理闭环]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# LCR 与 NSFR 因子必须按资产质量、交易对手、剩余期限和受限状态分类
*LCR and NSFR factors require classification by asset quality, counterparty, residual maturity and encumbrance*

> [!summary] 因子不是按名称一键匹配
> LCR 的 HQLA 折扣与现金流流失/流入率，以及 NSFR 的 ASF/RSF 因子，都取决于监管定义中的资产质量、交易对手、担保、剩余期限、行为稳定性和受限状态；同名“存款”“债券”或“资本”可能落入不同档。

LCR 先判断资产是否合格、无负担、可操作变现，再按 Level 1、2A、2B 的折扣与组合上限计入 HQLA；现金流出和流入又按零售/批发、稳定性、操作关系、担保和到期情况分类，并受总流入上限约束。仅有高信用评级不自动满足全部 HQLA 条件。

NSFR 的 ASF 看资金来源与剩余期限，RSF 看资产流动性、期限、抵押和受限状态，并对表外承诺配置所需稳定资金。合格监管资本一般获得 100% ASF，但剩余期限不足一年的 Tier 2 资本工具被排除在该项之外；因此不能只看到“Tier 2”就机械给 50% ASF。

例：两笔期限相同的存款，一笔来自符合稳定条件且受存款保险覆盖的零售客户，另一笔来自非经营性金融机构批发资金，其 LCR 流失率和 NSFR ASF 因子可能不同。

## 边界

- LCR/NSFR 因子是监管压力假设，不是实际流失概率或市场折价的点预测。
- 一项资产可抵押给央行不自动等于 LCR HQLA；是否受限和能否在压力期操作使用仍重要。
- 具体因子与国家裁量须按当日框架和本地实施查表，不能凭资产简称猜测。

> [!question]- 自检
> 为什么“这是一张 Tier 2 债券”不足以决定其 NSFR ASF 因子？
>
> **答案：** 还要看它是否为合格监管资本及剩余期限；不足一年的 Tier 2 不能按长期合格资本的 100% ASF 处理。

## 来源与核验

- [Basel Framework, LCR30](https://www.bis.org/basel_framework/chapter/LCR/30.htm) 与 [LCR40](https://www.bis.org/basel_framework/chapter/LCR/40.htm)：核对 HQLA、折扣、上限及现金流分类。
- [Basel Framework, NSF30](https://www.bis.org/basel_framework/chapter/NSF/30.htm)：核对 ASF、RSF、期限与受限状态；尤其是剩余期限不足一年的 Tier 2 处理。
- [[LCR 与 NSFR]]：复用两项比率的功能和时间窗。
- 口径核验日：2026-08-29。
