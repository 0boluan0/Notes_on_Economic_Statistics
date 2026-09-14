---
aliases:
  - "ROA、ROE 与杠杆必须联合阅读且高 ROE 不自动代表经营更好"
  - Bank ROA ROE and leverage
  - 银行盈利能力与杠杆
student_os: knowledge-atom
atom_id: MB-ALM-004
atom_set: commercial-bank-alm
atom_type: diagnostic
status: source-checked
mastery_state: unassessed
requires:
  - "[[银行利润归因]]"
  - "[[银行资产负债恒等式]]"
related:
  - "[[Basel 杠杆率]]"
part_of:
  - "[[商业银行资产负债管理.canvas]]"
---

# ROA、ROE 与杠杆必须联合阅读且高 ROE 不自动代表经营更好
*ROA, ROE, and leverage must be read together; high ROE does not automatically mean better banking*

> [!summary] 指标诊断
> 常见近似为
> $$ROA=\frac{\text{净利润}}{\text{平均总资产}},\qquad ROE=\frac{\text{净利润}}{\text{平均权益}}.$$
> 在忽略口径差异时，$ROE\approx ROA\times(\text{平均资产}/\text{平均权益})$。所以 ROE 上升可能来自更好的资产收益，也可能只是权益变薄、杠杆变高。

例：两家银行资产均为 100。甲净利润 1、权益 10，ROA 为 1%、ROE 为 10%；乙净利润同为 1、权益仅 5，ROA 仍为 1%、ROE 升到 20%。乙没有创造更多利润，只是由更薄的权益承受同样资产风险。

比较时还要统一平均或期末分母、年化方式、合并范围、一次性损益与风险结构。高 NIM 可能补偿更高信用风险或更贵融资；低 ROA 也可能来自临时增提准备，而非核心业务永久恶化。指标是追问原因的入口，不是单独结论。

## 边界

- 会计杠杆与 Basel 杠杆率不是同一公式；后者用 Tier 1 资本和监管 exposure measure。
- ROE 不直接衡量流动性、尾部损失、资本充足或增长可持续性。
- 横向比较应核对业务模式和会计分类，不能把零售银行、交易型银行和托管机构机械排名。

> [!question]- 自检
> 银行 ROE 从 10% 升到 20%，为什么不能立刻说经营效率翻倍？
>
> **答案：** 可能只是权益分母减半、杠杆上升；还要分解 ROA、利润来源、风险和资本变化。

## 来源与核验

- [FDIC QBP Graph Book](https://qbpgraphbook.fdic.gov/)：核对美国银行业官方报告同时观察 ROA、ROE、净息差、信用成本和资本指标。
- [[02_Economy/03_货币银行学/3_金融机构/09_银行业与金融机构的管理.md#资本充足性管理|课程：资本与 ROE]]：核对课程中的杠杆—收益直觉并修正为联合诊断。
