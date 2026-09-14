---
aliases:
  - "总 RWA 汇总多个风险模块且 12.5 只把资本要求转换为等价 RWA"
  - Total RWA aggregation and the 12.5 conversion
  - RWA 聚合与 12.5 倍转换
student_os: knowledge-atom
atom_id: MB-BAS-010
atom_set: basel-capital-liquidity-regulation
atom_type: regulatory-accounting
status: source-checked
mastery_state: unassessed
requires:
  - "[[风险加权资本率]]"
  - "[[Basel 最低比率基线]]"
related:
  - "[[当前 Basel 操作风险标准法]]"
  - "[[FRTB 市场风险]]"
  - "[[三类 CVA 口径]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# 总 RWA 汇总多个风险模块且 12.5 只把资本要求转换为等价 RWA
*Total RWA aggregates several risk modules, and 12.5 only converts a capital charge into RWA-equivalent units*

> [!summary] 统一分母的换算
> 总 RWA 不只等于资产余额乘信用权重。当前 Basel 的顶层预 floor 汇总是信用风险 RWA、市场风险 RWA 与操作风险 RWA；对手方信用风险和银行簿证券化等进入信用风险项，CVA 风险按 RBC20 的聚合口径进入市场风险项，不能再把这些子项与三大顶层项并列相加。某模块若先计算资本要求 $K$，乘 $12.5=1/0.08$ 只是把它换成与 8% 总资本基线一致的 RWA 单位。

例：某模块资本要求为 8，则等价 RWA 为 $12.5\times8=100$；再对 100 应用 8% 总资本率恰好回到 8。这个换算没有放大或缩小经济风险，也不表示该资产“风险是账面值的 12.5 倍”。

聚合前必须防止遗漏和不当重复。交易对手违约风险与 CVA 风险针对不同损失机制，可能同时进入不同上层项目；同一证券化或市场头寸则必须按框架指定边界处理，不能随意在信用和市场模块间重复计算或择低。应用 output floor 时，最终总 RWA 取预 floor 总 RWA 与按标准法基准及当期 floor 比例得到的 floored RWA 两者较高者，而不是在预 floor 结果上再机械加一遍 floor。

## 边界

- 12.5 来自 8% 的倒数，是单位转换，不是风险权重的普遍乘数。
- 每个模块的 $K$ 如何计算取决于标准法、获批模型和适用版本。
- 最终资本要求还可能受 output floor、缓冲、杠杆率和监督加成约束。

> [!question]- 自检
> 操作风险资本要求为 20 时，为什么写成 250 RWA 不代表损失预测为 250？
>
> **答案：** 250 是 $20\times12.5$ 的监管等价分母；它让 8% 总资本率回算出 20，而不是损失概率预测。

## 来源与核验

- [Basel Framework, RBC20](https://www.bis.org/basel_framework/chapter/RBC/20.htm)：核对总 RWA 的模块构成与 12.5 转换。
- [[风险加权资本率]]：复用资本率分子、分母和风险权重边界。
- 口径核验日：2026-08-29；模块范围和本地实施须按报告日复核。
