---
aliases:
  - "FRTB 内部模型资格按交易台接受回测与 P&L attribution 检验"
  - FRTB desk-level IMA backtesting and P&L attribution
  - 交易台级 IMA 资格
student_os: knowledge-atom
atom_id: MB-BAS-015
atom_set: basel-capital-liquidity-regulation
atom_type: model-governance
status: source-checked
mastery_state: unassessed
requires:
  - "[[FRTB 市场风险]]"
  - "[[FRTB回测损益]]"
related:
  - "[[风险承载力、偏好与限额]]"
  - "[[风险模型验证边界]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# FRTB 内部模型资格按交易台接受回测与 P&L attribution 检验
*FRTB internal-model eligibility is granted at trading-desk level through backtesting and P&L attribution tests*

> [!summary] 模型批准不是全行永久通行证
> FRTB 把内部模型法（IMA）资格落实到符合定义的交易台。交易台须满足组织、模型、数据与治理要求，并通过回测和损益归因（PLA）；不合格交易台回到标准法，资格也可随测试结果变化。

回测把一日实际损益（APL）和假设损益（HPL）分别与模型的一日 VaR 比较，数例外次数；全行回测分区使用两套损益所产生例外数中的较大者。PLA 则比较风险理论损益（RTPL）与 HPL，检验资本模型使用的风险因子是否能解释前台定价产生的主要损益。两项检验回答不同问题：例外过多说明尾部覆盖不足，PLA 失败更像风险表示与定价表示之间的缺口。

当前 FRTB 还把全行层面的回测结果用于资本乘数：以最近 250 个交易日中 99% 单尾一日 VaR 的例外数计，0–4 次对应 1.50，5 次 1.70，6 次 1.76，7 次 1.83，8 次 1.88，9 次 1.92，10 次及以上 2.00；另有最多 0.5 的定性附加。这个当前映射不能与旧框架的 3.00–4.00 经典乘数表互换。交易台资格又是另一层测试：97.5% 与 99% 的一日 VaR 回测例外过多会使交易台失去 IMA 资格；PLA 结果分绿、琥珀、红区，琥珀区仍可留在 IMA 但有附加资本，红区则不合格。

## 边界

- 回测通过不证明未来损失分布正确；有限样本、数据处理和结构变化仍可能失效。
- PLA 不是一般会计损益核对，而是特定 HPL/RTPL 的监管比较。
- IMA 不保证资本一定低于标准法；模型资本、NMRF、floor 与不合格交易台都可能使结果更高。

> [!question]- 自检
> 一个交易台回测例外很少，但 PLA 失败，能否继续只凭“尾部覆盖很好”使用 IMA？
>
> **答案：** 不能。PLA 失败表明模型风险表示不能充分解释前台假设损益；FRTB 资格同时受两类检验约束。

## 来源与核验

- [Basel Framework, MAR30](https://www.bis.org/basel_framework/chapter/MAR/30.htm)、[MAR32](https://www.bis.org/basel_framework/chapter/MAR/32.htm) 与 [MAR33](https://www.bis.org/basel_framework/chapter/MAR/33.htm)：核对交易台级回测/PLA 资格、全行回测乘数及 IMA 资本计算。
- [Basel Committee, Supervisory framework for the use of backtesting](https://www.bis.org/publ/bcbs22.htm)：只用于辨认旧框架的经典 3.00–4.00 乘数表，不能替代当前 FRTB 参数。
- 口径核验日：2026-08-29。
