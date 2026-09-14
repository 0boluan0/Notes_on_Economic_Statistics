---
aliases:
  - "重定价时点相同的头寸仍会因参考利率不同产生银行账簿基差风险"
student_os: knowledge-atom
atom_id: MB-ALM-020
atom_set: commercial-bank-alm
atom_type: mechanism
status: source-checked
mastery_state: unassessed
requires:
  - "[[IRRBB 的 NII 与 EVE]]"
  - "[[重定价缺口]]"
related:
  - "[[曲线形变]]"
  - "[[无到期存款行为]]"
leads_to:
  - "[[压力传导映射]]"
part_of:
  - "[[银行账簿利率风险.canvas|银行账簿利率风险]]"
---

# 重定价时点相同的头寸仍会因参考利率不同产生银行账簿基差风险

> [!summary] 时点匹配不等于利率变化匹配
> 两个头寸即使金额、期限和重定价日相同，只要分别挂钩不同参考利率，或对同一市场利率的传导幅度与滞后不同，收入与成本就可能不同比例变化。银行账簿基差风险因此来自参考利率之间相关性不完美，以及产品利率的不同 pass-through（传导）规则；静态重定价缺口为零不能消除它。

## 机制：分别冲击资产与负债利率

对同一规划期内同时重定价的资产和负债，设重定价后的计息权重均为 $w$，则净利息收入的一阶变化可写为

$$
\Delta NII
\approx
A\beta_A\Delta r_A w
-
L\beta_L\Delta r_L w,
$$

其中 $\Delta r_A$、$\Delta r_L$ 是各自参考利率的变化，$\beta_A$、$\beta_L$ 是合同或行为传导幅度。即使 $A=L$ 且重定价时点完全相同，只要

$$
\beta_A\Delta r_A\ne\beta_L\Delta r_L,
$$

$\Delta NII$ 就不为零。风险可以来自三类错位：

- **不同指数：** 资产挂钩隔夜利率，负债成本随另一条批发融资或存款定价基准变化；两条利率不会始终同幅变动。
- **不同传导：** 贷款可能按合同接近完全传导，管理利率存款则由 deposit beta（存款利率相对市场利率的变动比例）、floor 和定价策略决定。
- **相关性不稳定：** 平静期高度相关不等于压力期仍相关；基差情景应允许参考利率分化，而不能只施加所有曲线同幅平移。

EVE 视角也要让挂钩不同参考利率的现金流分别承受相应情景，再按适用估值框架折现；共同折现曲线本身并不必然漏掉基差。真正会隐藏基差的是把不同指数合并成同一参考利率，或只对它们施加共同同幅冲击。

## 可复算例：零重定价缺口仍有 NII 变化

设资产与负债均为 100，今天同时重定价，并在余下一年保持该利率，因此 $A=L=100$、$w=1$，静态重定价缺口为零。

- 资产参考利率上升 80bp，合同完全传导：$\beta_A\Delta r_A=0.008$。
- 负债管理利率对应的市场基准上升 100bp，但存款 beta 为 35%：$\beta_L\Delta r_L=0.35\times0.01=0.0035$。

于是

$$
\Delta NII
\approx
100\times0.008-100\times0.0035
=0.45.
$$

若压力情景改为资产指数只上升 30bp、负债实际付息率上升 70bp，则

$$
\Delta NII
\approx
100\times0.003-100\times0.007
=-0.40.
$$

金额与重定价时点没有改变，结果却因基准和传导关系改变而反向。

## 边界

- 本卡的“基差”是 **IRRBB 中相似期限头寸挂钩不同利率指数而产生的相对利率变化风险**，不是 CDS 与现金债之间的信用基差交易。
- IRRBB 基差不等于同一指数内部不同期限的非平行变化；后者见 [[曲线形变]]。
- 不把市场信用利差或个体信用利差变化并入 IRRBB。Basel 将不能由 IRRBB 或预期信用损失／跳跃至违约解释的信用风险工具价差风险划入银行账簿信用利差风险（CSRBB）。
- deposit beta 和 lag 是行为模型参数，不是稳定常数；它们应在 [[无到期存款行为]] 中单独受压。
- 本卡给出一阶 NII 示例，不替代逐现金流 EVE 重估、动态资产负债表、对冲成本或管理行动分析。

> [!question]- 最小自检
> 资产 200 与负债 200 同日重定价。资产指数上升 60bp并完全传导；负债基准上升 90bp，传导率为 50%。一年权重下，重定价缺口是多少，$\Delta NII$ 是多少？为什么风险仍存在？
>
> **答案：** 金额口径的重定价缺口为零；$\Delta NII\approx200\times0.006-200\times(0.5\times0.009)=0.30$。风险仍存在，因为两端参考利率变化与传导后的实际利率变化不相同。

## 定位性一手来源与复核状态

- Basel Committee on Banking Supervision, [Basel Framework SRP31.2（2026-01-01 生效，当前版本）](https://www.bis.org/basel_framework/chapter/SRP/31.htm?inforce=20260101&published=20240716)：将基差风险定位为相似期限金融工具因采用不同利率指数而产生的相对利率变化影响；同段把 gap、basis 与 option risk 分开。
- Basel Committee on Banking Supervision, [Basel Framework SRP98.15、98.31、98.49（2026-01-01 生效，当前版本）](https://www.bis.org/basel_framework/chapter/SRP/98.htm?inforce=20260101&published=20240716)：定位不同指数调整相关性不完美、共同平行冲击无法捕捉基差，以及用不同基准利率分化情景测量基差风险。
- 同一 Basel 当前框架的 SRP98.10、98.13：定位 IRRBB 与 CSRBB 的口径边界，防止把信用利差风险机械并入本卡。

> [!info] 核验状态
> 本轮已独立核对定义、算例、来源定位与适用边界，状态为 `source-checked`；掌握度仍为 `unassessed`。用于真实银行前仍须按实际指数、beta、lag、floor、相关性破裂情景及 NII/EVE 口径分别重算。
