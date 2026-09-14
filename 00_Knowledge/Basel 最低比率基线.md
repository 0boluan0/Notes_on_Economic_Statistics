---
aliases:
  - "Basel 的 4.5-6-8、杠杆 3 与 LCR-NSFR 100 是基线而不是完整最终约束"
  - Basel baseline ratios are not final requirements
  - 巴塞尔最低比率基线
student_os: knowledge-atom
atom_id: MB-BAS-005
atom_set: basel-capital-liquidity-regulation
atom_type: regulatory-baseline
status: source-checked
mastery_state: unassessed
requires:
  - "[[CET1、AT1 与 Tier 2]]"
  - "[[风险加权资本率]]"
  - "[[Basel 杠杆率]]"
  - "[[LCR 与 NSFR]]"
related:
  - "[[CCB 与 CCyB]]"
  - "[[G-SIB 与 D-SIB]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# Basel 的 4.5-6-8、杠杆 3 与 LCR-NSFR 100 是基线而不是完整最终约束
*Basel's 4.5-6-8, leverage 3 and LCR-NSFR 100 are baselines, not complete final constraints*

> [!summary] 数字必须连同分子、分母与层次
> 截至 2026-08-29 的 Basel 国际基线包括 CET1/RWA 4.5%、Tier 1/RWA 6%、Total Capital/RWA 8%、Tier 1/Exposure Measure 杠杆率 3%，以及 LCR、NSFR 常态最低值 100%。这些数字均不是某家银行最终需要满足的全部要求。

资本缓冲、系统重要性附加、逆周期要求、监督加成和本地规则会把实际约束推高；处于资本留存缓冲以内通常触发分派限制，而不是把 4.5%、6%、8% 的最低线重新定义为另一个数字。3% 也只是普通银行的国际杠杆率基线，G-SIB 还适用以 Tier 1 资本满足的杠杆率缓冲。杠杆率与风险加权比率使用不同分母，LCR 与 NSFR 又衡量现金流和融资结构，不能互相替换。

例：银行 CET1/RWA 为 7%，高于 4.5%，却可能低于 4.5% 加 2.5% 资本留存缓冲及其他适用缓冲之和；此时不能简单写成“资本充足且可自由分红”。

## 边界

- 4.5-6-8 是三个嵌套资本层级的最低比率，不是 18.5%，也不应相加。
- LCR/NSFR 的 100% 是 Basel 常态最低基线；压力时缓冲可被使用，监督反应取决于情境。
- 本地实施可能设更高要求或不同过渡；具体银行合规须查法域和日期。

> [!question]- 自检
> CET1 比率 6% 是否必然满足 Basel 的全部资本约束？
>
> **答案：** 不必然。还要看 Total/Tier 1、各种缓冲、杠杆率、监督加成、本地实施及其他约束。

## 来源与核验

- [Basel Framework, RBC20](https://www.bis.org/basel_framework/chapter/RBC/20.htm)：核对风险加权最低资本率与总 RWA。
- [Basel Framework, LEV20](https://www.bis.org/basel_framework/chapter/LEV/20.htm) 与 [LEV30](https://www.bis.org/basel_framework/chapter/LEV/30.htm)：核对 3% 杠杆率国际基线及 G-SIB 杠杆率缓冲。
- [Basel Framework, LCR20](https://www.bis.org/basel_framework/chapter/LCR/20.htm) 与 [NSF20](https://www.bis.org/basel_framework/chapter/NSF/20.htm)：核对 100% 常态最低值及两项比率目的。
- 数值核验日：2026-08-29；不替代本地有效规则。
