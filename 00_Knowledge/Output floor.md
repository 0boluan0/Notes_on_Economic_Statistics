---
aliases:
  - "Output floor 限制模型 RWA 低于标准法基准而不是给资本率封顶"
  - Basel output floor
  - RWA 产出下限
student_os: knowledge-atom
atom_id: MB-BAS-011
atom_set: basel-capital-liquidity-regulation
atom_type: model-constraint
status: source-checked
mastery_state: unassessed
requires:
  - "[[RWA 聚合与 12.5 换算]]"
related:
  - "[[风险加权资本率]]"
  - "[[资本监管的收益与边界]]"
part_of:
  - "[[巴塞尔银行资本与流动性监管.canvas]]"
---

# Output floor 限制模型 RWA 低于标准法基准而不是给资本率封顶
*The output floor limits modelled RWA below a standardised benchmark; it does not cap capital ratios*

> [!summary] 给分母设下限
> Output floor 要求采用内部模型的银行，其用于资本计算的总 RWA 不得低于按指定标准法计算基准的一定比例。Basel 国际过渡表在 2026、2027、2028 年分别为 65%、70%、72.5%，此后稳态为 72.5%；但只有完成本地转置后才是该法域的有效参数，本地实施日期和过渡表必须另查。

若模型总 RWA 为 60、标准法基准为 100，72.5% floor 给出下限 72.5，最终分母取较高者 72.5。若模型 RWA 为 90，则 floor 不绑定，仍用 90。它约束的是模型产出的过低分母，不是把银行资本率“封顶在 72.5%”，也不是要求每项资产的模型权重至少等于标准权重的 72.5%。

Output floor 降低跨银行模型差异和模型套利空间，但会削弱某些模型风险敏感性，且标准法基准本身也可能粗糙。因此它是后备约束，不是对真实风险的完美测量。

## 边界

- Floor 通常在聚合层面应用；不能机械逐资产相乘。
- 65% 是 Basel 国际过渡表中的 2026 年比例，72.5% 是 2028 年起的稳态基线；两者都不能越过本地实施规则直接套给具体银行。
- Floor 与 3% 杠杆率不同：前者仍依赖标准法 RWA，后者使用非风险加权的 exposure measure。

> [!question]- 自检
> 标准法基准 200、模型 RWA 180 时，72.5% floor 是否绑定？
>
> **答案：** 不绑定。下限为 145，模型 RWA 180 更高。

## 来源与核验

- [Basel Framework, RBC20](https://www.bis.org/basel_framework/chapter/RBC/20.htm)：核对 output floor 的聚合公式、标准法基准和稳态 72.5% 参数。
- [RCAP implementation monitoring](https://www.bis.org/bcbs/implementation/rcap_reports.htm)：核对各法域实施与过渡并不同步。
- 口径核验日：2026-08-29。
