---
aliases:
  - "杠杆率以一级资本约束总敞口并作为风险不敏感的后备"
  - Basel leverage ratio backstop
  - 银行监管杠杆率
student_os: knowledge-atom
atom_id: MB-ALM-011
atom_set: commercial-bank-alm
atom_type: regulatory-measure
status: source-checked
mastery_state: unassessed
requires:
  - "[[会计权益与监管资本]]"
  - "[[风险加权资本率]]"
related:
  - "[[ROA、ROE 与杠杆]]"
  - "[[Basel 最低比率基线]]"
part_of:
  - "[[商业银行资产负债管理.canvas]]"
---

# 杠杆率以一级资本约束总敞口并作为风险不敏感的后备
*The leverage ratio constrains total exposure with Tier 1 capital as a risk-insensitive backstop*

> [!summary] 后备指标
> Basel 杠杆率用 Tier 1 资本除以监管 exposure measure。分母包括规定的表内资产、衍生品、证券融资交易和表外项目，不采用 RWA 风险权重。它用简单后备限制风险权重过低或模型误差下的总杠杆积累。

“不采用风险权重”既是优点也是盲点。它较难通过调低风险权重来改善，却把低风险与高风险的同额敞口较粗地对待，可能对低风险、低收益业务形成较强约束。因此它与风险加权资本率互补，不是更“真实”的唯一资本率。

例：Tier 1 资本 12、杠杆暴露 200，杠杆率为 6%；同一银行若 RWA 为 100，Tier 1 风险加权资本率为 12%。两个比率不同来自分母不同，不能互相替代或相加。

## 边界

- 监管 exposure measure 不等于会计总资产；衍生品、证券融资和表外承诺有指定处理。
- 最低比率与附加要求依赖框架版本、银行类别和实施法域；当前 Basel 基线不能冒充每个法域的最终法律要求。
- 会计资产/权益倍数与 Basel 杠杆率方向相关但公式不同。

> [!question]- 自检
> 为什么一项资产风险权重下降可能提高 RWA 资本率，却不直接提高杠杆率？
>
> **答案：** 杠杆率分母不用该风险权重；只要监管总敞口不变，单纯降权重不会改变杠杆暴露。

## 来源与核验

- [Basel Framework, LEV20](https://www.bis.org/basel_framework/chapter/LEV/20.htm)：核对杠杆率目的、Tier 1 分子和非风险加权后备性质。
- [Basel Framework, LEV30](https://www.bis.org/basel_framework/chapter/LEV/30.htm)：核对表内、衍生品、证券融资和表外 exposure measure；实际实施须查本地规则。
