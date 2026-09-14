---
aliases:
  - "Harrod 的离轨反馈可能放大产能过紧或闲置但原模型未闭合完整动态"
  - Harrodian off-path feedback may amplify capacity pressure or slack
  - Harrod instability boundary
  - Harrod 刀锋不稳定的边界
student_os: knowledge-atom
atom_id: DEV-HD-004
atom_set: harrod-domar-growth
atom_type: mechanism-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Harrod 三种增长率]]"
part_of:
  - "[[Harrod—Domar 增长模型.canvas]]"
related:
  - "[[Solow 转型动态]]"
---

# Harrod 的离轨反馈可能放大产能过紧或闲置但原模型未闭合完整动态
<!-- bilingual-en:start -->
*Harrodian off-path feedback may amplify capacity pressure or slack, but the original model does not close the full dynamics*
<!-- bilingual-en:end -->

> [!summary] 原子机制与边界
> Harrod 的离轨直觉是：若实际增长高于保证增长，实际新增产出所配到的资本少于企业认为合适的数量，资本显得过紧，追加投资可能进一步推高需求；若实际增长低于保证增长，实际资本系数高于企业所需系数，闲置资本压低投资，又会进一步削弱需求。因此偏离可能被放大，而非自动回到保证路径。
>
> 但这不是从 $g_w=s/v$ 一条等式就严格推出的稳定性定理。原始分析没有完整规定企业怎样根据偏离调整投资，离轨动态需要额外行为方程才能闭合。
> <!-- bilingual-en:start -->
> The Harrodian mechanism compares the actual capital coefficient with firms' required coefficient: excess demand for capital can induce further investment, while surplus capacity can depress it. Yet $g_w=s/v$ alone does not prove instability; a complete off-equilibrium path requires an explicit adjustment rule that the original model did not fully supply.
> <!-- bilingual-en:end -->

更具体地，Harrod 用实际资本系数 $C=I/\Delta Y$ 与企业认为合适的系数 $C_r$ 作比较。$g_a>g_w$ 对应 $C<C_r$：相对于已经实现的产出增量，资本不足；$g_a<g_w$ 对应 $C>C_r$：资本超出企业所需。这个比较给出投资反应的方向直觉，但没有给出反应速度、时滞、预期形成或价格调整的完整方程。

所谓“刀锋”至少混合了两个问题：

1. **局部离轨：** $g_a$ 偏离 $g_w$ 后，企业的投资反应是否把经济推得更远；
2. **长期不相容：** 即使经济沿 $g_w$ 运行，$g_w$ 也可能不等于充分就业所需的 $g_n$。

第一问需要说明投资怎样响应资本系数、产能利用或预期误差；第二问是两种增长要求没有自动协调。把两者都压成一句“模型不稳定”，会看不出究竟是哪条机制在起作用。

Solow 模型通过可替代要素和资本边际收益递减提供了另一种调整结构，使每有效劳动资本可向稳态回归。但这说明模型机制不同，不等于现实经济必然稳定。

> [!question]- 自检
> 为什么只写出 $g_w=s/v$ 还不能证明偏离保证增长率后会越来越远？
>
> **答案：** 该式只刻画保证路径上的一致条件；要判断离轨方向，必须再给出企业如何根据利用率、销售或预期误差调整投资的行为规则。

## 来源与核验

- Harrod（1939），[An Essay in Dynamic Theory](https://doi.org/10.2307/2225181)：核对实际资本系数与企业所需资本系数的比较，以及偏离实际与保证增长率时的投资直觉。
- Blume 与 Sargent（2015），[Harrod 1939](https://doi.org/10.1111/ecoj.12224)：核对原始稳定性讨论并未闭合完整离轨模型的边界。
- [[Solow 转型动态]]：对比可替代要素、递减回报与明确资本运动方程所给出的稳定调整机制。
