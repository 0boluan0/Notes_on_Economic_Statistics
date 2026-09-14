---
aliases:
  - "DID 的平行趋势与处理效应会随结果尺度改变"
  - DID outcome scale
  - Functional-form sensitivity of parallel trends
  - 双重差分结果变换
student_os: knowledge-atom
atom_id: ECON-DID-016
atom_type: identification-boundary
status: source-checked
mastery_state: unassessed
part_of:
  - "[[双重差分法（DID）.canvas]]"
requires:
  - "[[双重差分法]]"
  - "[[平行趋势]]"
leads_to:
  - "[[DID 估计目标与对照组]]"
---

# DID 的平行趋势与处理效应会随结果尺度改变
<!-- bilingual-en:start -->
*Parallel trends and treatment effects in DID depend on the outcome scale*
<!-- bilingual-en:end -->

> [!summary] 识别边界
> 在水平值 $Y$ 上相信未处理变化平行，不等于在 $\log Y$、比例或其他单调变换 $g(Y)$ 上也相信平行趋势。变换结果变量不仅改变系数单位，还重新定义平行趋势假设和待估处理效应。
>
> <!-- bilingual-en:start -->
> Parallel untreated changes in outcome levels $Y$ do not imply parallel changes in $\log Y$, proportions, or another monotone transformation $g(Y)$. Transforming the outcome changes not only coefficient units but also the parallel-trends assumption and the treatment effect being estimated.
> <!-- bilingual-en:end -->

水平尺度上的条件是

$$
E[Y_1(0)-Y_0(0)\mid G=1]
=E[Y_1(0)-Y_0(0)\mid G=0].
$$

把结果改成 $g(Y)$ 后，需要另一项条件：

$$
E[g(Y_1(0))-g(Y_0(0))\mid G=1]
=E[g(Y_1(0))-g(Y_0(0))\mid G=0].
$$

第一条一般不推出第二条。只有在更强的分布条件下，平行趋势才会对所有严格单调变换保持不变；普通 DID 设计不能默认获得这种不变性。
<!-- bilingual-en:start -->
After replacing $Y$ with $g(Y)$, DID requires a new parallel-trends restriction on changes in $g(Y(0))$. The level condition generally does not imply the transformed condition. Parallel trends is invariant to all strictly monotone transformations only under much stronger restrictions on the untreated outcome distributions; an ordinary DID design does not receive that invariance automatically.
<!-- bilingual-en:end -->

## 一个方向相反的简单例子

假设无处理时，处理组从 100 变到 110，对照组从 50 变到 60。两组都增加 10，所以水平值满足平行趋势；但对数变化分别为 $\log(110/100)$ 与 $\log(60/50)$，并不相同。反过来，若处理组从 100 变到 110、对照组从 50 变到 55，两组增长率同为 10%，对数变化平行，水平变化却分别是 10 与 5。
<!-- bilingual-en:start -->
Suppose untreated outcomes rise from 100 to 110 in the treated group and from 50 to 60 in the comparison group. Both increase by 10, so trends are parallel in levels, but $\log(110/100)$ differs from $\log(60/50)$. Conversely, changes from 100 to 110 and from 50 to 55 imply the same 10 percent growth and therefore parallel log changes, while the level changes are 10 and 5.
<!-- bilingual-en:end -->

尺度也决定 estimand。水平 DID 回答平均结果单位改变多少；对数 DID 回答对数结果改变多少，在附加条件和适当幅度下才可近似读成百分比。对含零或负值的结果，对数还会改变可进入样本的单位。研究者应由机制和政策问题预先选择尺度，并说明为何未处理变化在这个尺度上可比，而不是挑选给出最好看结果的变换。
<!-- bilingual-en:start -->
The scale also defines the estimand. A level DID concerns changes in outcome units. A log DID concerns changes in log outcomes and can be read approximately as a percentage only under additional conditions and suitable magnitudes. Logging an outcome with zeros or negative values also changes which observations can enter. The scale should therefore be chosen in advance from the mechanism and policy question, with a justification for parallel untreated changes on that scale, rather than selected for an attractive estimate.
<!-- bilingual-en:end -->

把多个尺度作为敏感性分析时，不能用“多数规格显著”投票。每个变换对应不同的反事实限制和处理效应；结果不同时，应判断哪个尺度最符合机制，或明确结论依赖函数形式。
<!-- bilingual-en:start -->
Using several scales as a sensitivity analysis is not a vote over which specifications are significant. Each transformation carries a different counterfactual restriction and treatment effect. When conclusions differ, explain which scale is supported by the mechanism or state explicitly that the conclusion is functional-form dependent.
<!-- bilingual-en:end -->

> [!question]- 回忆与应用
> 两组处理前在水平值上每期都增加 5，能否因此断言对数结果满足平行趋势？
>
> **答案：** 不能。相同绝对变化在不同初始水平上对应不同增长率；需要单独论证对数未处理结果的变化为何可比。

## 来源与核验

- Roth & Sant’Anna (2023), [*When Is Parallel Trends Sensitive to Functional Form?*](https://doi.org/10.3982/ECTA19402)：核验平行趋势通常依赖结果变换，以及对所有严格单调变换保持不变所需的更强分布条件。
- Kahn-Lang & Lang (2020), [*The Promise and Pitfalls of Differences-in-Differences*](https://doi.org/10.1080/07350015.2018.1546591)：交叉核验 DID 需要为所选函数形式给出实质理由，处理前趋势证据也不能替代这一选择。
