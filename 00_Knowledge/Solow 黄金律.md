---
aliases:
  - "Solow 黄金律最大化稳态消费水平而不是一条完整的跨期福利最优规则"
  - "The Solow Golden Rule maximizes steady-state consumption rather than full intertemporal welfare"
  - "Solow Golden Rule"
student_os: knowledge-atom
atom_id: MACRO-SOLOW-005
atom_set: solow-growth
atom_type: welfare-objective-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[Solow 稳态条件]]"
part_of:
  - "[[Solow 增长模型、稳态与收敛.canvas]]"
related:
  - "[[储蓄率的水平效应]]"
---

# Solow 黄金律最大化稳态消费水平而不是一条完整的跨期福利最优规则
<!-- bilingual-en:start -->
*The Solow Golden Rule maximizes steady-state consumption rather than full intertemporal welfare*
<!-- bilingual-en:end -->

> [!summary] 原子目标
> 在每有效劳动稳态中，消费可写为
> $$c^*=f(k^*)-(n+g+\delta)k^*.$$
> 令稳态消费对资本的一阶条件为零，黄金律资本满足
> $$f'(k^*_{\mathrm{gold}})=n+g+\delta.$$
> 它选择的是稳态消费最高的资本水平；它没有权衡到达该稳态之前各期消费，也没有给出含贴现效用的完整社会福利最优路径。
> <!-- bilingual-en:start -->
> The Golden Rule sets the marginal product of capital equal to effective-labor growth plus depreciation and maximizes consumption in the steady state. Because the basic Solow model has exogenous saving and no intertemporal utility objective, this is not a complete welfare ranking of transition paths.
> <!-- bilingual-en:end -->

若不含人口与技术增长，条件退化为 $f'(k^*_{\mathrm{gold}})=\delta$。低于黄金律资本时，提高稳态资本可提高稳态消费；高于黄金律时，维持过多资本消耗的投资超过新增产出，降低稳态资本反而可提高稳态消费。

但从当前资本转到另一个稳态会改变过渡期消费。例如为了提高未来稳态资本，当前必须先多储蓄、少消费。没有效用函数、贴现率和代际权重，基本 Solow 模型不能把这条过渡成本与未来收益合成唯一福利判断。

> [!question]- 自检
> 为什么“黄金律储蓄率”不能直接叫作社会最优储蓄率？
>
> **答案：** 它最大化稳态消费水平，却没有评价过渡期消费、贴现和代际分配；完整的跨期福利最优需要另一个明确的效用目标。

## 来源与核验

- MIT 14.452，[The Solow Growth Model, Lectures 2–3](https://ocw.mit.edu/courses/14-452-economic-growth-fall-2016/2b68057aa4e74410d00ae89a0c49752f_MIT14_452F16_Lec2and3.pdf)，pp. 31–35：核对黄金律稳态消费、一阶条件及没有效用函数时福利表述必须谨慎的边界。
- 同一讲义 pp. 75–81：核对人口与劳动增进技术存在时 break-even 项为 $n+g+\delta$，据此重做一般化黄金律条件。
