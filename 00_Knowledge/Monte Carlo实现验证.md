---
aliases:
  - "Monte Carlo 实现应先用可解析特例核对目标，再分别测试随机生成、状态更新与汇总，并一次只改变一个误差旋钮"
  - "A Monte Carlo implementation should first match an analytic case, then test random generation, state updates, and aggregation separately while varying one error control at a time"
  - "模拟实现验证"
  - "Monte Carlo implementation validation"
student_os: knowledge-atom
atom_id: PROB-MC-006
atom_set: monte-carlo-methods
atom_type: procedure
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[Monte Carlo估计]]"
  - "[[Monte Carlo误差分层]]"
  - "[[伪随机复现]]"
  - "[[并行伪随机子流]]"
related:
  - "[[路径数不修模型]]"
  - "[[风险模型验证边界]]"
---

# Monte Carlo 实现应先用可解析特例核对目标，再分别测试随机生成、状态更新与汇总，并一次只改变一个误差旋钮
<!-- bilingual-en:start -->
*A Monte Carlo implementation should first match an analytic case, then test random generation, state updates, and aggregation separately while varying one error control at a time*
<!-- bilingual-en:end -->

> [!summary] 验证顺序必须让失败可以定位
> 先用有解析答案的退化或简化情形确认程序在算正确对象；再拆开生成、状态转移和汇总；最后分别改变经文档化派生的随机子流、路径数和数值步长，检查每种误差是否按自己的理论方向变化。
> <!-- bilingual-en:start -->
> Start with a simplified case whose answer is known, then isolate generation, state transition, and aggregation. Finally vary documented derived random substreams, path count, and numerical step separately so that each error source has a diagnostic signature.
> <!-- bilingual-en:end -->

## 最小可执行顺序

1. **锁定目标量。** 写明抽样分布、支持集、单位、期限、条件信息和 $g(X)$；选择至少一个能直接计算均值、方差或概率的特例。
2. **拆分组件。** 单独检查随机变量生成的边际与支持、状态更新的不变量、以及汇总函数对人工输入的结果。
3. **核对抽样规律。** 固定算法后，用生成器明确支持的 spawning、jump-ahead 或独立 key 构造可追踪重复；只是换一个 seed 只能检查 seed 敏感性，不自动证明流独立。增加 $N$ 时，检查设计匹配的标准误是否按预期收缩。
4. **核对数值规律。** 固定随机输入后缩小时间步长或求解容差；若结果系统性移动，应把它记作数值近似问题，而不是抽样噪声。
5. **保留可重放证据。** 记录版本、随机流分配、参数、输入摘要和输出统计，使异常能够在同一调用协议下重现。

<!-- bilingual-en:start -->
The minimum sequence is to state the estimand, test a tractable oracle case, isolate generation/state/aggregation, vary sampling controls separately from numerical controls, and retain enough environment and random-stream information to replay a failure.
<!-- bilingual-en:end -->

一次运行恰好接近解析答案不构成验证；它可能只是抽样幸运。反过来，抽样标准误走势正确也只验证了一种数值现象，不证明模型适合现实用途。
<!-- bilingual-en:start -->
One lucky run near the analytic answer is not validation. Conversely, a correct standard-error trend verifies one numerical signature, not the suitability of the model for its real use.
<!-- bilingual-en:end -->

> [!question]- 可核验自检
> 一个路径模型同时更换分布、时间步长和估值函数后更接近基准。为什么仍不能定位修复原因？
>
> **答案：** 三个误差旋钮同时改变，无法知道改进来自模型、离散化还是汇总；应从可重放基线开始逐项改变。
> <!-- bilingual-en:start -->
> Because three error controls changed at once. Return to a reproducible baseline and vary the model, discretisation, and aggregation one at a time.
> <!-- bilingual-en:end -->

## 来源与核验

- MIT 6.100L, [[03_Computer_Science/03_MIT 6.100L/MIT 6.100L-slides/mit6_100l_lec26.pdf|Lecture 26 slides, pp. 21–28]]：支持“定义一次实验—重复—记录—汇总”的可检查计算框架。
- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2](https://artowen.su.domains/mc/Ch-intro.pdf)：支持用重复样本、样本方差和标准误诊断抽样精度。
- Paul Glasserman (2003), [*Monte Carlo Methods in Financial Engineering*, Chapters 1 and 3](https://doi.org/10.1007/978-0-387-21617-1)：支持把随机抽样、路径生成与数值离散分别核对。
- NASA, [*NASA-STD-7009B: Standard for Models and Simulations*, §§4.2.2–4.2.4](https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf)：核对 benchmark 输入输出用例、代码实现验证与解验证的分离，以及数值近似、离散化与未验证部分应被单独记录。
