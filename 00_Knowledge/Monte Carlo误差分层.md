---
aliases:
  - "Monte Carlo 结果的误差必须把抽样、数值近似、实现与模型误差分开；增加样本量主要收缩抽样项"
  - "Monte Carlo error must separate sampling, numerical-approximation, implementation, and model error; increasing sample size mainly contracts the sampling component"
  - "模拟误差分层"
  - "Monte Carlo error taxonomy"
student_os: knowledge-atom
atom_id: PROB-MC-005
atom_set: monte-carlo-methods
atom_type: concept-distinction
status: source-checked
mastery_state: unassessed
part_of:
  - "[[Monte Carlo.canvas|Monte Carlo]]"
requires:
  - "[[Monte Carlo估计]]"
related:
  - "[[Monte Carlo均值标准误]]"
  - "[[路径数不修模型]]"
  - "[[Monte Carlo实现验证]]"
  - "[[风险模型验证边界]]"
---

# Monte Carlo 结果的误差必须把抽样、数值近似、实现与模型误差分开；增加样本量主要收缩抽样项
<!-- bilingual-en:start -->
*Monte Carlo error must separate sampling, numerical-approximation, implementation, and model error; increasing sample size mainly contracts the sampling component*
<!-- bilingual-en:end -->

> [!summary] “模拟误差”不是一个可以只靠路径数解释的数字
> Monte Carlo 输出与现实目标之间的差异，至少可能来自有限样本、数值近似、程序实现和概率模型。四者需要不同证据，不能把总偏差都包装成标准误。
> <!-- bilingual-en:start -->
> A Monte Carlo result can differ from its real target because of finite sampling, numerical approximation, implementation, or the probability model. These require different evidence and cannot all be reported as one standard error.
> <!-- bilingual-en:end -->

| 误差层 | 它回答什么 | 典型检查 |
|---|---|---|
| 抽样误差 | 在固定模型、算法和数值设置下，有限随机样本使估计量波动多少 | 用独立性有构造依据的随机子流重复运行、报告设计匹配的标准误 |
| 数值近似误差 | 时间离散、求解器、截断或代理估值把精确计算近似成什么 | 缩小步长、提高容差、与更精确基准比较 |
| 实现误差 | 代码是否真的计算了所声明的分布、状态更新与目标量 | 可解析特例、单元测试、支持集和不变量检查 |
| 模型误差 | 指定分布、参数、依赖或状态变量是否适合实际问题 | 外样本检验、回测、敏感性与替代模型 |

<!-- bilingual-en:start -->
Sampling error is repeated-run variation conditional on a fixed model and implementation. Numerical-approximation error comes from discretisation, solvers, truncation, or surrogate valuation. Implementation error means the code computes something other than the declared algorithm. Model error concerns whether the specified distribution, parameters, dependence, and state variables fit the real problem.
<!-- bilingual-en:end -->

这是**诊断分层**，不主张四项在一般问题中彼此独立或能精确相加。一个错误的离散方案可能改变抽样分布，参数重估也可能把模型不确定性带入数值输出。分层的目的，是让每项误差都有自己的旋钮和核验证据。
<!-- bilingual-en:start -->
This is a diagnostic taxonomy, not a claim that the four components are independent or exactly additive. Their interactions can matter; the purpose is to assign each source its own control and validation evidence.
<!-- bilingual-en:end -->

增加独立路径主要收缩第一行。若均值稳定但始终偏离解析答案，应该检查后三行，而不是继续堆路径；这条边界单独见 [[路径数不修模型]]。
<!-- bilingual-en:start -->
More independent draws mainly reduce the first component. A stable estimate that remains far from an analytic benchmark points to the other layers rather than to an insufficient path count.
<!-- bilingual-en:end -->

> [!question]- 可核验自检
> 把 Euler 时间步长减半后，结果系统性改变，但更换 seed 只带来很小波动。首先应怀疑哪一层？
>
> **答案：** 数值近似误差；步长改变的是离散方案，而不是有限样本噪声本身。
> <!-- bilingual-en:start -->
> Numerical-approximation error, because the time step changes the discretisation rather than merely resampling finite-path noise.
> <!-- bilingual-en:end -->

## 来源与核验

- Paul Glasserman (2003), [*Monte Carlo Methods in Financial Engineering*, Chapter 1 “Foundations”](https://doi.org/10.1007/978-0-387-21617-1)：支持把 Monte Carlo 抽样误差与金融路径、数值实现及模型问题分开讨论。
- Art B. Owen, [*Monte Carlo Theory, Methods and Examples*, Chapter 2](https://artowen.su.domains/mc/Ch-intro.pdf)：支持简单 Monte Carlo 抽样误差、估计标准误及其适用条件。
- NASA, [*NASA-STD-7009B: Standard for Models and Simulations*, §§4.2.3–4.2.6](https://standards.nasa.gov/sites/default/files/standards/NASA/B/1/NASA-STD-7009B-Final-3-5-2024.pdf)：核对代码实现验证、解验证、数值近似误差与面向现实用途的模型验证是不同证据责任；本卡的四层表是对这些责任与 Monte Carlo 抽样误差的诊断性整理，不声称四项可加。
