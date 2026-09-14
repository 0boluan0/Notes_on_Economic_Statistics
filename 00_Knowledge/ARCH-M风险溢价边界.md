---
aliases:
  - "ARCH-M 只把条件风险量放入均值而风险溢价符号需估计"
  - ARCH-in-mean model
  - GARCH-M risk premium boundary
  - ARCH-M 模型
student_os: knowledge-atom
atom_id: TS-VOL-023
atom_set: conditional-volatility
atom_type: model-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[条件尺度与标准化冲击]]"
related:
  - "[[GARCH-X因果边界]]"
  - "[[波动不对称与杠杆]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# ARCH-M 只把条件风险量放入均值而风险溢价符号需估计
<!-- bilingual-en:start -->
*ARCH-in-mean places a conditional risk measure in the mean, but the risk-premium sign must be estimated*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> ARCH-M/GARCH-M 在均值方程中加入条件风险量，例如
> $$y_t=\mu+\delta h_t+\varepsilon_t$$
> 或使用 $\sqrt{h_t}$、$\log h_t$。模型定义是“条件均值依赖条件风险量”，不是 $\delta>0$；符号、量纲和经济解释取决于资产、超额收益定义、风险量选择与信息集。

“风险越高，要求回报越高”可以是理论假设，但不能硬编码成 ARCH-M 的统计定义。原始期限结构应用发现某些正向关系，而后续股票收益研究也报告过零或负关系；不同方差规格和不对称处理会改变估计。

$h_t$ 是生成回归量，和其他均值变量共同估计。均值或方差设定错误、代理信息集不足、风险量测量误差都可能污染 $\hat\delta$。显著系数支持当前联合模型中的条件关联，不自动识别结构性风险价格。

> [!question]- 自检
> 若估得 $\hat\delta<0$，模型是否因违反 ARCH-M 定义而无效？
>
> **答案：** 不是。ARCH-M 不规定符号；应检查经济对象、风险量、信息集、规格与推断，再解释负关系。

## 来源与核验

- [Engle, Lilien & Robins (1987), *Estimating Time Varying Risk Premia in the Term Structure: The ARCH-M Model*](https://doi.org/10.2307/1913242)：核对条件方差进入均值的原始定义。
- [Glosten, Jagannathan & Runkle (1993)](https://doi.org/10.1111/j.1540-6261.1993.tb05128.x)：核对风险—收益符号对方差规格与信息集敏感的证据。
