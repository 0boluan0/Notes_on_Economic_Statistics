---
aliases:
  - "VaR 与 ES 可用严格一致的联合评分比较预测程序"
  - Strictly consistent joint scoring for VaR and ES
student_os: knowledge-atom
atom_id: RM-VAR-024
atom_set: var-es-backtesting
atom_type: forecast-comparison
status: source-checked
mastery_state: unassessed
requires:
  - "[[VaR定义]]"
  - "[[ES定义]]"
related:
  - "[[VaR-ES联合识别]]"
  - "[[滚动起点评估]]"
  - "[[风险模型验证边界]]"
part_of:
  - "[[VaR、ES与回测.canvas|VaR、ES与回测]]"
---

# VaR 与 ES 可用严格一致的联合评分比较预测程序
<!-- bilingual-en:start -->
*VaR and ES forecasts can be compared with a strictly consistent joint scoring function*
<!-- bilingual-en:end -->

> [!summary] 联合评分回答相对预测表现
> 在合适的行动域、可积性和正则条件下，存在严格一致联合评分，使真实 $(\mathrm{VaR}_\alpha,\mathrm{ES}_\alpha)$ 二元组唯一最小化期望分数。它允许比较两个预测程序，不等于绝对校准检验。
>
> <!-- bilingual-en:start -->
> On an appropriate action domain and under integrability and regularity conditions, a strictly consistent joint score is uniquely minimised in expectation by the true $(\mathrm{VaR}_\alpha,\mathrm{ES}_\alpha)$ pair. It compares forecasting procedures; it is not an absolute calibration test.
> <!-- bilingual-en:end -->

对预先选定的合法联合评分 $S_\alpha$，比较模型 A 与 B：

$$
\Delta_n=\frac1n\sum_{t=1}^n
\left[
S_\alpha(q_t^A,e_t^A,L_t)
-S_\alpha(q_t^B,e_t^B,L_t)
\right].
$$

$\Delta_n<0$ 表示 A 在这项评分下样本平均损失较低。若要声称总体预测表现不同，还需对 score differential 使用与时间依赖相容的推断，例如 Diebold–Mariano 型检验。

<!-- bilingual-en:start -->
$\Delta_n<0$ means A has the lower sample-average loss under this score. A population claim additionally requires inference for the score differential that accommodates time dependence, such as a Diebold–Mariano-type test.
<!-- bilingual-en:end -->

> [!question]- 自检
> A 的联合评分显著优于 B，能否推出 A 已通过绝对校准？
>
> **答案：** 不能。它只支持相对比较；绝对校准要检查 [[VaR-ES联合识别]] 的矩条件。
>
> <!-- bilingual-en:start -->
> **Self-check:** If model A significantly outperforms model B under a joint score, does that imply that A is absolutely calibrated?
>
> **Answer:** No. The result supports a relative comparison only; absolute calibration requires the moments from [[VaR-ES联合识别|joint VaR–ES identification]].
> <!-- bilingual-en:end -->

## 边界

- 评分必须在比较前选定，不能看见结果后挑最有利的评分。
- 合法行动域要求 VaR 与 ES 使用同一水平、期限和符号约定。
- 一项评分领先不证明完整条件分布正确。

<!-- bilingual-en:start -->
- Select the score before comparison rather than after observing results.
- The forecast pair must share the same level, horizon, and sign convention.
- Winning one score does not validate the full conditional distribution.
<!-- bilingual-en:end -->

## 来源与核验

- Fissler & Ziegel, [*Higher Order Elicitability and Osband's Principle*](https://doi.org/10.1214/16-AOS1439)：核对 VaR–ES 二元组的严格一致联合评分类。
- Fissler, Ziegel & Gneiting, [*Expected Shortfall is jointly elicitable with Value at Risk*](https://arxiv.org/abs/1507.00244)：核对联合评分、预测比较和 Diebold–Mariano 推断。
- Nolde & Ziegel, [*Elicitability and Backtesting*](https://arxiv.org/abs/1608.05498)：核对传统校准回测与比较回测的区别。
