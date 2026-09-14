---
aliases:
  - "ECM 同时保留滞后长期偏离与短期差分动态"
  - Error-correction model anatomy
  - ECM 长短期分解
student_os: knowledge-atom
atom_id: TS-CI-006
atom_set: cointegration-error-correction
atom_type: model-structure
status: source-checked
mastery_state: unassessed
requires:
  - "[[Granger表示定理]]"
related:
  - "[[误差修正方向判读]]"
  - "[[VECM的α与β]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# ECM 同时保留滞后长期偏离与短期差分动态
<!-- bilingual-en:start -->
*An ECM keeps the lagged long-run deviation and short-run differenced dynamics in one model*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 误差修正模型把“上期离长期关系多远”和“近期变化如何传递”分开写进同一个平稳方程，而不是在长期回归与短期模型之间二选一。

二变量 ECM 可写为
$$
\Delta y_t=c+\alpha e_{t-1}+\sum_i\phi_i\Delta y_{t-i}+\sum_j\theta_j\Delta x_{t-j}+u_t,
\qquad e_{t-1}=y_{t-1}-\beta_0-\beta_1x_{t-1}.
$$
$e_{t-1}$ 承载水平变量的长期约束；差分滞后承载短期惯性和相互预测；$u_t$ 是在该信息集下尚未解释的创新。$\alpha e_{t-1}$ 的存在并不保证每期都回到均衡，只有结合符号、大小、其他方程和稳定性条件，才能判断调整机制。

若协整成立却删去 $e_{t-1}$，短期方程可能遗漏一个持久的可预测分量；若不协整却加入一个非平稳“误差”，方程两侧的整合阶数可能不相容。

> [!question]- 自检
> ECM 中的 $\Delta x_{t-1}$ 与 $e_{t-1}$ 各回答什么问题？
>
> **答案：** 前者回答近期变化怎样传递，后者回答上一期偏离长期组合后本期怎样修正。

## 来源与核验

- [Engle & Granger (1987)](https://doi.org/10.2307/1913236)：核对误差修正表示与协整关系。
- [[01_Math/06_时间序列分析/07_协整和误差修正模型.md]]：对照课程 ECM 记号与例题。
