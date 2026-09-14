---
aliases:
  - "Student-t GARCH 必须标准化冲击才能让 h_t 保持条件方差含义"
  - Standardized Student-t GARCH innovations
  - GARCH Student-t normalization
  - t 分布 GARCH 标准化
student_os: knowledge-atom
atom_id: TS-VOL-013
atom_set: conditional-volatility
atom_type: distribution-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[条件尺度与标准化冲击]]"
  - "[[GARCH条件似然]]"
related:
  - "[[Gaussian QMLE]]"
  - "[[ARMA预测区间]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# Student-t GARCH 必须标准化冲击才能让 h_t 保持条件方差含义
<!-- bilingual-en:start -->
*Student-t GARCH innovations must be standardized for h_t to remain a conditional variance*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 若 $u_t\sim t_\nu$ 采用普通 Student-$t$ 尺度，则 $\operatorname{Var}(u_t)=\nu/(\nu-2)$，只在 $\nu>2$ 时有限。要让
> $$\varepsilon_t=\sqrt{h_t}\,z_t$$
> 中的 $h_t$ 仍等于条件方差，须令
> $$z_t=u_t\sqrt{\frac{\nu-2}{\nu}},\qquad E(z_t^2)=1.$$

若软件用的是未标准化 $t$，$h_t$ 只是条件**尺度**，实际条件方差要再乘 $\nu/(\nu-2)$。不同包对“scale”“variance”和自由度参数的定义可能不同，所以迁移参数或复算 VaR 前必须读文档。

Student-$t$ 可比 Gaussian 更好地描述厚尾，但不自动处理偏态、不对称方差响应、极端断点或参数不确定性。$\nu\le2$ 时连条件方差口径都不存在，不能仍称 $h_t$ 为二阶矩。

> [!question]- 自检
> 直接取 $u_t\sim t_5$ 并写 $\varepsilon_t=\sqrt{h_t}u_t$，此时 $\operatorname{Var}(\varepsilon_t\mid\mathcal F_{t-1})$ 等于多少？
>
> **答案：** 等于 $(5/3)h_t$，而不是 $h_t$。要保持 $h_t$ 的方差含义，应把 $u_t$ 乘以 $\sqrt{3/5}$。

## 来源与核验

- [Bollerslev (1987), *A Conditionally Heteroskedastic Time Series Model for Speculative Prices and Rates of Return*](https://doi.org/10.2307/1925546)：核对条件 Student-$t$ 在 GARCH 中处理厚尾的原始应用。
- [[01_Math/06_时间序列分析/lecture.pdf#page=168|课程讲义 p. 168]]：核对课程将 Gaussian 与 $t$ 创新作为不同似然规格。
