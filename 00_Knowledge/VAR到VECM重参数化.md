---
aliases:
  - "VAR(p) 可重参数化为含 Pi 水平项的 VECM"
  - VAR to VECM reparameterization
  - VAR 改写为 VECM
student_os: knowledge-atom
atom_id: TS-CI-016
atom_set: cointegration-error-correction
atom_type: derivation
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR(p)模型]]"
related:
  - "[[VECM的Π秩]]"
  - "[[ECM长短期结构]]"
part_of:
  - "[[协整与误差修正模型.canvas]]"
---

# VAR(p) 可重参数化为含 Pi 水平项的 VECM
<!-- bilingual-en:start -->
*A VAR(p) can be reparameterized as a VECM with a Pi level term*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 同一个 VAR($p$) 可通过加减滞后水平改写为差分动态加一个滞后水平项；$\Pi$ 汇总长期水平反馈，$\Gamma_i$ 汇总短期差分反馈。

从
$$x_t=A_1x_{t-1}+\cdots+A_px_{t-p}+u_t$$
出发，一种常用约定是
$$
\Delta x_t=\Pi x_{t-1}+\sum_{i=1}^{p-1}\Gamma_i\Delta x_{t-i}+u_t,
$$
其中
$$
\Pi=\sum_{j=1}^{p}A_j-I,
\qquad
\Gamma_i=-\sum_{j=i+1}^{p}A_j.
$$
有些教材把长期矩阵定义为 $I-\sum A_j$，于是方程前出现负号；只要定义与方程一致，经济内容相同。

这一步是代数重参数化，不是检验结果。只有再结合变量至多 $I(1)$、系统根和 $\Pi$ 的秩，才能把 $\Pi x_{t-1}$ 解释成误差修正项。把“能写成 VECM”误当成“已经证明协整”，会把恒等变形和统计识别混在一起。

> [!question]- 自检
> VAR 能代数改写为含 $\Pi$ 的差分形式，是否已经证明 $0<\operatorname{rank}(\Pi)<n$？
>
> **答案：** 没有。秩和整合阶数仍需从模型条件与数据中判断。

## 来源与核验

- [Johansen (1988)](https://doi.org/10.1016/0165-1889(88)90041-3)：核对 VAR 的误差修正重参数化。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程的 $\Pi$、$\Gamma_i$ 记号。
