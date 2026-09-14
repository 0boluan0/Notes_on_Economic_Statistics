---
aliases:
  - "ARCH(q) 用过去创新平方递推条件方差"
  - ARCH(q) model
  - Autoregressive conditional heteroskedasticity
  - ARCH 模型
student_os: knowledge-atom
atom_id: TS-VOL-002
atom_set: conditional-volatility
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[条件异方差]]"
  - "[[条件尺度与标准化冲击]]"
related:
  - "[[ARCH-LM检验]]"
  - "[[GARCH(p,q)模型]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# ARCH(q) 用过去创新平方递推条件方差
<!-- bilingual-en:start -->
*ARCH(q) recursively models conditional variance with past squared innovations*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> ARCH($q$) 写成
> $$\varepsilon_t=\sqrt{h_t}z_t,\qquad
> h_t=\omega+\sum_{i=1}^{q}\alpha_i\varepsilon_{t-i}^2,$$
> 其中 $E(z_t\mid\mathcal F_{t-1})=0$、$E(z_t^2\mid\mathcal F_{t-1})=1$。过去的大创新不论正负，都通过平方项提高当前条件方差；$q$ 决定直接进入方差方程的冲击记忆长度。

标准线性 ARCH 通常取 $\omega>0$、$\alpha_i\ge0$，从而对所有历史保证 $h_t>0$。这只是方差非负条件；有限无条件方差、严格平稳和高阶矩存在还要另外检查，不能用“参数非负”一句包办。

ARCH 模型允许 $\varepsilon_t$ 在均值上是鞅差、跨期线性不相关，却让 $\varepsilon_t^2$ 可预测。因此它正好说明“方向不可预测”与“风险恒定”是两回事。

> [!question]- 自检
> ARCH(2) 中 $\alpha_2=0$ 时，模型是否仍需要称为二阶 ARCH？
>
> **答案：** 不需要。它退化为 ARCH(1)；报告阶数时应使用最小有效表示，而不是保留无作用的最高滞后。

## 来源与核验

- [Engle (1982)](https://doi.org/10.2307/1912773)：核对 ARCH 线性条件方差、有限滞后与 LM 检验的原始定义。
- [[01_Math/06_时间序列分析/lecture.pdf#page=161|课程讲义 pp. 161–162]]：核对课程中的 ARCH(1) 记号与鞅差解释。
