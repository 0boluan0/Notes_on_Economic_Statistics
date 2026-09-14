---
aliases:
  - "I(d) 表示恰好差分 d 次成为 I(0) 而 I(0) 不等于 iid"
  - Integration order
  - Order of integration
  - I(d)
  - 单整阶数
student_os: knowledge-atom
atom_id: TS-UR-006
atom_set: trends-unit-roots-differencing
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[单位根最小表示]]"
  - "[[宽平稳定义]]"
related:
  - "[[普通与季节差分]]"
  - "[[协整与差分边界]]"
part_of:
  - "[[趋势、单位根与差分.canvas]]"
---

# I(d) 表示恰好差分 d 次成为 I(0) 而 I(0) 不等于 iid
<!-- bilingual-en:start -->
*I(d) means that exactly d ordinary differences produce an I(0) process; I(0) is not the same as i.i.d.*
<!-- bilingual-en:end -->

> [!summary] 原子定义
> 在本课程的整数单整口径下，若
> $$(1-B)^dy_t\sim I(0),$$
> 而 $d>0$ 时 $(1-B)^{d-1}y_t$ 仍非 $I(0)$，则 $y_t\sim I(d)$。$d$ 是达到 $I(0)$ 的最小普通差分次数，不是任意做了多少次差分。
> <!-- bilingual-en:start -->
> The integration order is the minimum number of ordinary differences required to obtain an I(0) process.
> <!-- bilingual-en:end -->

$I(0)$ 表示适合平稳短期动态与常规渐近分析的一类过程；它可以有 ARMA 自相关、非 Gaussian 边际甚至条件异方差，不等于 i.i.d. 或白噪声。不同文献对 $I(0)$ 的弱依赖正则条件略有差异，使用检验或定理时必须沿用该来源的定义，不能把“看起来平稳”当作全部条件。

> [!question]- 自检
> 平稳 AR(1) 是否可以是 $I(0)$？若可以，它为何不一定是 i.i.d.？
>
> **答案：** 可以；$I(0)$ 允许稳定自相关，而 i.i.d. 要求跨期独立且同分布。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=259|课程讲义 ARIMA(p,d,q)]]：核对单整阶数与差分次数。
- [Engle & Granger (1987), *Co-Integration and Error Correction: Representation, Estimation, and Testing*](https://doi.org/10.2307/1913236)：核对 $I(d)$ 的最小差分阶定义以及 $I(0)$ 作为基准短记忆对象的口径。
