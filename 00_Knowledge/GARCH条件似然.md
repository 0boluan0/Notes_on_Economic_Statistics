---
aliases:
  - "GARCH 条件似然必须说明创新分布与初值处理"
  - GARCH conditional likelihood
  - GARCH likelihood initialization
  - GARCH 条件似然初始化
student_os: knowledge-atom
atom_id: TS-VOL-011
atom_set: conditional-volatility
atom_type: estimation-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH(p,q)模型]]"
related:
  - "[[Gaussian QMLE]]"
  - "[[Student-t GARCH标准化]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH 条件似然必须说明创新分布与初值处理
<!-- bilingual-en:start -->
*A GARCH conditional likelihood must state both the innovation distribution and the treatment of initial values*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 若假定 $z_t\mid\mathcal F_{t-1}\sim N(0,1)$，单期条件对数似然为
> $$\ell_t=-\frac12\left[\log(2\pi)+\log h_t+\frac{\varepsilon_t^2}{h_t}\right].$$
> 把 $\varepsilon_t$ 与 $h_t$ 的递推代入并数值最大化，才得到条件 Gaussian MLE；换用 Student-$t$、GED 或偏态分布会改变密度与参数含义。

样本开头没有全部滞后 $\varepsilon$ 和 $h$。常见做法包括用样本方差、无条件方差、backcast 或一段 presample 值初始化，并可能丢弃 burn-in。稳定递推下初值影响通常随时间衰减，但在短样本、高持久性或近边界模型中不能假装不存在。

因此比较两个软件或两组模型时，要锁定：同一均值方程、同一创新分布、同一有效样本、同一初始化和同一常数项。只看到“log likelihood”相近而忽略这些口径，不是可比的估计。

> [!question]- 自检
> 两个 GARCH 模型使用相同数据，但一个以长期方差初始化、另一个删去前 100 期。其 AIC 能否直接比较？
>
> **答案：** 不能直接比较；有效样本和似然构造不同。应先统一样本与初始化口径，再比较最大化似然及参数数目。

## 来源与核验

- [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)：核对 GARCH 回归模型的条件似然构造。
- [[01_Math/06_时间序列分析/lecture.pdf#page=168|课程讲义 pp. 168–169]]：核对 Gaussian 条件对数似然与数值优化。
