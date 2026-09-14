---
aliases:
  - "GARCH(1,1) 严平稳由 log-moment 条件而非 alpha+beta 小于一刻画"
  - GARCH(1,1) strict stationarity log-moment condition
  - Strict stationarity of GARCH(1,1)
  - GARCH(1,1) 严平稳条件
student_os: knowledge-atom
atom_id: TS-VOL-005
atom_set: conditional-volatility
atom_type: theorem-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH(p,q)模型]]"
  - "[[严平稳定义]]"
related:
  - "[[GARCH有限方差条件]]"
  - "[[IGARCH平稳与矩边界]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH(1,1) 严平稳由 log-moment 条件而非 alpha+beta 小于一刻画
<!-- bilingual-en:start -->
*Strict stationarity of GARCH(1,1) is governed by a log-moment condition, not by alpha plus beta below one*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 对 $\omega>0$ 的 GARCH(1,1)
> $$h_t=\omega+(\alpha z_{t-1}^2+\beta)h_{t-1},$$
> 在常见非退化条件下，存在与未来创新无关的唯一严格平稳因果解的核心条件是
> $$E\!\left[\log(\alpha z_t^2+\beta)\right]<0,$$
> 并需相应的对数可积条件。$\alpha+\beta<1$ 是更强的有限二阶矩条件，不是严格平稳的定义。

原因是严格平稳解来自随机系数乘积持续收缩；收缩看的是长期平均**对数增长率**。若 $E(z_t^2)=1$ 且 $\alpha+\beta<1$，Jensen 不等式通常可推出 log-moment 为负，所以有限方差条件足以给严格平稳；反向却不成立。

这里的标量公式只针对 GARCH(1,1)。一般 GARCH($p,q$) 要把递推写成随机矩阵乘积，由最高 Lyapunov 指数是否为负来刻画收缩；不能把 $E\log(\alpha z_t^2+\beta)<0$ 原样套到所有阶数。

这一区分解释了为什么某些过程的分布可以时间不变，却没有有限无条件方差。此时不能用协方差、ACF 或“长期方差”公式描述它。

> [!question]- 自检
> 已估得 $\alpha+\beta=1$，是否仅凭这个等式就能断言不存在严格平稳解？
>
> **答案：** 不能。还要检查 $E\log(\alpha z_t^2+\beta)$；非退化标准化冲击下它可能小于零，因此可严平稳但二阶矩发散。

## 来源与核验

- [Nelson (1990), *Stationarity and Persistence in the GARCH(1,1) Model*](https://doi.org/10.1017/S0266466600005296)：核对严格平稳、遍历与矩存在的分离。
- [Bougerol & Picard (1992), *Stationarity of GARCH Processes*](https://doi.org/10.1016/0304-4076(92)90067-2)：核对严格平稳解的必要充分条件框架。
