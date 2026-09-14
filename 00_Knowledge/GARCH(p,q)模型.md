---
aliases:
  - "GARCH(p,q) 让条件方差同时依赖过去冲击与自身"
  - GARCH(p,q) model
  - Generalized autoregressive conditional heteroskedasticity
  - GARCH 模型
student_os: knowledge-atom
atom_id: TS-VOL-003
atom_set: conditional-volatility
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARCH(q)模型]]"
related:
  - "[[平方创新ARMA表示]]"
  - "[[ARCH-GARCH正性条件]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH(p,q) 让条件方差同时依赖过去冲击与自身
<!-- bilingual-en:start -->
*GARCH(p,q) lets conditional variance depend on both past shocks and its own lags*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> 采用“$p$ 个方差滞后、$q$ 个平方创新滞后”的 Bollerslev 记号，
> $$\varepsilon_t=\sqrt{h_t}z_t,$$
> $$h_t=\omega+\sum_{i=1}^{q}\alpha_i\varepsilon_{t-i}^2+
> \sum_{j=1}^{p}\beta_j h_{t-j}.$$
> ARCH($q$) 是 $p=0$ 的特例。方差自身的递推让低阶 GARCH 往往能比高阶 ARCH 更简约地表示缓慢衰减的波动动态。

不同教材和软件会交换 $p,q$ 的顺序，所以不能只看“GARCH(1,2)”这个标签；应同时写清哪一阶对应 $h$、哪一阶对应 $\varepsilon^2$。模型还必须连同条件均值、信息集与 $z_t$ 的条件分布一起定义。

GARCH 是条件二阶矩模型。它不自动说明波动的经济原因，也不等于随机波动率模型：前者让 $h_t$ 由过去观测递推，后者通常另设不可直接观察的潜在波动状态。

> [!question]- 自检
> 为什么 GARCH(1,1) 可能比 ARCH(20) 参数更少，却仍表现出较长的波动记忆？
>
> **答案：** 因为 $h_{t-1}$ 本身已经汇总更早的平方冲击；把它递推下去会产生无限衰减权重，而不必逐个估计 20 个 ARCH 系数。

## 来源与核验

- [Bollerslev (1986), *Generalized Autoregressive Conditional Heteroskedasticity*](https://doi.org/10.1016/0304-4076(86)90063-1)：核对 GARCH($p,q$) 定义、ARCH 特例与简约性。
- [Bollerslev (2023), “GARCH Musings”](https://public.econ.duke.edu/~boller/Papers/GARCH_Musings.pdf)：核对 GARCH(1,1) 的原始递推动机。
