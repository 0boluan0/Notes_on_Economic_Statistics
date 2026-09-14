---
aliases:
  - "GARCH 多步方差预测回归长期方差且半衰期只描述期望缺口"
  - Multi-step GARCH variance forecast
  - GARCH variance half-life
  - GARCH 多步波动预测
student_os: knowledge-atom
atom_id: TS-VOL-017
atom_set: conditional-volatility
atom_type: forecasting-rule
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH一步方差预测]]"
  - "[[GARCH有限方差条件]]"
related:
  - "[[GARCH参数持久性]]"
  - "[[IGARCH平稳与矩边界]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH 多步方差预测回归长期方差且半衰期只描述期望缺口
<!-- bilingual-en:start -->
*Multi-step GARCH variance forecasts revert to long-run variance, and half-life describes only the expected gap*
<!-- bilingual-en:end -->

> [!summary] 原子规则
> 对有限二阶矩的 GARCH(1,1)，令 $\rho=\alpha+\beta<1$、$\bar h=\omega/(1-\rho)$。先用已实现的 $\varepsilon_t^2$ 算出 $h_{t+1\mid t}$，此后对 $j\ge1$：
> $$h_{t+j\mid t}=\bar h+\rho^{j-1}\bigl(h_{t+1\mid t}-\bar h\bigr).$$
> 未来平方创新尚未观察，用 $E_t(\varepsilon_{t+k}^2)=h_{t+k\mid t}$ 逐步递推。

当 $0<\rho<1$，期望方差缺口的半衰期为
$$k_{1/2}=\frac{\log(1/2)}{\log\rho}.$$
它以数据频率的“期数”为单位，描述的是模型内**期望条件方差与长期方差的差**减半，不是实现波动、资产价格冲击或经济危机影响保证在该时点减半。

期限为 $H$ 的累计收益风险通常需要汇总各期预测方差并考虑均值动态与跨期协方差；不能把某个终点 $h_{t+H\mid t}$ 直接当成整段累计方差。

> [!question]- 自检
> $\rho=0.9$、$h_{t+1\mid t}-\bar h=4$，三步前预测的缺口是多少？
>
> **答案：** $j=3$ 时缺口为 $0.9^{2}\times4=3.24$；指数是 $j-1$，因为一步预测已作为起点。

## 来源与核验

- [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)：核对 GARCH 二阶递推与长期方差。
- [[01_Math/06_时间序列分析/lecture.pdf#page=174|课程讲义 pp. 173–174]]：核对多步条件方差递推；此卡以正确的一步预测作为起点修正原讲义索引。
