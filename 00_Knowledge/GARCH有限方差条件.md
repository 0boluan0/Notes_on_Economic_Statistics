---
aliases:
  - "GARCH(1,1) 的有限无条件方差要求 alpha+beta 小于一"
  - GARCH(1,1) finite unconditional variance
  - GARCH covariance stationarity
  - GARCH 长期方差
student_os: knowledge-atom
atom_id: TS-VOL-006
atom_set: conditional-volatility
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH严格平稳条件]]"
  - "[[宽平稳定义]]"
related:
  - "[[GARCH参数持久性]]"
  - "[[IGARCH平稳与矩边界]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# GARCH(1,1) 的有限无条件方差要求 alpha+beta 小于一
<!-- bilingual-en:start -->
*Finite unconditional variance in GARCH(1,1) requires alpha plus beta below one*
<!-- bilingual-en:end -->

> [!summary] 原子定理
> 对
> $$h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1},
> \qquad E(z_t^2)=1,$$
> 若 $\omega>0$、$\alpha,\beta\ge0$，则有限且时间不变的二阶矩存在当且仅当
> $$\alpha+\beta<1.$$
> 此时
> $$\bar h=E(h_t)=E(\varepsilon_t^2)=\frac{\omega}{1-\alpha-\beta}.$$

推导只需对递推取无条件期望，并用
$E(\varepsilon_{t-1}^2)=E(h_{t-1}z_{t-1}^2)=E(h_{t-1})$。但这一步已经假定目标二阶矩有限；当 $\alpha+\beta\ge1$ 时，把同一个代数式写成负数或“无穷公式”都不是合法长期方差。

对 GARCH($p,q$)，相同标准化口径下有限二阶矩条件相应变为 $\sum_i\alpha_i+\sum_j\beta_j<1$。它描述协方差平稳，不应简称为所有意义下的“平稳”。

> [!question]- 自检
> $\omega=0.02,\alpha=0.08,\beta=0.90$ 时长期方差是多少？
>
> **答案：** $0.02/(1-0.98)=1$。数值很大正反映分母接近零；还应核对单位与收益缩放，不能只看公式。

## 来源与核验

- [Bollerslev (1986)](https://doi.org/10.1016/0304-4076(86)90063-1)：Theorem 1 核对宽平稳条件与无条件方差。
- [[01_Math/06_时间序列分析/lecture.pdf#page=164|课程讲义 pp. 163–164]]：核对课程的 GARCH($p,q$) 长期方差表达。
