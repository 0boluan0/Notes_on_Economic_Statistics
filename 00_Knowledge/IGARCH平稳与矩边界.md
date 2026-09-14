---
aliases:
  - "IGARCH 可严平稳但没有有限无条件方差"
  - IGARCH strict stationarity boundary
  - Integrated GARCH
  - IGARCH 模型
student_os: knowledge-atom
atom_id: TS-VOL-018
atom_set: conditional-volatility
atom_type: boundary-case
status: source-checked
mastery_state: unassessed
requires:
  - "[[GARCH严格平稳条件]]"
  - "[[GARCH多步方差预测]]"
related:
  - "[[方差断点伪GARCH持久性]]"
part_of:
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
---

# IGARCH 可严平稳但没有有限无条件方差
<!-- bilingual-en:start -->
*IGARCH can be strictly stationary while lacking a finite unconditional variance*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> IGARCH(1,1) 通常指 $\alpha+\beta=1$ 的 GARCH(1,1)。这个等式处在有限二阶矩边界，因此 $\omega>0$ 时不存在有限的 $E(h_t)$，也不能使用 $\omega/(1-\alpha-\beta)$ 作为长期方差。

“integrated” 不等于“绝不可能严格平稳”。若随机系数满足
$$E\log(\alpha z_t^2+\beta)<0,$$
典型的非退化、$\alpha>0$ IGARCH 仍可有严格平稳遍历解，只是其二阶矩发散。严格平稳、有限方差与二阶意义上的冲击持久性必须分开陈述。

Jensen 不等式给出边界：当 $E(z_t^2)=1$ 且 $\alpha+\beta=1$ 时，$E\log(\alpha z_t^2+\beta)\le\log 1=0$；只有随机系数非退化时才通常严格小于零。若 $\alpha=0,\beta=1$，随机系数恒为一、log-moment 等于零，配合 $\omega>0$ 的递推不会产生上述严格平稳解。

在 $\rho=1$ 下，多步期望方差不向有限 $\bar h$ 回归，而是
$$h_{t+j\mid t}=h_{t+1\mid t}+(j-1)\omega.$$
这条期望公式本身可能为无穷均值世界中的条件预测，不能据此把每条实现路径都描述为线性爆炸。

> [!question]- 自检
> 一个 IGARCH 拟合满足 log-moment 为负。可以同时说“严平稳”和“无有限无条件方差”吗？
>
> **答案：** 可以。严平稳约束分布的时间平移，二阶矩是否有限是另一条条件。

## 来源与核验

- [Nelson (1990)](https://doi.org/10.1017/S0266466600005296)：核对 IGARCH 正漂移情形的严格平稳遍历性与矩边界。
- [[01_Math/06_时间序列分析/lecture.pdf#page=175|课程讲义 p. 175]]：核对课程的 IGARCH 预测主题；此卡修正“永远记住”与严格平稳混写的表述。
