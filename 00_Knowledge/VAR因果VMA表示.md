---
aliases:
  - "有限二阶矩下稳定 VAR 有唯一因果 VMA 表示"
  - Stable VAR causal VMA
  - VAR 的 VMA 表示
student_os: knowledge-atom
atom_id: TS-VAR-007
atom_set: vector-autoregression
atom_type: theorem
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR稳定根条件]]"
  - "[[白噪声二阶定义]]"
related:
  - "[[ARMA无限MA表示]]"
  - "[[非正规矩阵瞬态]]"
  - "[[VAR多步预测误差]]"
  - "[[VAR无条件协方差]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# 有限二阶矩下稳定 VAR 有唯一因果 VMA 表示
<!-- bilingual-en:start -->
*With finite second moments, a stable VAR has a unique causal VMA representation*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 当 VAR 稳定且创新有有限二阶矩时，它存在唯一的因果协方差平稳解，可写为过去和当前创新的收敛 VMA；任意给定初值只会生成逐渐消失的过渡项，不使全过程从第一期起自动平稳。

令 $\mu$ 为稳定 VAR 的无条件均值。其因果解为
$$
y_t-\mu=\sum_{h=0}^{\infty}\Phi_hu_{t-h},
\qquad \Phi_0=I_K,
$$
其中系数递推为
$$
\Phi_h=\sum_{i=1}^{\min(p,h)}A_i\Phi_{h-i},
\qquad \Phi_h=0\ \text{当 }h<0.
$$
稳定根条件保证这些系数以足够快的速度衰减；若 $u_t$ 是均值为零、跨期不相关且协方差有限的创新，上式在均方意义下收敛，并确定唯一的因果协方差平稳解。

只有 VAR(1) 才有 $\Phi_h=A_1^h$。对 VAR($p>1$)，可以用伴随矩阵写成 $\Phi_h=JF^hJ'$，或使用上面的递推，不能直接把 $A_1^h$ 当作全部动态。

若从某个任意有限初始状态启动，解还带有 $F^t(Y_0-Y_0^{\mathrm{stat}})$ 一类过渡项。稳定性使该项趋于零，所以矩逐渐接近平稳分布；但初始状态若不是从平稳分布抽取，早期观测本身并非严格或弱平稳。稳定性也不排除非正规矩阵在有限期限出现较大瞬态，不能只看谱半径描述短期峰值。

> [!question]- 自检
> 一个稳定 VAR 从固定的全零初值启动，是否从第一期开始就是协方差平稳过程？
>
> **答案：** 不一定。稳定性保证初值效应消失并趋近唯一平稳因果解；只有按平稳分布初始化时，全部时点的矩才立即时间不变。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 2 章：核对稳定 VAR 的因果 VMA、均值与唯一平稳解。
- [[ARMA无限MA表示]]：复用标量 ARMA 的因果展开接口。
- [[非正规矩阵瞬态]]：保留渐近稳定与有限时传播强度的区别。
