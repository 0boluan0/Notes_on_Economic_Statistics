---
aliases:
  - "VAR 多步预测误差累积未来创新且协方差随期限展开"
  - VAR multi-step forecast error
  - VAR 预测误差协方差
student_os: knowledge-atom
atom_id: TS-VAR-008
atom_set: vector-autoregression
atom_type: derivation
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR因果VMA表示]]"
related:
  - "[[VAR无条件协方差]]"
  - "[[预测误差方差分解]]"
  - "[[ARMA多步预测]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# VAR 多步预测误差累积未来创新且协方差随期限展开
<!-- bilingual-en:start -->
*A VAR multi-step forecast error accumulates future innovations and has a horizon-specific covariance*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> VAR 的 $h$ 步预测误差由预测时点之后、截至目标时点的创新线性叠加；其协方差是对应 VMA 系数与创新协方差的有限和，并随预测期限变化。

设 $\mathcal H_t$ 是由常数与截至 $t$ 的系统变量张成的闭线性预测空间，$\widehat y^{L}_{t+h\mid t}$ 是 $y_{t+h}$ 在该空间上的最佳线性预测。稳定 VAR 的 VMA 系数为 $\Phi_0=I,\Phi_1,\ldots$。在创新零均值、有限方差且跨期不相关时，对 $h\ge1$，
$$
y_{t+h}-\widehat y^{L}_{t+h\mid t}
=\sum_{i=0}^{h-1}\Phi_i u_{t+h-i}.
$$
较早的创新已被 $\mathcal H_t$ 中的过去信息线性吸收，不再构成预测误差；未知的是 $u_{t+1},\ldots,u_{t+h}$。因此
$$
\Sigma_h
=\operatorname{Var}\!\left(y_{t+h}-\widehat y^{L}_{t+h\mid t}\right)
=\sum_{i=0}^{h-1}\Phi_i\Sigma_u\Phi_i'.
$$
$h=1$ 时 $\Sigma_1=\Sigma_u$；期限增加时，会加入更多未来创新的传播贡献。稳定且有限二阶矩时，$\Sigma_h$ 随 $h\to\infty$ 收敛到平稳解的无条件协方差。

只有二阶白噪声条件时，上面保证的是最佳线性预测，不一定是使用全部信息的非线性条件期望。若进一步有 $E(u_{t+s}\mid\mathcal F_t)=0$ 对所有 $s>0$ 成立，例如在合适的马氏差序列、独立创新或联合 Gaussian 线性系统条件下，$\widehat y^{L}_{t+h\mid t}$ 才与 $E(y_{t+h}\mid\mathcal F_t)$ 重合。

这个总线性预测误差协方差完全由简约型 VAR 决定，不需要先识别结构冲击。只有进一步问“每个经济冲击贡献多少”时，才进入依赖识别方案的 FEVD。预测区间还可能需要参数估计不确定性、非高斯分布或条件异方差修正，不能把 $\Sigma_h$ 当作所有不确定性的自动汇总。

> [!question]- 自检
> 为什么两步预测误差不仅包含 $u_{t+2}$，还包含经 $\Phi_1$ 传播的 $u_{t+1}$？
>
> **答案：** 因为在时点 $t$，$u_{t+1}$ 和 $u_{t+2}$ 都尚未知；前者还会通过系统动态影响 $y_{t+2}$。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 2 章：核对多步预测误差与预测均方误差矩阵。
- [[VAR因果VMA表示]]：复用 $\Phi_h$ 递推和跨期创新展开。
