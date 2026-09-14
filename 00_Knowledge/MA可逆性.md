---
aliases:
  - "MA 可逆性规范创新表示但不是有限 MA 的存在条件"
  - MA invertibility
  - Invertible MA process
  - ARMA invertibility
  - 可逆移动平均过程
student_os: knowledge-atom
atom_id: TS-ARMA-007
atom_set: arma-modeling
atom_type: guarantee-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[MA(q)模型]]"
  - "[[滞后算子约定]]"
related:
  - "[[ARMA无限AR表示]]"
  - "[[ARMA公共因子]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# MA 可逆性规范创新表示但不是有限 MA 的存在条件
<!-- bilingual-en:start -->
*MA invertibility normalizes the innovation representation; it is not an existence condition for a finite MA process*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 在 $\theta(z)=1+\theta_1z+\cdots+\theta_qz^q$ 的约定下，可逆性要求 $\theta(z)=0$ 的所有零点位于单位圆外。此时 $\theta(B)^{-1}$ 有稳定的单边幂级数，当前创新可由当前与过去观测恢复。
> <!-- bilingual-en:start -->
> Under the stated plus-sign convention, invertibility requires all zeros of the MA polynomial to lie outside the unit circle. Then the inverse filter is stable and the current innovation can be recovered from current and past observations.
> <!-- bilingual-en:end -->

可逆性不是有限 MA 能否存在或宽平稳的条件。任何有限 MA 都是有限个白噪声项的和；即使不可逆，它仍定义良好并且宽平稳。可逆性首先解决的是**同一观测二阶结构采用哪个创新参数化**。

MA(1) 显示为什么需要规范化。$y_t=\varepsilon_t+\theta\varepsilon_{t-1}$ 与系数 $1/\theta$ 的另一个 MA(1)，在适当缩放创新方差后拥有相同自协方差，因而仅凭二阶信息无法识别哪组参数。若创新 jointly Gaussian，相同均值与自协方差进一步给出相同完整有限维分布；非 Gaussian 时，高阶分布结构未必相同。选取 $|\theta|<1$（等价于零点 $-1/\theta$ 在单位圆外）留下唯一的可逆二阶代表，并使“创新”确实是由过去观测逐步得到的一步线性预测误差。
<!-- bilingual-en:start -->
Every finite MA exists and is covariance-stationary whether or not it is invertible. For MA(1), reciprocal coefficients can yield the same autocovariances after rescaling innovation variance, so invertibility selects a unique second-order representative. Full observational equivalence additionally follows under joint Gaussianity; it need not follow from autocovariances alone for non-Gaussian innovations.
<!-- bilingual-en:end -->

> [!question]- 自检
> $y_t=\varepsilon_t+2\varepsilon_{t-1}$ 是否“不存在”或“不平稳”？
>
> **答案：** 都不是。它存在且宽平稳，但在本文约定下不可逆；可用一个系数 $1/2$、创新方差相应缩放的可逆表示描述同一二阶结构。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §9.4](https://otexts.com/fpp3/MA.html)：核对 MA(1) 等价参数化、可逆规范与单位圆条件。
- [MIT OCW 14.384, Recitation 1](https://ocw.mit.edu/courses/14-384-time-series-analysis-fall-2013/ca390a7534c2594b397af2164697352b_MIT14_384F13_rec1.pdf)：核对 invertibility、innovation recovery 与 ARMA 表示边界。
- [[01_Math/06_时间序列分析/lecture.pdf#page=116|课程讲义 pp. 116–119]]：核对 MA(1) 递归恢复误差的课程例子。
