---
aliases:
  - "可逆 ARMA 能用当前与过去观测恢复为无限 AR 表示"
  - Infinite AR representation of ARMA
  - Innovation recovery from observations
  - ARMA 的 AR 无穷表示
student_os: knowledge-atom
atom_id: TS-ARMA-010
atom_set: arma-modeling
atom_type: representation
status: source-checked
mastery_state: unassessed
requires:
  - "[[ARMA(p,q)模型]]"
  - "[[MA可逆性]]"
related:
  - "[[ARMA似然初值处理]]"
  - "[[ARMA多步预测]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# 可逆 ARMA 能用当前与过去观测恢复为无限 AR 表示
<!-- bilingual-en:start -->
*An invertible ARMA recovers innovations from current and past observations through an infinite-AR filter*
<!-- bilingual-en:end -->

> [!summary] 原子表示
> 若 $\theta(z)$ 的零点都在单位圆外，则
> $$\frac{\phi(B)}{\theta(B)}=\pi(B)=1-\pi_1B-\pi_2B^2-\cdots,$$
> 并且
> $$\varepsilon_t=\pi(B)(y_t-\mu).$$
> 当前创新因此可由 $y_t,y_{t-1},\ldots$ 的稳定线性滤波得到。
> <!-- bilingual-en:start -->
> When the MA polynomial is invertible, $\phi(B)/\theta(B)$ has a stable one-sided expansion, so the current innovation is a linear filter of the current and past observations.
> <!-- bilingual-en:end -->

MA(1) 的例子最直观：若 $y_t=\varepsilon_t+\theta\varepsilon_{t-1}$ 且 $|\theta|<1$，则
$$\varepsilon_t=y_t-\theta y_{t-1}+\theta^2y_{t-2}-\cdots.$$
逆滤波系数几何衰减；在有限二阶矩下，这个级数可按均方意义解释为稳定极限。这正是用观测递归计算 fitted innovations 的理论基础，而不是仅凭形式代数断言逐路径收敛。

“无限 AR 表示”并不意味着重新声称 $y_t$ 是一个有限阶 AR。它描述的是逆滤波器：为了恢复本期创新，可能需要无限长的观测历史。有限样本软件必须为样本前状态作条件化、回推或积分处理，这也是不同似然实现会有差异的原因。
<!-- bilingual-en:start -->
For invertible MA(1), $\varepsilon_t=y_t-\theta y_{t-1}+\theta^2y_{t-2}-\cdots$. With finite second moments, the geometrically decaying inverse series has a mean-square interpretation. This is an inverse filter, not a claim that $y_t$ is a finite-order AR process. In finite samples, pre-sample observations or states remain unknown and must be conditioned on, backcast, or integrated out.
<!-- bilingual-en:end -->

> [!question]- 自检
> 因果性和可逆性分别决定哪个展开方向？
>
> **答案：** 因果性把观测展开为过去创新的 MA($\infty$)；可逆性把创新展开为过去观测的 AR($\infty$) 逆滤波。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=116|课程讲义 pp. 116–119]]：核对 MA(1) 从观测递归恢复创新的级数。
- [Hyndman & Athanasopoulos, FPP3 §9.4](https://otexts.com/fpp3/MA.html)：核对可逆 MA 的 AR($\infty$) 表示。
- [R `stats::arima` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html)：核对有限样本初始化与 CSS/ML 对早期创新的不同处理。
