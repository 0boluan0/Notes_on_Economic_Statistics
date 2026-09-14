---
aliases:
  - "Yule–Walker 方程把平稳 AR 参数与自协方差递推联系起来"
  - Yule-Walker equations
  - Yule–Walker equations
  - 尤尔沃克方程
student_os: knowledge-atom
atom_id: TS-ARMA-013
atom_set: arma-modeling
atom_type: moment-equation
status: source-checked
mastery_state: unassessed
requires:
  - "[[AR(p)模型]]"
  - "[[AR因果根条件]]"
  - "[[宽平稳定义]]"
related:
  - "[[ACF-PACF阶数识别]]"
  - "[[偏自相关函数]]"
  - "[[ARMA似然初值处理]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# Yule–Walker 方程把平稳 AR 参数与自协方差递推联系起来
<!-- bilingual-en:start -->
*Yule–Walker equations link stationary AR parameters to an autocovariance recursion*
<!-- bilingual-en:end -->

> [!summary] 原子方程
> 对零均值平稳 AR($p$)
> $$y_t=\phi_1y_{t-1}+\cdots+\phi_py_{t-p}+\varepsilon_t,$$
> 用 $y_{t-k}$ 同乘并取期望，可得
> $$\gamma(k)=\sum_{i=1}^p\phi_i\gamma(k-i),\qquad k\ge1,$$
> 而 $k=0$ 的方程是 $\gamma(0)=\sum_{i=1}^p\phi_i\gamma(i)+\sigma^2$。这里使用因果平稳 AR 的创新与过去观测正交。
> <!-- bilingual-en:start -->
> Multiplying a causal stationary AR($p$) equation by lagged observations and taking expectations yields a recursion linking its AR coefficients to its autocovariances. Orthogonality of the innovation to past observations is essential; the lag-zero equation additionally contains the innovation variance.
> <!-- bilingual-en:end -->

把前 $p$ 个方程写成 Toeplitz 线性系统；当滞后向量的协方差矩阵正定（因而非奇异）时，才可唯一地由总体 $\gamma(0),\ldots,\gamma(p)$ 求 $\phi$。用样本自协方差替代则得到 Yule–Walker 矩估计，而不是精确 MLE。反过来，给定 $\phi$ 与 $\sigma^2$ 也可递推出理论 ACF。PACF 的 Levinson–Durbin 递推建立在同一 Toeplitz 结构上。

边界在“对象”上：这些是**纯 AR 的二阶矩方程**。一般 ARMA 的正滞后方程会在低阶处含 MA 创新交叉项；Yule–Walker 不是对所有 MA/ARMA 都可直接套用的通用 MLE，更不等同于条件平方和或精确似然。
<!-- bilingual-en:start -->
The first $p$ equations form a Toeplitz system, with a unique coefficient solution when the lag-vector covariance matrix is positive definite. Replacing population autocovariances with sample estimates gives the Yule–Walker moment estimator, not exact MLE; known coefficients instead generate the theoretical ACF. These equations are specifically AR moment relations, not a general MA/ARMA likelihood algorithm.
<!-- bilingual-en:end -->

> [!question]- 自检
> “用 Yule–Walker 估计 MA(1)”为什么不是直接照搬？
>
> **答案：** MA(1) 的参数不满足纯 AR 的同一自协方差递推；它需要利用自己的矩关系或似然/其他估计方法。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=98|课程讲义 pp. 98–99]]：核对 AR(2) 的 Yule–Walker 方程、ACF 递推与例题。
- [MIT OCW 18.S096, Lecture 8](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/1926c83ecd7ea700f7cb63914c6d7c0f_MIT18_S096F13_lecnote8.pdf)：核对一般 AR($p$) 的 Yule–Walker 线性系统。
