---
aliases:
  - "ARMA(p,q) 结合观测递归与有限创新记忆"
  - ARMA
  - ARMA Models
  - ARMA model
  - 自回归移动平均模型
student_os: knowledge-atom
atom_id: TS-ARMA-005
atom_set: arma-modeling
atom_type: definition
status: source-checked
mastery_state: unassessed
requires:
  - "[[AR(p)模型]]"
  - "[[MA(q)模型]]"
  - "[[创新]]"
related:
  - "[[ARMA公共因子]]"
  - "[[Box-Jenkins流程]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# ARMA(p,q) 结合观测递归与有限创新记忆
<!-- bilingual-en:start -->
*An ARMA(p,q) combines recursion through observations with finite innovation memory*
<!-- bilingual-en:end -->

> [!summary] 它是什么
> ARMA($p,q$) 的中心化形式是
> $$\phi(B)(y_t-\mu)=\theta(B)\varepsilon_t,$$
> 即
> $$y_t-\mu=\sum_{i=1}^p\phi_i(y_{t-i}-\mu)+\varepsilon_t+
> \sum_{j=1}^q\theta_j\varepsilon_{t-j}.$$
> AR 部分把影响经观测递归传播，MA 部分允许最近的创新直接改变当前值。
> <!-- bilingual-en:start -->
> ARMA($p,q$) combines an AR recursion through the previous $p$ observations with a finite MA filter of the current and previous $q$ innovations.
> <!-- bilingual-en:end -->

$q=0$ 得 AR($p$)，$p=0$ 得 MA($q$)。ARMA 用少量参数描述一个宽平稳序列的二阶线性动态；若要把递推称为完整条件均值，还需对创新的信息集给出 MDS、独立性、joint Gaussian 等额外条件。它并不自动描述趋势、条件方差、结构突变或非线性依赖。实践中还必须另外说明：AR 多项式是否给出因果解、MA 多项式是否采用可逆规范，以及两者是否互素。

创新 $\varepsilon_t$ 是相对于模型信息集的新线性冲击，不是直接观测到的普通误差列。拟合后得到的 residual 只是创新的估计，因此 “把残差放回模型” 与 “知道真实创新” 不是同一件事。
<!-- bilingual-en:start -->
ARMA is a finite-parameter model for second-order linear dynamics in a covariance-stationary series. Interpreting its recursion as the full conditional mean needs an MDS, independence, joint Gaussianity, or another sufficient innovation assumption. ARMA does not by itself model trend, time-varying conditional variance, breaks, or nonlinear dependence. Causality, invertibility, and absence of common polynomial factors must be checked separately. Fitted residuals estimate the latent innovations; they are not the innovations observed without error.
<!-- bilingual-en:end -->

> [!question]- 自检
> 残差 ACF 不显著，是否说明一个 ARMA 已同时解释了收益均值与波动率聚集？
>
> **答案：** 不说明。它最多支持这些滞后上没有检测到剩余线性均值相关；绝对或平方残差仍可能有持续相关。

## 来源与核验

- [[01_Math/06_时间序列分析/lecture.pdf#page=73|课程讲义 pp. 73–74]]：核对 ARMA($p,q$) 方程、$\beta_0=1$ 规范与创新角色。
- [MIT OCW 18.S096, Lecture 8](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/1926c83ecd7ea700f7cb63914c6d7c0f_MIT18_S096F13_lecnote8.pdf)：核对 ARMA 多项式表示与线性过程边界。
