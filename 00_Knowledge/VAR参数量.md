---
aliases:
  - "VAR 的动态系数参数量随 K²p 增长而确定项与外生项另计"
  - VAR parameter count
  - VAR 参数量
student_os: knowledge-atom
atom_id: TS-VAR-002
atom_set: vector-autoregression
atom_type: derivation
status: source-checked
mastery_state: unassessed
requires:
  - "[[VAR(p)模型]]"
related:
  - "[[VAR规格选择]]"
  - "[[SVAR识别条件]]"
part_of:
  - "[[VAR、脉冲响应与 Granger 因果.canvas]]"
---

# VAR 的动态系数参数量随 K²p 增长而确定项与外生项另计
<!-- bilingual-en:start -->
*A VAR has K-squared-p dynamic coefficients, with deterministic and exogenous terms counted separately*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 一个 $K$ 变量 VAR($p$) 有 $K^2p$ 个滞后动态系数；截距、趋势、季节项和外生回归量会另外增加参数，创新协方差矩阵也不属于这 $K^2p$ 个系数。

每条方程包含每个系统变量的 $p$ 个滞后，所以一条方程有 $Kp$ 个动态斜率。系统共有 $K$ 条方程，因此
$$
\underbrace{K}_{\text{方程数}}
\times
\underbrace{Kp}_{\text{每条方程的动态斜率}}
=K^2p.
$$
例如，三变量 VAR(4) 单是滞后矩阵就有 $3^2\times4=36$ 个系数。

若每条方程还共享 $d$ 个确定或外生回归量，则这些均值方程参数再增加 $Kd$ 个；一个截距对应 $d=1$，即增加 $K$ 个。若用高斯似然估计，还要估计对称创新协方差矩阵 $\Sigma_u$ 的 $K(K+1)/2$ 个自由元素。因而“VAR 有多少参数”必须先说清是在数动态系数、全部条件均值参数，还是连协方差参数一起数。

这一增长速度解释了小样本 VAR 容易过度参数化：增加一个变量会同时扩展每条方程，增加一个滞后也会新增整块 $K\times K$ 系数。它也说明不能把简约型可估参数数目直接与某个结构参数化作机械比较；结构模型的计数还取决于冲击方差归一化、矩阵方向和所施加限制。

> [!question]- 自检
> 一个四变量 VAR(2) 带截距，条件均值部分共有多少个系数？
>
> **答案：** 动态系数为 $4^2\times2=32$ 个，四条截距再加 4 个，共 36 个；创新协方差参数若要计入，还另有 $4(4+1)/2=10$ 个。

## 来源与核验

- [Lütkepohl (2005), *New Introduction to Multiple Time Series Analysis*](https://doi.org/10.1007/978-3-540-27752-1)，第 3–4 章：核对 VAR 估计、阶数选择与参数维度。
- [[01_Math/06_时间序列分析/lecture.pdf]]：对照课程对二变量 VAR 参数数目的讨论，并分离均值方程与协方差参数。
