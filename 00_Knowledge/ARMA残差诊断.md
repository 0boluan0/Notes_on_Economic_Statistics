---
aliases:
  - "残差近似白噪声只说明所检滞后未发现剩余线性相关"
  - ARMA residual diagnostics
  - Residual white noise boundary
  - Innovation residual diagnostics
student_os: knowledge-atom
atom_id: TS-ARMA-018
atom_set: arma-modeling
atom_type: diagnostic-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[白噪声二阶定义]]"
  - "[[Ljung-Box检验]]"
related:
  - "[[ACF信息边界]]"
  - "[[波动率聚集]]"
  - "[[条件异方差：ARCH 与 GARCH.canvas]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# 残差近似白噪声只说明所检滞后未发现剩余线性相关
<!-- bilingual-en:start -->
*Approximately white residuals say only that the checked lags reveal no remaining linear correlation*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> 残差均值接近零、ACF 无明显结构且 portmanteau 检验未拒绝，是 ARMA 线性均值模型的必要诊断：在所选滞后、样本与检验力下没有检测到剩余线性自相关。它不证明这些相关恰为零，更不证明残差 i.i.d.、Gaussian、同方差或模型是真实 DGP。
> <!-- bilingual-en:start -->
> Near-zero-mean residuals, an unstructured sample ACF, and a portmanteau non-rejection say only that remaining linear autocorrelation was not detected at the checked lags and available power. They do not establish exact zero correlations, i.i.d. errors, Gaussianity, homoscedasticity, or model truth.
> <!-- bilingual-en:end -->

原因有三层：第一，ACF 与 Ljung–Box 只针对二阶线性相关；非线性条件均值或平方相关可以完全漏过。第二，未拒绝受样本量与检验滞后影响，可能只是检验力不足。第三，不同模型都可能产生近似不相关残差，诊断合格不能替代模型之间的预测比较。

因此还要分别看：残差路径与均值、绝对/平方残差 ACF 或 ARCH 检验、分布尾部/QQ 图、异常点、结构突变和外样本误差。正态与同方差主要影响区间构造与风险解释；它们不是“线性均值预测没有可利用自相关”的同义条件。
<!-- bilingual-en:start -->
ACF diagnostics can miss nonlinear dependence, a non-rejection may reflect low power, and several different models can all produce uncorrelated residuals. Inspect squared or absolute residuals, distributional shape, outliers, breaks, and out-of-sample performance separately.
<!-- bilingual-en:end -->

> [!question]- 自检
> 原残差的 Ljung–Box p 值很大，但平方残差 ACF 持续显著。可以宣布模型完全合格吗？
>
> **答案：** 不可以。线性均值结构可能尚可，但条件方差仍有可预测性，需要 ARCH/GARCH 等波动模型或相应区间修正。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §5.4](https://otexts.com/fpp3/diagnostics.html)：明确零均值与不相关是必要诊断，同方差/正态是有用但非必要，且通过诊断仍可能有更好模型。
- [[01_Math/06_时间序列分析/lecture.pdf#page=122|课程讲义 pp. 122–123]]：核对 Box–Jenkins residual white-noise check。
- [[ACF信息边界]]：复用 ACF 对高阶与非线性结构的已核验边界。
