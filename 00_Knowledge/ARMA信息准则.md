---
aliases:
  - "AIC AICc 与 BIC 只在同一可比似然问题中作相对排序"
  - ARMA AIC and BIC
  - ARIMA information criteria
  - AICc for ARIMA
  - ARMA 模型选择准则
student_os: knowledge-atom
atom_id: TS-ARMA-016
atom_set: arma-modeling
atom_type: selection-boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[模型比较可比性]]"
  - "[[AIC]]"
  - "[[BIC]]"
  - "[[ARMA似然初值处理]]"
related:
  - "[[Box-Jenkins流程]]"
  - "[[滚动起点评估]]"
part_of:
  - "[[ARMA 模型：识别、估计、诊断与预测.canvas]]"
---

# AIC AICc 与 BIC 只在同一可比似然问题中作相对排序
<!-- bilingual-en:start -->
*AIC, AICc, and BIC rank models only within the same comparable likelihood problem*
<!-- bilingual-en:end -->

> [!summary] 原子边界
> AIC、AICc 与 BIC 都用“拟合的 log-likelihood + 参数复杂度惩罚”比较候选。数值较小只表示在这组**可比候选**中准则更偏好该模型；它不证明模型正确，也不替代残差或预测检查。
> <!-- bilingual-en:start -->
> Information criteria trade log-likelihood fit against a parameter penalty. A smaller value is a relative preference within a comparable candidate set, not proof that the selected model is true or diagnostically adequate.
> <!-- bilingual-en:end -->

通用定义与目标分别见 [[AIC]] 和 [[BIC]]。ARMA/ARIMA 的专门问题是有限样本修正、参数计数与有效 likelihood：

$$\mathrm{AICc}=\mathrm{AIC}+\frac{2K(K+1)}{n-K-1}.$$

这里 $K$ 应计入该实现中所有自由估计参数；是否把创新方差、截距/漂移、回归项计入，必须与软件文档和 likelihood 定义一起核对。FPP3 的非季节 ARIMA 约定写作 $K=p+q+k+1$，其中最后的 $1$ 计创新方差，$k$ 表示常数项是否存在。

可比至少要求：使用同一响应数据与有效样本、同一变换/差分对象、同一种 likelihood 定义与常数项规范。AIC 只对真正的 likelihood 有通常解释；把 CSS 的平方和数值与 exact Gaussian likelihood 的 AIC 混排没有统一基准。不同差分阶 $d$ 会改变 likelihood 所用数据，所以 FPP 明确不建议用 AIC 直接选择 $d$。

AICc 在参数相对样本不够少时增加有限样本惩罚；BIC 的惩罚随 $\log T$ 增大。它们回答的目标不同，排序可以不同。无论选哪个，都应先限定合理候选，再诊断残差，并用与决策损失匹配的外样本评估检验用途。
<!-- bilingual-en:start -->
The generic definitions and targets are supplied by [[AIC]] and [[BIC]]. For ARMA/ARIMA, the specialised finite-sample correction is $\mathrm{AICc}=\mathrm{AIC}+2K(K+1)/(n-K-1)$. $K$ must count all freely estimated parameters under the implementation's convention. Comparability requires the same observed target and effective sample, transformation or differencing, likelihood definition, and deterministic-term convention. AIC from CSS should not be mixed with exact-likelihood AIC. Differencing changes the data entering the likelihood, so AIC is not a sound direct selector of $d$.
<!-- bilingual-en:end -->

> [!question]- 自检
> 候选 A 的 AICc 比 B 小 0.4，能否说 A 是“正确模型”？
>
> **答案：** 不能。它只在当前可比候选与准则下略占优；还需检查差值的不确定意义、残差、稳定性与外样本用途。

## 来源与核验

- [Hyndman & Athanasopoulos, FPP3 §9.6](https://otexts.com/fpp3/arima-estimation.html)：核对 AIC/AICc/BIC 公式及不同差分阶不可比。
- [R `stats::arima` documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/arima.html)：明确 `aic` 仅对 maximum-likelihood fits 有效，并说明拟合方法差异。
- [[01_Math/06_时间序列分析/lecture.pdf#page=120|课程讲义 p. 120]]：核对课程中的 fit–parsimony 权衡与 AIC/SBC 口径。
