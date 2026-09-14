---
aliases:
  - "Breusch–Pagan 检验用指定方差解释变量检验条件方差是否保持常数"
  - Breusch-Pagan test
  - BP heteroskedasticity test
student_os: knowledge-atom
atom_id: ECON-ERR-005
atom_set: regression-error-covariance
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[异方差]]"
  - "[[残差协方差诊断]]"
related:
  - "[[White异方差检验]]"
leads_to:
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# Breusch–Pagan 检验用指定方差解释变量检验条件方差是否保持常数
<!-- bilingual-en:start -->
*The Breusch–Pagan test uses specified variance covariates to test whether conditional error variance is constant*
<!-- bilingual-en:end -->

> [!summary] 检验对象
> 先拟合均值方程并取得残差 $\hat u_i$，再用平方残差对预先选择的方差解释变量 $z_i$ 做辅助回归：
> $$\hat u_i^2=\alpha_0+z_i'\alpha+v_i.$$
> 同方差原假设对应 $H_0:\alpha=0$。常见 LM 形式使用 $nR^2$，在相应正则条件下与自由度为 $\dim(z_i)$ 的卡方分布比较。
>
> <!-- bilingual-en:start -->
> Fit the mean equation, regress squared residuals on pre-specified variance covariates, and test whether their slopes are jointly zero. A common LM version uses $nR^2$ with degrees of freedom equal to the number of tested non-constant terms under its regularity conditions.
> <!-- bilingual-en:end -->

BP 的优势正是它的针对性：如果经济机制认为工资误差的尺度随教育或经验线性变化，就把这些变量放进 $z_i$。拒绝说明平方残差与这组变量存在系统关系；未拒绝只表示没有发现**所指定方向**的方差变化，不等于证明所有形式的同方差。
<!-- bilingual-en:start -->
The test is deliberately targeted. If theory suggests that wage-error dispersion changes with education or experience, use those variables in the variance equation. Rejection finds systematic dependence along those directions; non-rejection does not prove homoskedasticity against every alternative.
<!-- bilingual-en:end -->

原始 Breusch–Pagan 版本使用更强的分布条件；常用 Koenker 学生化版本放松正态性要求，但仍依赖观测独立等条件。软件实现还要求辅助回归含常数、设计满秩才能正确计算 $R^2$ 和自由度；中小样本中 LM 卡方近似可能过度拒绝，软件提供的 F 版本可作为敏感性核对。
<!-- bilingual-en:start -->
The original version uses stronger distributional assumptions. The common Koenker studentized version relaxes normality but still relies on conditions such as independent observations. Software implementations also need an intercept and full-rank auxiliary design for correct degrees of freedom; in modest samples the F version can be a useful sensitivity check because the LM approximation may over-reject.
<!-- bilingual-en:end -->

检验不是补救方法，也不解释经济来源。即使不拒绝，只要研究设计已允许未知异方差，大样本推断仍可预先采用合适的 HC 口径；即使拒绝，也要先判断均值函数是否错设，再在[[异方差稳健协方差]]与[[加权最小二乘]]等不同目标之间选择。
<!-- bilingual-en:start -->
The test is neither a remedy nor a causal explanation. A design may justify HC inference even without rejection, while rejection still requires checking the mean specification before choosing between robust inference and variance modelling.
<!-- bilingual-en:end -->

> [!question]- 自检
> BP 检验没有拒绝“方差不随收入线性变化”。能否据此宣布误差同方差？
>
> **答案：** 不能。它没有发现指定的线性方向；方差仍可能随收入非线性变化，或随没有进入辅助回归的变量变化。

## 来源与核验

- [[02_Economy/01_Econometrics/07_异方差.md#3.2. Breusch–Pagan 拉格朗日乘数检验|本地课程：LM 检验步骤]]：核对残差平方辅助回归和 $nR^2$ 的课程口径。
- [Breusch and Pagan (1979), *A Simple Test for Heteroscedasticity and Random Coefficient Variation*](https://doi.org/10.2307/1911963)：核对 LM 检验的原始来源。
- [statsmodels `het_breuschpagan`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html)：核对 Koenker 默认版本、常数/满秩要求和小样本 F 版本边界。
