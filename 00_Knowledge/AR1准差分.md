---
aliases:
  - "AR(1) 误差的准差分消去相邻误差相关，而首期处理区分 Prais–Winsten 与 Cochrane–Orcutt"
  - AR(1) quasi-differencing
  - Prais-Winsten transformation
  - Cochrane-Orcutt transformation
student_os: knowledge-atom
atom_id: ECON-ERR-014
atom_set: regression-error-covariance
atom_type: transformation
status: source-checked
mastery_state: unassessed
requires:
  - "[[误差序列相关]]"
  - "[[广义最小二乘]]"
related:
  - "[[可行GLS]]"
  - "[[Durbin-Watson检验]]"
leads_to:
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# AR(1) 误差的准差分消去相邻误差相关，而首期处理区分 Prais–Winsten 与 Cochrane–Orcutt
<!-- bilingual-en:start -->
*Quasi-differencing removes AR(1) error dependence, while treatment of the first period distinguishes Prais–Winsten from Cochrane–Orcutt*
<!-- bilingual-en:end -->

> [!summary] 变换主体
> 若
> $$Y_t=\beta_0+X_t'\beta+u_t,
> \qquad u_t=\rho u_{t-1}+\varepsilon_t,$$
> 则对 $t\ge2$ 做
> $$
> Y_t-\rho Y_{t-1}
> =(1-\rho)\beta_0+(X_t-\rho X_{t-1})'\beta+\varepsilon_t.
> $$
> 正确 AR(1) 模型和已知 $\rho$ 下，变换后的误差是新息 $\varepsilon_t$，相邻相关被消去。
>
> <!-- bilingual-en:start -->
> Under a correctly specified AR(1) error process, subtracting $\rho$ times the previous equation leaves the innovation $\varepsilon_t$. The transformed intercept is $(1-\rho)\beta_0$, so every regressor—including the constant—must be transformed consistently.
> <!-- bilingual-en:end -->

Cochrane–Orcutt 直接用 $t=2,\ldots,T$ 的准差分方程，因而丢掉第一期。Prais–Winsten 在平稳 AR(1) 假设下把第一期乘以 $\sqrt{1-\rho^2}$，保留它并完成精确 GLS 白化。大样本中丢一期可能影响很小；短样本中，两种方法可能明显不同，不能都含糊称作“完整 GLS”。
<!-- bilingual-en:start -->
Cochrane–Orcutt uses only the quasi-differenced equations from period two onward and drops the first observation. Prais–Winsten rescales the first observation by $\sqrt{1-\rho^2}$ under stationary AR(1) errors, retaining it and completing the exact GLS transformation. The distinction can matter in short samples.
<!-- bilingual-en:end -->

现实中 $\rho$ 通常未知。用初始 OLS 残差估计 $\hat\rho$ 再变换，就是[[可行GLS]]；反复用原尺度残差更新 $\hat\rho$ 和 $\hat\beta$ 是迭代版本。变换回归的残差估计的是 $\varepsilon_t$，不能错拿它直接当成原模型 $u_t$ 去更新而不先重建原尺度残差。
<!-- bilingual-en:start -->
With unknown $\rho$, estimating it from initial OLS residuals and substituting $\hat\rho$ produces FGLS. Iteration must reconstruct residuals on the original equation: residuals from the transformed regression estimate innovations, not the original AR(1) errors.
<!-- bilingual-en:end -->

准差分是否有意义取决于 AR(1) 误差设定。若相关来自遗漏趋势、季节或更高阶动态，变换可能掩盖而非解决均值问题；若 $|\rho|$ 接近 1，截距识别和有限样本行为也会敏感。估计后必须检查变换残差，而不是因为算法收敛就宣布问题消失。
<!-- bilingual-en:start -->
Quasi-differencing is useful only insofar as the AR(1) error model is credible. Omitted trend, seasonality, or higher-order dynamics may be obscured rather than solved, and near-unit-root dependence creates additional finite-sample sensitivity. Diagnose transformed residuals after estimation.
<!-- bilingual-en:end -->

> [!question]- 自检
> 对 $t\ge2$ 做准差分后仍把截距写成 $\beta_0$，哪里错了？
>
> **答案：** 常数列也要变换，所以新截距是 $(1-\rho)\beta_0$；估计后需据此还原原模型截距。

## 来源与核验

- [MIT 14.32, *Serial Correlation*, pp. 2–3](https://ocw.mit.edu/courses/14-32-econometrics-spring-2007/79b4f5053aa4b0b555869b2bd8d372b5_rec_4_27.pdf)：核对准差分代数、估计 $\rho$ 与迭代时原尺度残差的要求。
- [Stata, *Time-Series Reference Manual*, pp. 559–560](https://www.stata.com/manuals18/ts.pdf)：核对 Cochrane–Orcutt 丢首期和 Prais–Winsten 保留首期的区别。
- [[02_Economy/01_Econometrics/08_自相关.md#5.2. 科克伦–奥克特 CO 迭代估计|本地课程：CO 迭代]]：核对课程操作顺序与需要修正的首期边界。
