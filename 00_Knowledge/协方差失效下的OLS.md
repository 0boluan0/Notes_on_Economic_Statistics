---
aliases:
  - "异方差或误差相关不会单独决定 OLS 点估计的偏误方向，但会使经典效率与协方差公式失效"
  - OLS under nonspherical errors
  - OLS with heteroskedastic or correlated errors
student_os: knowledge-atom
atom_id: ECON-ERR-003
atom_set: regression-error-covariance
atom_type: boundary
status: source-checked
mastery_state: unassessed
requires:
  - "[[异方差]]"
  - "[[误差序列相关]]"
  - "[[零条件均值无偏性]]"
related:
  - "[[经典 Gauss–Markov 定理]]"
  - "[[标准误口径匹配]]"
leads_to:
  - "[[异方差稳健协方差]]"
  - "[[HAC协方差]]"
  - "[[广义最小二乘]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# 异方差或误差相关不会单独决定 OLS 点估计的偏误方向，但会使经典效率与协方差公式失效
<!-- bilingual-en:start -->
*Heteroskedasticity or correlated errors alone do not determine the bias of OLS point estimates, but they invalidate classical efficiency and covariance formulas*
<!-- bilingual-en:end -->

> [!summary] 把“系数”和“误差条”分开
> 若 $X$ 满列秩且全样本条件均值满足 $E(u\mid X)=0$，则
> $$E(\hat\beta_{OLS}\mid X)=\beta$$
> 并不需要 $\operatorname{Var}(u\mid X)=\sigma^2I$。因此异方差或误差相关本身不会自动使 OLS 有偏；它们首先改变的是 $\hat\beta$ 的抽样协方差与效率。
>
> <!-- bilingual-en:start -->
> With full-rank $X$ and $E(u\mid X)=0$, OLS remains conditionally unbiased without requiring spherical errors. Heteroskedasticity or correlation alone therefore does not automatically bias OLS; it first changes sampling covariance and efficiency.
> <!-- bilingual-en:end -->

一般协方差为
$$
\operatorname{Var}(\hat\beta_{OLS}\mid X)
=(X'X)^{-1}X'\Omega X(X'X)^{-1},
$$
其中 $\Omega=\operatorname{Var}(u\mid X)$。只有 $\Omega=\sigma^2I$ 时，它才化简为经典公式 $\sigma^2(X'X)^{-1}$，并得到通常的 BLUE 结论。继续套用经典标准误，就会错算 $t$、$F$ 和置信区间。
<!-- bilingual-en:start -->
The general covariance is the sandwich expression above. It reduces to the classical formula only under $\Omega=\sigma^2I$, the spherical-error case that supports the usual BLUE result. Applying classical standard errors outside that case miscalibrates tests and intervals.
<!-- bilingual-en:end -->

误差相关下，常规标准误是偏大还是偏小**没有固定方向**。方向取决于解释变量的时间路径和 $\Omega$ 的形状；正序列相关在许多平滑解释变量的例子中会让经典标准误偏小，但这是一种常见情形，不是定理。也不能从一次样本中 $t$ 值升降，倒推出误差相关的方向。
<!-- bilingual-en:start -->
There is no universal direction for the error in conventional standard errors under serial dependence. It depends on the regressor path and the shape of $\Omega$. Positive serial correlation often causes understatement with smooth regressors, but this is a common case rather than a theorem.
<!-- bilingual-en:end -->

最后，点估计仍可信的结论有条件。若模型含滞后因变量，而误差又序列相关，滞后结果往往与当前误差相关，$E(u\mid X)=0$ 随之失效；遗漏趋势、季节或动态项也可能同时破坏均值设定。此时换 HC/HAC 标准误只是在给错误中心加上新误差条，不能修复系数含义。
<!-- bilingual-en:start -->
The point-estimate result is conditional. With a lagged dependent variable and serially correlated errors, the lagged outcome is generally correlated with the current error, so exogeneity fails. Omitted trends, seasonality, or dynamics can do the same. HC or HAC standard errors cannot repair a misspecified centre.
<!-- bilingual-en:end -->

> [!question]- 自检
> OLS 系数在改用 HAC 后没有变化，是否证明原来的回归完全正确？
>
> **答案：** 不能。HAC 本来就只改协方差估计；均值设定、外生性和目标参数仍要另行检查。

## 来源与核验

- [[零条件均值无偏性]]与[[经典 Gauss–Markov 定理]]：核对无偏性条件、球形误差和 BLUE 的逻辑分工。
- [MIT 14.32, *Serial Correlation*](https://ocw.mit.edu/courses/14-32-econometrics-spring-2007/79b4f5053aa4b0b555869b2bd8d372b5_rec_4_27.pdf)：核对序列相关下 OLS 中心、效率、HAC 与 GLS 两条路线。
- [Penn State STAT 501, Topic 2](https://online.stat.psu.edu/stat501/Topic2TSandAutocorr)：核对标准误偏小只是常见/平均倾向而非固定方向。
