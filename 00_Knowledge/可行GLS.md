---
aliases:
  - "可行 GLS 先估计受限误差协方差再做 GLS，其效率收益依赖协方差模型是否可信"
  - Feasible generalized least squares
  - FGLS
student_os: knowledge-atom
atom_id: ECON-ERR-013
atom_set: regression-error-covariance
atom_type: estimator
status: source-checked
mastery_state: unassessed
requires:
  - "[[广义最小二乘]]"
related:
  - "[[异方差稳健协方差]]"
  - "[[HAC协方差]]"
leads_to:
  - "[[AR1准差分]]"
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# 可行 GLS 先估计受限误差协方差再做 GLS，其效率收益依赖协方差模型是否可信
<!-- bilingual-en:start -->
*Feasible GLS estimates a restricted error covariance before applying GLS, so its efficiency gain depends on the credibility of that covariance model*
<!-- bilingual-en:end -->

> [!summary] 为什么叫“可行”
> 理论 GLS 需要已知 $\Omega$。FGLS 先用一个初始一致估计量（常为 OLS）取得残差，据此估计低维参数 $\hat\theta$ 和 $\hat\Omega(\hat\theta)$，再计算
> $$
> \hat\beta_{FGLS}
> =[X'\hat\Omega^{-1}X]^{-1}X'\hat\Omega^{-1}Y.
> $$
> 例如，异方差 FGLS 可先拟合方差函数；AR(1) FGLS 可先从相邻残差估计 $\rho$。
>
> <!-- bilingual-en:start -->
> Theoretical GLS needs known $\Omega$. FGLS obtains residuals from an initial consistent estimator, estimates a low-dimensional covariance model, and substitutes the resulting $\hat\Omega$ into GLS.
> <!-- bilingual-en:end -->

这个方法依赖两层模型：条件均值要足以让初始估计一致，协方差限制也要足以逼近真实 $\Omega$。若方差函数或 AR 阶数错设，FGLS 权重会错误，有限样本甚至可能比 OLS 更不稳定。即使点估计在某些条件下仍一致，软件把 $\hat\Omega$ 当作已知所给的朴素标准误也可能忽略第一阶段估计带来的不确定性。
<!-- bilingual-en:start -->
FGLS relies on both a correct-enough mean model for the initial estimator and a credible restricted covariance model. Misspecified weights can make finite-sample performance worse than OLS, and naive second-stage standard errors may ignore uncertainty from estimating $\Omega$.
<!-- bilingual-en:end -->

所以 FGLS 不是“发现异方差/自相关后的默认升级”。若科学目标只是给稳定的 OLS 目标做大样本推断，[[异方差稳健协方差]]或[[HAC协方差]]少依赖一个完整协方差模型；若目标是提高效率、预测，或协方差动态本身有科学意义，FGLS 才更有吸引力。实际报告应给出协方差设定、权重范围、变换后残差诊断和对替代设定的敏感性。
<!-- bilingual-en:start -->
FGLS is not the automatic response to a rejected diagnostic. HC or HAC requires less complete covariance modelling when the aim is inference about a stable OLS target. FGLS is more attractive when efficiency, prediction, or the covariance dynamics themselves matter. Report the covariance specification, weight range, post-transformation diagnostics, and sensitivity.
<!-- bilingual-en:end -->

“迭代到收敛”只解决给定算法的固定点，不证明协方差模型正确。Cochrane–Orcutt 迭代 300 次终于收敛，也可能只是精确求出了一个不合适 AR(1) 假设下的答案。
<!-- bilingual-en:start -->
Iteration to convergence solves an algorithmic fixed point; it does not validate the covariance model. A slowly converged Cochrane–Orcutt estimate can still be the precise answer to an inappropriate AR(1) assumption.
<!-- bilingual-en:end -->

> [!question]- 自检
> FGLS 比 OLS 使用了更多结构，为什么不能保证更好？
>
> **答案：** 额外结构只有在大致正确时才带来效率；错设的 $\hat\Omega$ 会产生错误权重，还会加入第一阶段估计噪声。

## 来源与核验

- [MIT 14.32, *More on FGLS and How to Test for Heteroskedasticity*](https://ocw.mit.edu/courses/14-32-econometrics-spring-2007/c68f54a5cac9d51f45f5916c123a223f_rec_4_20.pdf)：核对残差—方差方程—变换回归的三步结构及方差模型风险。
- [MIT 14.32, *Serial Correlation*](https://ocw.mit.edu/courses/14-32-econometrics-spring-2007/79b4f5053aa4b0b555869b2bd8d372b5_rec_4_27.pdf)：核对以 OLS 残差估计 $\rho$ 和迭代 FGLS。
- [[02_Economy/01_Econometrics/07_异方差.md#4.2. 可行广义最小二乘 FGLS|本地课程：异方差 FGLS]]与[[02_Economy/01_Econometrics/08_自相关.md#5.2. 科克伦–奥克特 CO 迭代估计|本地课程：AR(1) FGLS]]：核对课程操作顺序。
