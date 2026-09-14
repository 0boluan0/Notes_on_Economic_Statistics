---
aliases:
  - "White 异方差检验用解释变量、平方项和交叉项检验平方残差的广泛可预测性"
  - White heteroskedasticity test
  - White test
student_os: knowledge-atom
atom_id: ECON-ERR-006
atom_set: regression-error-covariance
atom_type: diagnostic-test
status: source-checked
mastery_state: unassessed
requires:
  - "[[异方差]]"
  - "[[残差协方差诊断]]"
related:
  - "[[Breusch-Pagan检验]]"
  - "[[异方差稳健协方差]]"
leads_to:
  - "[[误差协方差决策]]"
part_of:
  - "[[异方差与自相关.canvas|异方差与自相关]]"
---

# White 异方差检验用解释变量、平方项和交叉项检验平方残差的广泛可预测性
<!-- bilingual-en:start -->
*White's heteroskedasticity test uses regressors, squares, and interactions to test broad predictability of squared residuals*
<!-- bilingual-en:end -->

> [!summary] 检验怎样构造
> 拟合原均值方程后，把 $\hat u_i^2$ 对原解释变量、它们的平方项和交叉项做辅助回归。若辅助回归中有 $q$ 个非截距项，常见 LM 统计量是
> $$LM=nR^2\ \overset{H_0}{\approx}\ \chi_q^2,$$
> 原假设是这些项不能系统解释误差尺度，即条件方差保持常数。
>
> <!-- bilingual-en:start -->
> After fitting the mean equation, regress squared residuals on the original regressors, their squares, and interactions. The common LM statistic $nR^2$ is compared with a chi-squared reference distribution whose degrees of freedom equal the number of tested non-constant auxiliary terms.
> <!-- bilingual-en:end -->

与[[Breusch-Pagan检验]]相比，White 检验不必先指定一个窄的线性方差方向；代价是辅助回归会迅速膨胀。原模型有许多连续变量时，平方项和两两交互会吃掉自由度、制造共线性并降低有限样本检验质量。软件还可能删除重复或线性依赖列，所以自由度应按实际进入辅助回归的独立非截距列计算，而不是照公式机械计数。
<!-- bilingual-en:start -->
Unlike a targeted Breusch–Pagan test, White's test does not require a narrow linear variance alternative. The price is rapid growth of the auxiliary design. With many regressors, squares and pairwise products consume degrees of freedom and may become collinear; use the rank of the columns actually tested rather than a mechanical term count.
<!-- bilingual-en:end -->

拒绝不能单独证明“真正原因就是异方差”。若原均值方程漏掉平方项，$\hat u_i^2$ 也可能被 $X_i^2$ 解释；所以 White 检验也可暴露一般设定错误。它更不能告诉你应使用 WLS、变换因变量，还是只换协方差口径。检验之后仍要回到残差图、理论机制和研究目标。
<!-- bilingual-en:start -->
Rejection does not uniquely identify changing variance. If the conditional mean omits curvature, squared residuals can also be predicted by squared regressors, so the test may reveal broader misspecification. It does not choose among WLS, outcome transformation, or a covariance correction.
<!-- bilingual-en:end -->

“White 检验”和“White/HC 标准误”是两个不同对象：前者产生一个诊断统计量，后者估计 $\hat\beta$ 的协方差。可以不先做 White 检验就预先报告 HC 推断，也不能因为检验未拒绝就断言经典标准误一定正确。
<!-- bilingual-en:start -->
White's test and White/HC standard errors are different objects. The former is a diagnostic statistic; the latter estimates coefficient covariance. HC inference can be specified without first testing, and non-rejection does not certify classical standard errors.
<!-- bilingual-en:end -->

> [!question]- 自检
> 为什么有 20 个解释变量时，直接加入所有平方项和交叉项的 White 检验可能很差？
>
> **答案：** 辅助回归维度会很快接近样本量，消耗自由度并出现列依赖；“更一般”不等于有限样本中更有检验力。

## 来源与核验

- [[02_Economy/01_Econometrics/07_异方差.md#3.3. White 异方差检验|本地课程：White 检验]]：核对辅助回归、$nR^2$ 与课程记号。
- [White (1980), *A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity*](https://doi.org/10.2307/1912934)：核对检验与 HC 协方差的原始区分。
- [statsmodels `het_white`](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_white.html)：核对平方项/交叉项、常数和自由度实现口径。
