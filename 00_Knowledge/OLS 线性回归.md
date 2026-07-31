---
aliases:
  - "Ordinary Least Squares"
  - "OLS"
  - "线性回归"
status: source-checked
---

# OLS 线性回归
<!-- bilingual-en:start -->
*Ordinary least squares linear regression*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 用一条线或一个线性平面总结结果变量随解释变量怎样变化，并把残差平方和降到最小。
> **具体锚点：** 工资对教育和经验回归中，教育系数比较“经验等其他已含变量相同”时多一年教育对应的条件均值差。
> **核心难点：** OLS 的代数投影总能计算；无偏、一致或因果解释需要额外的数据生成假设。
> **为什么重要：** 后续稳健标准误、IV、面板、DID 和许多机器学习方法都以线性投影为参照。
> **继续：** 先掌握系数解释和正规方程，再读 [[回归系数推断与线性假设检验]]；外生性失败见 [[内生性、识别、IV、2SLS 与 GMM]]。
> <!-- bilingual-en:start -->
> **What it solves:** OLS summarizes how an outcome changes with explanatory variables by fitting a line or linear surface that minimizes the sum of squared residuals.
> **Concrete anchor:** In a regression of wages on education and experience, the education coefficient compares conditional mean wages for people with the same values of the other included variables, including experience, but one additional year of education.
> **Central difficulty:** The OLS projection can always be computed algebraically. Unbiasedness, consistency, and a causal interpretation require additional assumptions about the data-generating process.
> **Why it matters:** Robust standard errors, IV, panel methods, DID, and many machine-learning methods all use the linear projection as a reference point.
> **Continue with:** First master coefficient interpretation and the normal equations, then read [[回归系数推断与线性假设检验|regression coefficient inference and linear hypothesis testing]]. For failures of exogeneity, see [[内生性、识别、IV、2SLS 与 GMM|endogeneity, identification, IV, 2SLS, and GMM]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[02_Economy/01_Econometrics/太白金星v计量.pdf]]：支持课程范围、初级计量顺序与示例；核心结论另与权威教材或原始论文交叉核验。
> - Wooldridge, *Introductory Econometrics: A Modern Approach*：核验线性模型、推断、内生性、面板与时序计量。
> - MIT 18.06SC 本地官方 lecture/recitation transcripts、session summaries、习题与解答：支持线性代数的定义、算法、证明与几何解释。
> <!-- bilingual-en:start -->
> - [[02_Economy/01_Econometrics/太白金星v计量.pdf|Local econometrics course notes]] support the course scope, introductory sequence, and examples; the core results were also cross-checked against an authoritative textbook or original research.
> - Wooldridge, *Introductory Econometrics: A Modern Approach*, supports the treatment of linear models, inference, endogeneity, panel data, and time-series econometrics.
> - Local official MIT 18.06SC lecture and recitation transcripts, session summaries, exercises, and solutions support the linear-algebra definitions, algorithms, proofs, and geometric interpretations.
> <!-- bilingual-en:end -->

## 模型、条件均值与线性投影
<!-- bilingual-en:start -->
*The model, conditional means, and linear projections*
<!-- bilingual-en:end -->

$Y=X\beta+u$ 可作结构模型，也可把 $X\beta$ 理解为对 Y 的最佳线性预测。含截距时误差定义为未被 X 线性解释的部分。若真实条件均值非线性，OLS 仍估计一个线性投影，但系数解释依赖 X 的分布。
<!-- bilingual-en:start -->
$Y=X\beta+u$ may be interpreted as a structural model, or $X\beta$ may be understood as the best linear predictor of $Y$. With an intercept, the error is the part not explained linearly by $X$. If the true conditional mean is nonlinear, OLS still estimates a linear projection, but the interpretation of its coefficients depends on the distribution of $X$.
<!-- bilingual-en:end -->

## 最小二乘与正规方程
<!-- bilingual-en:start -->
*Least squares and the normal equations*
<!-- bilingual-en:end -->

$\hat\beta$ 最小化 $\sum_i\hat u_i^2$。满列秩时 $\hat\beta=(X^TX)^{-1}X^Ty$，正规方程 $X^T\hat u=0$ 表示残差与每个回归变量样本正交；含截距时残差和为零、拟合值均值等于 y 均值。这些是样本代数事实，不证明外生性。
<!-- bilingual-en:start -->
$\hat\beta$ minimizes $\sum_i\hat u_i^2$. With full column rank, $\hat\beta=(X^TX)^{-1}X^Ty$. The normal equations $X^T\hat u=0$ say that the sample residuals are orthogonal to every regressor. With an intercept, the residuals sum to zero and the mean fitted value equals the mean of $y$. These are algebraic facts about the sample; they do not establish exogeneity.
<!-- bilingual-en:end -->

## 一元与多元系数解释
<!-- bilingual-en:start -->
*Interpreting simple and multiple-regression coefficients*
<!-- bilingual-en:end -->

一元斜率是 X 与 Y 的样本协方差除以 X 方差。多元系数可用 Frisch–Waugh–Lovell 理解：先从目标 X 和 Y 中分别剔除其他控制的线性部分，再回归残差。它是“控制其他变量后的线性关联”，不是简单的两变量关系。
<!-- bilingual-en:start -->
The simple-regression slope is the sample covariance between $X$ and $Y$ divided by the variance of $X$. The Frisch–Waugh–Lovell theorem interprets a multiple-regression coefficient by first removing the linear contribution of the other controls from both the focal $X$ and $Y$, then regressing one residual on the other. It is a linear association after controlling for the other variables, not a simple bivariate relationship.
<!-- bilingual-en:end -->

## 外生性、无偏与一致
<!-- bilingual-en:start -->
*Exogeneity, unbiasedness, and consistency*
<!-- bilingual-en:end -->

条件均值零 $E[u\mid X]=0$ 给有限样本条件无偏并排除误差与全部回归变量的系统关系；较弱的正交和大样本条件可给一致性。它不能靠残差与 X 样本正交验证，因为这正是 OLS 构造出来的。
<!-- bilingual-en:start -->
The zero conditional-mean condition $E[u\mid X]=0$ yields finite-sample conditional unbiasedness and rules out systematic relationships between the error and all regressors. Weaker orthogonality and large-sample conditions may be enough for consistency. This condition cannot be verified from the sample orthogonality of residuals and $X$, because OLS constructs that orthogonality mechanically.
<!-- bilingual-en:end -->

## Gauss–Markov 的准确含义
<!-- bilingual-en:start -->
*What the Gauss–Markov theorem actually says*
<!-- bilingual-en:end -->

在线性、外生、满秩、同方差且误差不相关等条件下，OLS 是线性无偏估计量中方差最小的 BLUE。它不说 OLS 在所有估计量中最好，也不要求正态；正态主要用于小样本精确 t/F 分布。
<!-- bilingual-en:start -->
Under linearity, exogeneity, full rank, homoskedasticity, and uncorrelated errors, OLS is BLUE: it has the smallest variance among linear unbiased estimators. The theorem does not say that OLS is best among all possible estimators, nor does it require normality. Normality is mainly used to obtain exact small-sample $t$ and $F$ distributions.
<!-- bilingual-en:end -->

## 函数形式与变换
<!-- bilingual-en:start -->
*Functional forms and transformations*
<!-- bilingual-en:end -->

level–level、log–level、level–log、log–log 系数分别对应单位变化、半弹性或弹性近似。含二次项/交互项时边际效应随变量值变化，不能单独解释一个系数。
<!-- bilingual-en:start -->
Coefficients in level–level, log–level, level–log, and log–log specifications represent unit changes, semielasticities, or elasticity approximations as appropriate. With quadratic or interaction terms, the marginal effect varies with the values of the variables, so one coefficient cannot be interpreted in isolation.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 为什么 OLS 残差与 X 正交不能证明 $E[u\mid X]=0$？
<!-- bilingual-en:start -->
*Why does orthogonality between the OLS residuals and $X$ not prove $E[u\mid X]=0$?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 样本正交是最小二乘正规方程的机械结果；总体结构误差是否外生是关于数据生成过程的假设。
> <!-- bilingual-en:start -->
> Sample orthogonality follows mechanically from the least-squares normal equations. Whether the population structural error is exogenous is an assumption about the data-generating process.
> <!-- bilingual-en:end -->
### 用自己的话解释多元回归中某个系数。
<!-- bilingual-en:start -->
*Explain a coefficient in a multiple regression in your own words.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 在把其他已含回归变量的线性部分从该解释变量和结果中剔除后，二者剩余变化之间的线性斜率。
> <!-- bilingual-en:start -->
> It is the linear slope between the remaining variation in the explanatory variable and the remaining variation in the outcome after removing the linear contribution of the other included regressors from both.
> <!-- bilingual-en:end -->
### Gauss–Markov 没有保证什么？
<!-- bilingual-en:start -->
*What does Gauss–Markov not guarantee?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它只在指定类别和条件下给 BLUE，不保证因果正确、非线性估计量劣于 OLS或模型设定正确。
> <!-- bilingual-en:start -->
> It establishes BLUE only within a specified class and under stated conditions. It does not guarantee a valid causal interpretation, that nonlinear estimators are inferior to OLS, or that the model is correctly specified.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[02_Economy/01_Econometrics/太白金星v计量.pdf]]：支持课程范围、初级计量顺序与示例；核心结论另与权威教材或原始论文交叉核验。
- Wooldridge, *Introductory Econometrics: A Modern Approach*：核验线性模型、推断、内生性、面板与时序计量。
- MIT 18.06SC 本地官方 lecture/recitation transcripts、session summaries、习题与解答：支持线性代数的定义、算法、证明与几何解释。
<!-- bilingual-en:start -->
- [[02_Economy/01_Econometrics/太白金星v计量.pdf|Local econometrics course notes]] support the course scope, introductory sequence, and examples; the core results were also cross-checked against an authoritative textbook or original research.
- Wooldridge, *Introductory Econometrics: A Modern Approach*, was used to verify the linear model, inference, endogeneity, panel data, and time-series econometrics.
- Local official MIT 18.06SC lecture and recitation transcripts, session summaries, exercises, and solutions support the linear-algebra definitions, algorithms, proofs, and geometric interpretations.
<!-- bilingual-en:end -->
