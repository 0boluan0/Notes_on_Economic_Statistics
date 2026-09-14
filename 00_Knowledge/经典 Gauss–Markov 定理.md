---
aliases:
  - "经典 Gauss–Markov 定理比较线性无偏估计量"
  - Classical Gauss-Markov theorem for linear unbiased estimators
  - BLUE scope in the classical theorem
  - 经典高斯马尔可夫定理的比较范围
student_os: knowledge-atom
atom_id: ECON-OLS-012
atom_type: proposition
status: source-checked
mastery_state: unassessed
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
  - "[[异方差与自相关.canvas|异方差与自相关]]"
requires:
  - "[[满列秩与OLS唯一性]]"
  - "[[零条件均值无偏性]]"
contrasts_with:
  - "[[BLUE正态性边界]]"
---

# 经典 Gauss–Markov 定理比较线性无偏估计量
<!-- bilingual-en:start -->
*The classical Gauss–Markov theorem compares linear unbiased estimators*
<!-- bilingual-en:end -->

> [!summary] 原子命题
> 这里采用初级计量教材中的经典版本。在 $y=X\beta+u$、$X$ 满列秩、$E(u\mid X)=0$ 且 $\operatorname{Var}(u\mid X)=\sigma^2I$ 时，OLS 是 BLUE：在所有形如 $C(X)y$ 且满足 $C(X)X=I$ 的线性无偏估计量中，$\hat\beta_{OLS}$ 的条件方差最小。更准确地说，任何这类 $\tilde\beta$ 都满足
> $$
> \operatorname{Var}(\tilde\beta\mid X)-
> \operatorname{Var}(\hat\beta_{OLS}\mid X)\succeq0.
> $$
> <!-- bilingual-en:start -->
> This is the classical version used in introductory econometrics. Under $y=X\beta+u$, full column rank of $X$, $E(u\mid X)=0$, and $\operatorname{Var}(u\mid X)=\sigma^2I$, OLS is BLUE: among estimators of the form $C(X)y$ satisfying $C(X)X=I$, it has the smallest conditional variance. More precisely, every such $\tilde\beta$ satisfies $\operatorname{Var}(\tilde\beta\mid X)-\operatorname{Var}(\hat\beta_{OLS}\mid X)\succeq0$.
> <!-- bilingual-en:end -->

## “最佳”的准确范围
<!-- bilingual-en:start -->
*The exact scope of “best”*
<!-- bilingual-en:end -->

“线性”指估计量对观测结果向量 $y$ 线性；“无偏”指在给定 $X$ 下期望为 $\beta$；“最佳”指方差差矩阵半正定，而不是每个样本里误差最小。这个经典 BLUE 表述不比较有偏估计量、非线性估计量或采用额外分布信息的估计量，也不保证模型设定正确、系数具有因果意义或 OLS 对异常值稳健。
<!-- bilingual-en:start -->
“Linear” means linear in the observed outcome vector $y$; “unbiased” means having conditional expectation $\beta$ given $X$; “best” means that the variance-difference matrix is positive semidefinite, not that the error is smallest in every realised sample. This classical BLUE formulation does not compare biased estimators, nonlinear estimators, or estimators using additional distributional information, and it does not guarantee correct specification, causal meaning, or robustness to outliers.
<!-- bilingual-en:end -->

“经典版本比较线性无偏估计量”不应被扩张成“任何 Gauss–Markov 表述都必须保留线性限制”。Hansen（2022）给出了现代版本：在不增加额外条件的情况下，可把同一方差下界扩展到适当定义的全部无偏估计量，并用 BUE 而非 BLUE 表述。这个原子保留经典版本，是因为它对应本课程的初级定理；同时明确它不是现代文献的唯一表述。
<!-- bilingual-en:start -->
The statement that the classical version compares linear unbiased estimators should not be expanded into a claim that every Gauss–Markov formulation must retain the linearity restriction. Hansen (2022) gives a modern formulation that, without additional conditions, extends the same variance lower bound to the appropriately defined class of unbiased estimators and can therefore be stated as BUE rather than BLUE. This atom retains the classical version because it matches the introductory course theorem while making clear that it is not the only modern formulation.
<!-- bilingual-en:end -->

## 假设各自做什么
<!-- bilingual-en:start -->
*What the assumptions do*
<!-- bilingual-en:end -->

满列秩让系数被唯一识别；零条件均值给无偏性；$\sigma^2I$ 同时表示条件同方差和不同误差之间的条件协方差为零，是效率比较的关键。若存在异方差或相关误差，OLS 在外生性下仍可无偏，但通常不再是该类别中最有效率的估计量，常规方差公式也需要修正。
<!-- bilingual-en:start -->
Full column rank identifies a unique coefficient vector; zero conditional mean supplies unbiasedness; $\sigma^2I$ encodes conditional homoskedasticity and zero conditional covariance across errors and is central to the efficiency comparison. With heteroskedasticity or correlated errors, OLS can remain unbiased under exogeneity but is generally no longer efficient within this class, and its conventional variance formula must be changed.
<!-- bilingual-en:end -->

> [!question]- 回忆提示
> 经典 Gauss–Markov 中的 “best” 为什么不能读成“所有方法里 OLS 永远最好”？
> <!-- bilingual-en:start -->
> Why can “best” in the classical Gauss–Markov theorem not be read as “OLS is always best among all methods”?
> <!-- bilingual-en:end -->
>
> **答案：** 它只在给定模型和方差假设下，比较线性且无偏的估计量，并以方差为准则。
> <!-- bilingual-en:start -->
> **Answer:** It compares only linear unbiased estimators under the stated model and variance assumptions, using variance as the criterion.
> <!-- bilingual-en:end -->

## 继续

- [[BLUE正态性边界]]：区分效率定理与精确小样本分布。
- [[协方差失效下的OLS]]：查看 $\operatorname{Var}(u\mid X)\neq\sigma^2I$ 时 OLS 的中心、协方差与效率怎样分开变化。
<!-- bilingual-en:start -->
- [[BLUE正态性边界|Normality is not required for OLS to be BLUE]] separates the efficiency theorem from exact finite-sample distributions.
- [[协方差失效下的OLS|OLS under nonspherical errors]] explains how the estimator's centre, covariance, and efficiency separate when $\operatorname{Var}(u\mid X)\neq\sigma^2I$.
<!-- bilingual-en:end -->

## 来源与核验

- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 3 §§3.3–3.5：核验 MLR.1–MLR.5、OLS 的条件无偏性、同方差方差公式与 Gauss–Markov 定理的比较类别。
- Bruce E. Hansen, [*A Modern Gauss–Markov Theorem*](https://users.ssc.wisc.edu/~behansen/papers/ecnmt_2022.html), *Econometrica* 90 (2022)：核验现代版本可去掉经典 BLUE 表述中的线性估计量限制，以及经典与现代版本不能混称为唯一范围。
- [[02_Economy/01_Econometrics/02_一元线性回归.md#4.2. 高斯-马尔可夫定理内容|本地课程：一元 Gauss–Markov]] 与 [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1.2. 多元线性回归的高斯-马尔可夫定理|本地课程：多元 Gauss–Markov]]：核对本课程采用的 BLUE 表述。
<!-- bilingual-en:start -->
- Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 3 §§3.3–3.5, supports assumptions MLR.1–MLR.5, conditional unbiasedness, the homoskedastic variance formula, and the comparison class in the Gauss–Markov theorem.
- Bruce E. Hansen, [*A Modern Gauss–Markov Theorem*](https://users.ssc.wisc.edu/~behansen/papers/ecnmt_2022.html), *Econometrica* 90 (2022), supports the modern removal of the linear-estimator restriction and the need to distinguish the classical BLUE formulation from newer statements.
- [[02_Economy/01_Econometrics/02_一元线性回归.md#4.2. 高斯-马尔可夫定理内容|The local simple-regression Gauss–Markov section]] and [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1.2. 多元线性回归的高斯-马尔可夫定理|the local multiple-regression Gauss–Markov section]] fix the course's BLUE formulation.
<!-- bilingual-en:end -->
