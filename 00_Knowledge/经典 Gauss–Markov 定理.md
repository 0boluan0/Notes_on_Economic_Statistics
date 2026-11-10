---
aliases:
  - Classical Gauss–Markov theorem
  - Gauss-Markov theorem
  - 经典高斯马尔可夫定理
student_os: knowledge-atom
atom_id: ECON-OLS-012
atom_type: proposition
status: source-checked
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
  - "[[异方差与自相关.canvas|异方差与自相关]]"
requires:
  - "[[满列秩与OLS唯一性]]"
  - "[[零条件均值无偏性]]"
contrasts_with:
  - "[[BLUE正态性边界]]"
---

<!-- bilingual-en:start -->
# Classical Gauss–Markov theorem
<!-- bilingual-en:end -->

*经典 Gauss–Markov 定理*

<!-- bilingual-en:start -->
The classical Gauss–Markov theorem explains when ordinary least squares uses noisy observations as precisely as any **linear unbiased** rule can. Its claim is about the spread of estimates across repeated samples. It does not promise the closest answer in every individual sample.
<!-- bilingual-en:end -->

经典 Gauss–Markov 定理说明：在什么条件下，普通最小二乘利用带噪声的观测，能达到所有**线性无偏**规则中最高的估计精度。它比较的是重复抽样时估计值的分散程度，并不保证每一次样本都给出最接近真值的答案。

<!-- bilingual-en:start -->
## Start with two measurements
<!-- bilingual-en:end -->

*从两次测量开始*

<!-- bilingual-en:start -->
Suppose two readings measure the same unknown quantity $\beta$: $Y_1=\beta+u_1$ and $Y_2=\beta+u_2$. Each error has mean zero and finite variance $\sigma^2>0$, and their [[协方差|covariance]] is zero. A rule $T_a=aY_1+(1-a)Y_2$ assigns weight $a$ to the first reading and $1-a$ to the second. The weights add to one, so $E[T_a]=\beta$ for every possible $\beta$: the rule is [[估计量无偏性|unbiased]].
<!-- bilingual-en:end -->

设两次读数测量同一个未知量 $\beta$：$Y_1=\beta+u_1$、$Y_2=\beta+u_2$。两个误差的均值都是零，具有相同的有限方差 $\sigma^2>0$，且[[协方差]]为零。规则 $T_a=aY_1+(1-a)Y_2$ 给第一次读数权重 $a$，给第二次权重 $1-a$。权重相加为一，因此对每个可能的 $\beta$ 都有 $E[T_a]=\beta$，即[[估计量无偏性|无偏]]。

$$
\operatorname{Var}(T_a)=\sigma^2\{a^2+(1-a)^2\}=\frac{\sigma^2}{2}+2\sigma^2\left(a-\frac12\right)^2.
$$

> [!example] Equal weights minimise the variance · 等权重使方差最小
> <!-- bilingual-en:start -->
> The extra square is never negative. Taking $a=1/2$ gives the average $(Y_1+Y_2)/2$ and variance $\sigma^2/2$. Taking $a=3/4$ gives variance $5\sigma^2/8$. The average wins this variance comparison among all fixed weights whose sum is one, even though a different rule may happen to land closer to $\beta$ in a particular sample.
> <!-- bilingual-en:end -->
>
> 额外的平方项不可能为负。取 $a=1/2$ 得到平均数 $(Y_1+Y_2)/2$，方差为 $\sigma^2/2$；取 $a=3/4$ 时方差为 $5\sigma^2/8$。在所有权重和为一的固定加权规则中，平均数的方差最小，但另一个规则仍可能碰巧在某次样本中更接近 $\beta$。

<!-- bilingual-en:start -->
This is an intercept-only regression. The general theorem makes the same comparison when several coefficients determine the mean of the observations.
<!-- bilingual-en:end -->

这就是只有截距项的回归。一般定理把同样的比较推广到由多个系数共同决定观测均值的情况。

<!-- bilingual-en:start -->
## Set up the regression comparison
<!-- bilingual-en:end -->

*建立回归比较*

<!-- bilingual-en:start -->
Write $y=X\beta+u$ over the real numbers. Here $y$ contains $n$ observed outcomes; $X$ is the $n\times k$ design matrix, with one row per observation and $k$ included regression terms; $\beta\in\mathbb R^k$ is the unknown coefficient vector; and $u$ is the vector of random model errors. A constant column, when included, counts among the $k$ terms. The errors $u$ are not the fitted residuals $y-X\hat\beta$.
<!-- bilingual-en:end -->

在实数范围内写 $y=X\beta+u$。其中 $y$ 收集 $n$ 个观测结果；$X$ 是 $n\times k$ 设计矩阵，每行对应一个观测，包含 $k$ 个回归项；$\beta\in\mathbb R^k$ 是未知系数向量；$u$ 是随机模型误差向量。若包含常数列，它也计入这 $k$ 项。模型误差 $u$ 不等于拟合后的残差 $y-X\hat\beta$。

<!-- bilingual-en:start -->
Fix $X$ while comparing rules. Assume its columns are [[线性无关|linearly independent]], so $n\ge k$ and [[满列秩与OLS唯一性|the OLS coefficient vector is unique]]. Assume $E(u\mid X)=0$ and $\operatorname{Cov}(u\mid X)=\sigma^2I_n$, with $0<\sigma^2<\infty$. Here $I_n$ is the identity matrix: every error has the same conditional variance, and distinct errors have zero conditional covariance. [[普通最小二乘|Ordinary least squares]] minimises the sum of squared residuals and gives
<!-- bilingual-en:end -->

比较规则时先固定 $X$。假设各列[[线性无关]]，因而 $n\ge k$ 且[[满列秩与OLS唯一性|OLS 系数向量唯一]]。再假设 $E(u\mid X)=0$、$\operatorname{Cov}(u\mid X)=\sigma^2I_n$，其中 $0<\sigma^2<\infty$。$I_n$ 是单位矩阵：每个误差的条件方差相同，不同误差之间的条件协方差为零。[[普通最小二乘]]最小化残差平方和，得到

$$
\hat\beta=B_0y,\qquad B_0=(X^TX)^{-1}X^T,\qquad \operatorname{Cov}(\hat\beta\mid X)=\sigma^2(X^TX)^{-1}.
$$

<!-- bilingual-en:start -->
A competing **linear** rule has the form $\tilde\beta=Cy$, where the $k\times n$ matrix $C=C(X)$ may depend on the design but not on the observed outcomes $y$ or unknown $\beta$. Its conditional mean is $CX\beta$. Therefore it is conditionally unbiased for **every** $\beta\in\mathbb R^k$ exactly when $CX=I_k$. Being correct on average only at one lucky parameter value is not enough.
<!-- bilingual-en:end -->

作为比较对象的**线性**规则写成 $\tilde\beta=Cy$，其中 $k\times n$ 矩阵 $C=C(X)$ 可以依赖设计矩阵，但不能依赖观测结果 $y$ 或未知的 $\beta$。其条件期望为 $CX\beta$。因此，对**每个** $\beta\in\mathbb R^k$ 都条件无偏，当且仅当 $CX=I_k$；只在某个碰巧的参数值处平均正确并不够。

<!-- bilingual-en:start -->
## What “best” means
<!-- bilingual-en:end -->

## “最佳”的准确范围

> [!important] BLUE is a covariance comparison · BLUE 比较协方差
> <!-- bilingual-en:start -->
> Under the assumptions above, every competing linear unbiased rule satisfies $\operatorname{Cov}(Cy\mid X)-\operatorname{Cov}(B_0y\mid X)\succeq0$. Thus OLS is the **best linear unbiased estimator (BLUE)**. The symbol $\succeq0$ means [[半正定矩阵|positive semidefinite]]: every fixed linear combination of the coefficients has variance at least as large under the competing rule.
> <!-- bilingual-en:end -->
>
> 在上述假设下，任一线性无偏比较规则都满足 $\operatorname{Cov}(Cy\mid X)-\operatorname{Cov}(B_0y\mid X)\succeq0$。因此 OLS 是**最佳线性无偏估计量（BLUE）**。符号 $\succeq0$ 表示[[半正定矩阵|半正定]]：无论固定哪一个系数线性组合，比较规则的估计方差都不会更小。

<!-- bilingual-en:start -->
For a two-coefficient model, a fixed vector $a=(1,-1)^T$ selects the coefficient difference $\beta_1-\beta_2$. The matrix statement implies $\operatorname{Var}(a^TCy\mid X)\ge\operatorname{Var}(a^TB_0y\mid X)$. It does **not** mean that every matrix entry is larger, or that $Cy$ is farther from $\beta$ in every realised sample. The comparison is over linear unbiased rules in this model; it does not rank biased or nonlinear rules, certify a causal interpretation, or guarantee robustness to outliers.
<!-- bilingual-en:end -->

例如，在有两个系数的模型中，固定向量 $a=(1,-1)^T$ 选出系数差 $\beta_1-\beta_2$。矩阵结论就给出 $\operatorname{Var}(a^TCy\mid X)\ge\operatorname{Var}(a^TB_0y\mid X)$。它**不**表示矩阵的每个元素都更大，也不表示每次实现中 $Cy$ 都离 $\beta$ 更远。这里比较的是此模型中的线性无偏规则，不负责给有偏或非线性规则排序，也不认证因果解释或保证对异常值稳健。

<!-- bilingual-en:start -->
## Why the variance difference cannot be negative
<!-- bilingual-en:end -->

*为什么方差差不能为负*

<!-- bilingual-en:start -->
Separate the competing weights into the OLS weights and an addition: $C=B_0+D$. Since both rules are unbiased, $DX=CX-B_0X=0$. Expanding $CC^T$ produces two cross terms, but $B_0D^T=(X^TX)^{-1}X^TD^T=0$ and its transpose is also zero. Hence
<!-- bilingual-en:end -->

把比较规则的权重分成 OLS 权重和一个增量：$C=B_0+D$。因为两个规则都无偏，$DX=CX-B_0X=0$。展开 $CC^T$ 会出现两个交叉项，但 $B_0D^T=(X^TX)^{-1}X^TD^T=0$，它的转置也为零。因此

$$
\operatorname{Cov}(Cy\mid X)-\operatorname{Cov}(B_0y\mid X)=\sigma^2(CC^T-B_0B_0^T)=\sigma^2DD^T.
$$

<!-- bilingual-en:start -->
For any fixed coefficient weights $a$, its quadratic form is $\sigma^2a^TDD^Ta=\sigma^2\sum_i[(D^Ta)_i]^2\ge0$. Each extra term is a square. This is the matrix counterpart of the extra $2\sigma^2(a-1/2)^2$ in the two-reading example.
<!-- bilingual-en:end -->

对任意固定的系数组合权重 $a$，相应二次型为 $\sigma^2a^TDD^Ta=\sigma^2\sum_i[(D^Ta)_i]^2\ge0$。增加的每一项都是平方。这正是两次测量例子中额外项 $2\sigma^2(a-1/2)^2$ 的矩阵版本。

<!-- bilingual-en:start -->
## What the assumptions contribute
<!-- bilingual-en:end -->

## 假设各自做什么

<!-- bilingual-en:start -->
Full column rank makes the coefficient and inverse formula unique. [[零条件均值无偏性|Zero conditional mean]] gives conditional unbiasedness. The covariance condition $\sigma^2I_n$ makes the extra variance a nonnegative sum of squares. Normality and independence of the errors are **not required**: zero covariance suffices for this proof, and [[零协方差不推独立|zero covariance does not generally mean independence]]. If $\sigma^2=0$, the errors vanish almost surely and all the stated linear unbiased rules recover $\beta$ with zero conditional variance; the comparison is then trivial.
<!-- bilingual-en:end -->

满列秩保证系数与逆矩阵公式唯一；[[零条件均值无偏性|零条件均值]]给出条件无偏性；协方差条件 $\sigma^2I_n$ 使额外方差成为非负的平方和。误差**不必**正态，也**不必**独立：此证明只需要零协方差，而[[零协方差不推独立|零协方差一般不等于独立]]。若 $\sigma^2=0$，误差几乎必然为零，上述所有线性无偏规则都以零条件方差恢复 $\beta$，比较退化为平凡情形。

<!-- bilingual-en:start -->
When $X$ is random, the argument applies conditional on almost every realised full-rank design for which these moment conditions hold. An unconditional variance comparison additionally needs the relevant unconditional second moments to exist. Heteroskedasticity or correlated errors removes the guarantee supplied by this theorem; it does not automatically introduce bias or prove OLS inefficient in every design. See [[协方差失效下的OLS|OLS under nonspherical errors]] for the distinction.
<!-- bilingual-en:end -->

若 $X$ 随机，只要这些矩条件成立，上述论证就适用于几乎每个实现出的满列秩设计。要进一步比较无条件方差，还需相应的无条件二阶矩存在。异方差或误差相关会使本定理提供的保证失效，但不会自动引入偏差，也不能证明 OLS 在每种设计中都无效率；这一区分见[[协方差失效下的OLS]]。

> [!note] Modern extensions change the comparison class · 现代推广须说明比较类别
> <!-- bilingual-en:start -->
> Hansen’s modern results can compare beyond explicitly linear rules, but their model-wide unbiasedness requirements matter. His 2024 clarification distinguishes a broad dependence setting where the required unbiasedness already forces linearity from an independent-sampling result requiring unbiasedness over the full specified class, including heteroskedastic models. Independence is stronger than the zero-covariance assumption used above. These results should not be read as removing “linear” from this classical statement while leaving every other qualification unchanged.
> <!-- bilingual-en:end -->
>
> Hansen 的现代结果可以超出显式限定的线性规则，但其中“在整个模型类别上无偏”的要求至关重要。他在 2024 年澄清：广泛允许依赖的一种设定中，无偏要求本身已迫使估计量线性；另一种独立抽样结果则要求在所指定的整个类别中无偏，包括异方差模型。独立比上面使用的零协方差更强。因此不能把这些结果读成：从本卡经典表述中删去“线性”，其他限定全部原样不动。

<!-- bilingual-en:start -->
## Continue with the relevant boundary
<!-- bilingual-en:end -->

## 继续

<!-- bilingual-en:start -->
For exact small-sample inference, open [[BLUE正态性边界|normality and the BLUE boundary]]. For a variance bound based on a specified parametric distribution rather than linearity of the estimator, open [[Cramér–Rao下界|the Cramér–Rao lower bound]]. Both are examples of why [[估计量有效性|efficiency]] always needs a model and comparison class.
<!-- bilingual-en:end -->

精确小样本推断的条件见[[BLUE正态性边界]]；若需要依靠具体参数分布而不是估计量线性来限制方差，见[[Cramér–Rao下界]]。这些区别说明了为什么[[估计量有效性]]总要指明模型与比较类别。

## 来源与核验

<!-- bilingual-en:start -->
- Bruce E. Hansen, [Econometrics, §4.5, Theorem 4.5.1 and proof, printed pp. 50–51 / PDF pp. 57–58](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf#page=57): supports the classical linear-unbiased comparison and positive-semidefinite covariance-difference proof. The proof uses the displayed conditional moments; the text’s broader sampling assumptions are not needed once these moments are assumed directly.
- Hansen, [A Modern Gauss–Markov Theorem (2022), §2–3](https://users.ssc.wisc.edu/~behansen/papers/ecnmt_2022.pdf#page=4), read together with [Reply (2024), §§1–2, pp. 925–927](https://users.ssc.wisc.edu/~behansen/papers/ecnmt_2024.pdf): supports the qualified distinction between classical and modern comparison classes.
- [[02_Economy/01_Econometrics/02_一元线性回归.md#4.2. 高斯-马尔可夫定理内容|The permanent simple-regression course section]] and [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1.2. 多元线性回归的高斯-马尔可夫定理|the multiple-regression section]] retain the course’s BLUE formulation. The two-reading calculation and the conditional-moment proof are expanded explicitly above.
<!-- bilingual-en:end -->

- Hansen《Econometrics》第 4.5 节、定理 4.5.1 及证明（印刷 pp. 50–51 / PDF pp. 57–58）核对经典比较类别与半正定差矩阵证明；本文直接假设所需条件矩，不额外引入教材的更强抽样设定。
- Hansen 2022 年原文与 2024 年答复第 1–2 节共同核对现代结果的模型类别、无偏性要求与独立抽样边界，不能只据原文摘要作无限定推广。
- [[02_Economy/01_Econometrics/02_一元线性回归.md#4.2. 高斯-马尔可夫定理内容|永久一元回归课程记录]]和[[02_Economy/01_Econometrics/03_多元线性回归.md#2.1.2. 多元线性回归的高斯-马尔可夫定理|永久多元回归课程记录]]核对本课程 BLUE 表述。两次测量例子与条件矩推导已在正文直接展开。
