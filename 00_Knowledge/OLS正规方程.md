---
aliases:
- OLS normal equations
- Sample residual orthogonality
- OLS 正规方程
student_os: knowledge-atom
atom_id: ECON-OLS-005
atom_type: proposition
status: source-checked
part_of:
- '[[OLS 线性回归.canvas|OLS 线性回归]]'
requires:
- '[[普通最小二乘]]'
related:
- '[[最小二乘正规方程]]'
contrasts_with:
- '[[样本正交与总体外生性]]'
implies:
- '[[一元 OLS 斜率]]'
---

<!-- bilingual-en:start -->
# OLS normal equations
<!-- bilingual-en:end -->

*OLS 正规方程*

<!-- bilingual-en:start -->
At an [[普通最小二乘|ordinary least-squares]] fit, moving a coefficient slightly in either direction cannot reduce the sum of squared residuals. The **normal equations** express this balance: the residuals have zero dot product with every predictor column actually included in the fit. They describe a property enforced by the fitting rule, rather than evidence that the population errors are unrelated to the predictors.
<!-- bilingual-en:end -->

在[[普通最小二乘]]拟合处，把某个系数稍微往任一方向调整，都不能继续降低残差平方和。**正规方程**表达这种平衡：残差与实际纳入拟合的每一个解释变量列的点积都为零。它描述拟合规则强制产生的性质，不是在提供“总体误差与解释变量无关”的证据。

> [!example] The intercept and slope impose two balances / 截距与斜率带来两条平衡式
> <!-- bilingual-en:start -->
> Fit $a+cx$ to $x=(0,1,2)$ and $y=(1,2,2)$. Adjusting $a$ changes all fitted values equally; adjusting $c$ changes them in proportion to $(0,1,2)$. The two balance equations are
> <!-- bilingual-en:end -->
>
> 对 $x=(0,1,2)$、$y=(1,2,2)$ 拟合 $a+cx$。调整 $a$ 会同量改变全部拟合值；调整 $c$ 则按 $(0,1,2)$ 的比例改变它们。两条平衡式为
>
> $$\begin{aligned}
> (1-a)+(2-a-c)+(2-a-2c)&=0,\\
> 0(1-a)+1(2-a-c)+2(2-a-2c)&=0.
> \end{aligned}$$
>
> <!-- bilingual-en:start -->
> Simplifying gives $3a+3c=5$ and $3a+5c=6$. Subtract the first from the second: $2c=1$, so $c=1/2$ and $a=7/6$. The residuals $(-1/6,1/3,-1/6)$ sum to zero, and their predictor-weighted sum is $0(-1/6)+1(1/3)+2(-1/6)=0$. Individual residuals need not vanish.
> <!-- bilingual-en:end -->
>
> 整理得到 $3a+3c=5$、$3a+5c=6$。两式相减得 $2c=1$，所以 $c=1/2$、$a=7/6$。残差 $(-1/6,1/3,-1/6)$ 的和为零，按解释变量取值加权后的和也为零；但每个残差不必为零。

<!-- bilingual-en:start -->
## The same balances in matrix form
<!-- bilingual-en:end -->

*用矩阵把这些平衡式放在一起*

<!-- bilingual-en:start -->
Take real observed data $y\in\mathbb R^n$ and a fixed $n\times k$ design matrix $X$, with $n,k\ge1$. Here $n$ counts rows and $k$ counts all fitted coefficients, including an intercept when present. Row $i$ of $X$ is $x_i'$; column $j$ contains the values $x_{1j},\ldots,x_{nj}$. A prime denotes [[矩阵转置|transpose]]. Coefficients range freely over $\mathbb R^k$, and the objective is $Q(b)=\sum_i(y_i-x_i'b)^2$.

For any minimizer $\hat\beta$, define the [[回归残差|residual vector]] $\hat u=y-X\hat\beta$. The normal equations are
<!-- bilingual-en:end -->

取实观测结果 $y\in\mathbb R^n$ 与固定的 $n\times k$ 设计矩阵 $X$，$n,k\ge1$。$n$ 数观测行，$k$ 数全部拟合系数，有截距时也计入。$X$ 的第 $i$ 行为 $x_i'$，第 $j$ 列包含 $x_{1j},\ldots,x_{nj}$；撇号表示[[矩阵转置|转置]]。系数可在整个 $\mathbb R^k$ 中自由取值，目标为 $Q(b)=\sum_i(y_i-x_i'b)^2$。

对任一最小化解 $\hat\beta$，定义[[回归残差|残差向量]] $\hat u=y-X\hat\beta$。正规方程为

$$\boxed{X'\hat u=0,\qquad\text{equivalently }X'X\hat\beta=X'y.}$$

<!-- bilingual-en:start -->
The zero on the right is a $k$-component vector. Its $j$th equation is $\sum_i x_{ij}\hat u_i=0$. This sum is the [[向量点积|dot product]] of predictor column $j$ and the residual vector; zero dot product is [[正交|orthogonality]]. The columns can be linearly dependent: these equations characterize minimizers even then, though they need not determine a unique coefficient vector. “Normal” here refers to perpendicularity, not to a normal probability distribution.
<!-- bilingual-en:end -->

右侧的零是有 $k$ 个分量的向量；第 $j$ 条式子是 $\sum_i x_{ij}\hat u_i=0$，即解释变量第 $j$ 列与残差向量的[[向量点积]]。点积为零称为[[正交]]。即使各列线性相关，这些方程仍刻画最小化解，只是未必决定唯一系数向量。这里的“normal”指垂直关系，不是正态概率分布。

<!-- bilingual-en:start -->
## Why no further reduction is possible
<!-- bilingual-en:end -->

## 自然解释

<!-- bilingual-en:start -->
Let $X_j$ denote column $j$ and change only its coefficient by a small number $\delta$. The residual vector changes from $\hat u$ to $\hat u-\delta X_j$, so its squared total changes by
<!-- bilingual-en:end -->

用 $X_j$ 表示第 $j$ 列，只把它的系数改变一个小量 $\delta$。残差从 $\hat u$ 变成 $\hat u-\delta X_j$，平方和的改变量为

$$-2\delta X_j'\hat u+\delta^2 X_j'X_j.$$

<!-- bilingual-en:start -->
If $X_j'\hat u$ were nonzero, choose $\delta$ with the same sign and sufficiently small magnitude. The negative first term would outweigh the squared term, contradicting a minimum. Hence each dot product must be zero. Equivalently, differentiating $Q$ with respect to coefficient $b_j$ gives $-2\sum_i x_{ij}(y_i-x_i'b)$, which must vanish.

For this unconstrained quadratic objective the conditions are also sufficient. If $X'\hat u=0$, then for any coefficient change $d\in\mathbb R^k$,
<!-- bilingual-en:end -->

若 $X_j'\hat u$ 不为零，就选与它同号、绝对值足够小的 $\delta$；负的一次项会压过平方项，使目标继续降低，与最小值矛盾。因此每个点积都必须为零。等价地，对 $b_j$ 求导得到 $-2\sum_i x_{ij}(y_i-x_i'b)$，最优处必须为零。

对这个无约束的二次目标，条件还足以保证最小值。若 $X'\hat u=0$，对任意系数改变量 $d\in\mathbb R^k$，

$$Q(\hat\beta+d)=Q(\hat\beta)+\lVert Xd\rVert_2^2\ge Q(\hat\beta).$$

<!-- bilingual-en:start -->
The cross term vanishes because $\hat u'Xd=0$, and a sum of squares cannot be negative. Thus the equations certify a **global** minimum, not merely a stationary point. The general geometric statement is [[最小二乘正规方程|the least-squares normal-equation equivalence]]. Here its statistical use is checking the actual fitted sample and predictor columns.
<!-- bilingual-en:end -->

交叉项因 $\hat u'Xd=0$ 而消失，平方和又不可能为负。因此方程验证的是**全局**最小值，而不仅是驻点。一般几何表述见[[最小二乘正规方程]]；在回归中，它用来检查本次实际使用的样本与解释变量列。

<!-- bilingual-en:start -->
## Sample consequences and their limits
<!-- bilingual-en:end -->

## 使用条件与直接推论

<!-- bilingual-en:start -->
If the design contains an intercept column of ones, its equation gives $\sum_i\hat u_i=0$. Since $y_i=\hat y_i+\hat u_i$, the sample average fitted outcome then equals the sample average outcome. With an intercept, each included predictor also has zero sample covariance with the residuals; calling this a zero sample **correlation** additionally requires both sample variances to be positive.

Without an intercept, a zero residual mean is not guaranteed. For example, fit only $b x$ to $x=(1,2)$ and $y=(1,1)$. OLS gives $b=3/5$ and residuals $(2/5,-1/5)$. The normal equation is satisfied, $1(2/5)+2(-1/5)=0$, but the residual sum is $1/5$, not zero.
<!-- bilingual-en:end -->

若设计包含全一的截距列，该列的方程给出 $\sum_i\hat u_i=0$。再由 $y_i=\hat y_i+\hat u_i$，得到样本拟合值均值等于样本结果均值。含截距时，各已纳入解释变量与残差的样本协方差也为零；若称为样本**相关系数**等于零，还要求两边样本方差都为正。

没有截距时，残差均值不保证为零。例如对 $x=(1,2)$、$y=(1,1)$ 只拟合 $bx$，OLS 得到 $b=3/5$、残差 $(2/5,-1/5)$。正规方程 $1(2/5)+2(-1/5)=0$ 成立，但残差和为 $1/5$，并非零。

<!-- bilingual-en:start -->
These are algebraic statements about the same rows and columns used to fit the model. They do not establish [[样本正交与总体外生性|population exogeneity]], absence of omitted structure, or predictive accuracy on new data. Restricting coefficients or adding a penalty changes the optimality conditions. With fixed positive observation weights $w_i$, [[加权最小二乘|weighted least squares]] instead gives $\sum_i w_i x_{ij}\hat u_i=0$; its residuals need not satisfy the unweighted equations. No Gaussian error assumption is needed for any of the unweighted sample identities above.
<!-- bilingual-en:end -->

这些都是关于本次拟合所用同一批行、同一组列的代数结论，不能证明[[样本正交与总体外生性|总体外生性]]、没有遗漏结构，或新数据上的预测准确性。约束系数或加入惩罚项，会改变最优条件。若每条观测有固定正权重 $w_i$，[[加权最小二乘]]给出的则是 $\sum_i w_i x_{ij}\hat u_i=0$，残差未必满足无权重方程。上述无权重样本恒等式均不要求误差正态。

## 继续

- [[一元 OLS 斜率|The simple-regression slope · 一元 OLS 斜率]]
- [[样本正交与总体外生性|Sample identity and population assumptions · 样本正交与总体外生性]]

## 来源与核验

<!-- bilingual-en:start -->
- Hansen, [Econometrics, January 2010](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf), §§3.3–3.6, PDF pp.36–42: first-order equations, sample residual moments, the intercept consequence and matrix notation.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf#page=2|MIT 18.06SC Session 2.3 summary]], p.2, and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S03_Lecture_Lecture_16_Projection_Matrices_and_Least_Squares.pdf#page=9|Lecture 16 transcript]], pp.9–14: normal equations, direct minimization and orthogonal residuals. The examples and perturbation identities here are recalculated directly.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1. 回归系数的估计|Permanent multiple-regression course section]]: columnwise equations and the local coefficient notation.
- Further reading retained from the original card: Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 2 §§2.2–2.3 and Chapter 3 §§3.2a,3.2e.
<!-- bilingual-en:end -->

- Hansen 指定页核对一阶方程、样本残差矩、截距推论与矩阵表达。
- MIT 指定页核对正规方程、直接最小化和正交残差；本文算例与扰动恒等式直接复算。
- 永久多元回归课程保留逐列方程与本地符号；Wooldridge 为原卡保留的延伸阅读。
