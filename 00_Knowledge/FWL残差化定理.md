---
aliases:
  - Frisch-Waugh-Lovell theorem
  - FWL theorem
  - Partialling-out theorem
  - 弗里施-沃-洛维尔定理
student_os: knowledge-atom
atom_id: ECON-OLS-009
atom_type: theorem
status: source-checked
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
  - "[[多重共线性.canvas|多重共线性与设计矩阵诊断]]"
requires:
  - "[[OLS正规方程]]"
explains:
  - "[[多元回归系数]]"
---

<!-- bilingual-en:start -->
# The Frisch–Waugh–Lovell theorem
<!-- bilingual-en:end -->

FWL 残差化定理

<!-- bilingual-en:start -->
How does a regression “control for experience” when estimating an education coefficient? The Frisch–Waugh–Lovell (FWL) theorem makes the calculation visible: remove the part of education linearly predicted by the controls; remove the part of wages predicted by the **same** controls; then fit a slope between the two remaining parts. That slope is exactly the education coefficient from the full [[普通最小二乘|OLS regression]]. This explains a [[多元回归系数|multiple-regression coefficient]] without treating “control for” as a black box.
<!-- bilingual-en:end -->

估计教育系数时，回归究竟怎样“控制经验”？Frisch–Waugh–Lovell（FWL）定理把这个过程展示出来：先去掉教育中能被控制变量线性预测的部分，再去掉工资中能被**同一组**控制变量预测的部分，然后在两个剩余部分之间拟合斜率。所得斜率与完整[[普通最小二乘|OLS 回归]]中的教育系数完全相同。这样就能理解[[多元回归系数]]，而不用把“控制”当成黑箱。

<!-- bilingual-en:start -->
## Set up the same regression problem
<!-- bilingual-en:end -->
## 使用条件与含义边界

<!-- bilingual-en:start -->
Let $y$ and $x$ be $n$-dimensional real column vectors containing the outcome and one focal predictor. Let $Z$ be an $n\times q$ matrix of controls, with $q\ge1$; include a column of ones in $Z$ if the model has an intercept. Assume the full matrix $[Z\ x]$ has [[满列秩与OLS唯一性|full column rank]] $q+1$. Thus $Z$ has independent columns, and $x$ cannot be fitted exactly from them. Full OLS chooses a control-coefficient vector $\gamma$ and a focal slope $b$ to minimise
<!-- bilingual-en:end -->

令 $y$、$x$ 为 $n$ 维实列向量，分别装入结果和一个焦点预测变量。$Z$ 是 $n\times q$ 控制矩阵，$q\ge1$；如果模型含截距，就把全为 $1$ 的一列放入 $Z$。假定完整矩阵 $[Z\ x]$ [[满列秩与OLS唯一性|满列秩]]，秩为 $q+1$。这意味着 $Z$ 各列线性无关，而且不能用它们精确拟合 $x$。完整 OLS 选择控制系数向量 $\gamma$ 和焦点斜率 $b$，使

$$\|y-Z\gamma-xb\|_2^2$$

<!-- bilingual-en:start -->
The squared [[欧氏范数|Euclidean norm]] $\|v\|_2^2=\sum_i v_i^2$ is simply the sum of squared entries. All steps below use the same observations and unweighted squared-error criterion.
<!-- bilingual-en:end -->

最小。平方[[欧氏范数]] $\|v\|_2^2=\sum_i v_i^2$ 就是各分量的平方和。以下每一步都使用相同观测和不加权的平方误差准则。

<!-- bilingual-en:start -->
The fitted part of a vector on $Z$ is its [[正交投影|orthogonal projection]] into the [[列空间|column space]] of $Z$. The [[满列秩投影公式|projection matrix]] is $P_Z=Z(Z^TZ)^{-1}Z^T$; the superscript $T$ denotes [[矩阵转置|transpose]], and $-1$ denotes a [[逆矩阵|matrix inverse]]. Define $M_Z=I_n-P_Z$, where $I_n$ is the $n\times n$ identity matrix. This residual-making matrix subtracts the fitted part:
<!-- bilingual-en:end -->

一个向量对 $Z$ 的拟合部分，就是它在 $Z$ [[列空间]]中的[[正交投影]]。[[满列秩投影公式|投影矩阵]]为 $P_Z=Z(Z^TZ)^{-1}Z^T$，上标 $T$ 表示[[矩阵转置|转置]]，$-1$ 表示[[逆矩阵]]。令 $M_Z=I_n-P_Z$，其中 $I_n$ 为 $n\times n$ 单位矩阵。这个残差生成矩阵把拟合部分减掉：

$$\tilde x=M_Zx,\qquad \tilde y=M_Zy.$$

> [!important] FWL for one focal coefficient / 一个焦点系数的 FWL 公式
> <!-- bilingual-en:start -->
> Regress $\tilde y$ on $\tilde x$ **through the origin**. The slope and its residual vector are
> <!-- bilingual-en:end -->
> 将 $\tilde y$ 对 $\tilde x$ 作**不另加截距**的回归。所得斜率及残差向量为
>
> $$\hat b=\frac{\tilde x^T\tilde y}{\tilde x^T\tilde x}
> =\frac{x^TM_Zy}{x^TM_Zx},\qquad
> \hat u=\tilde y-\tilde x\hat b.$$
>
> <!-- bilingual-en:start -->
> They are exactly the focal coefficient and the residuals of the full regression on $[Z\ x]$. The denominator is positive because $x$ is outside the column space of $Z$.
> <!-- bilingual-en:end -->
> 它们恰好是对 $[Z\ x]$ 完整回归中的焦点系数与残差。由于 $x$ 不在 $Z$ 的列空间中，分母为正。

<!-- bilingual-en:start -->
If $Z$ contains an intercept, both residual vectors have mean zero. The formula then also has the centered [[一元 OLS 斜率|simple-slope]] form; adding an intercept to the last regression gives a zero intercept and the same slope. If the original model has no intercept, adding one only at the last stage can change the problem. If $x$ lies in the span of $Z$, $\tilde x=0$ and the separate coefficient is not uniquely determined. Redundant controls can instead be represented by a linearly independent basis of the same column space before using the displayed inverse.
<!-- bilingual-en:end -->

如果 $Z$ 含截距，两个残差向量的均值均为零，公式也就具有中心化[[一元 OLS 斜率]]的形式；最后一步再加截距，会得到零截距和同一斜率。如果原模型没有截距，仅在最后一步加入截距可能改变问题。如果 $x$ 已落在 $Z$ 的张成空间中，$\tilde x=0$，单独系数便不能唯一确定。控制列存在冗余时，可以先用同一列空间的一组线性无关基替代它们，再使用这里的逆矩阵公式。

<!-- bilingual-en:start -->
## Why the two remaining parts are enough
<!-- bilingual-en:end -->
## 自然解释

<!-- bilingual-en:start -->
For any proposed slope $b$, the best control coefficients fit $y-xb$ on $Z$. The remaining error is therefore $M_Z(y-xb)=\tilde y-b\tilde x$. Minimising over the controls has reduced the original problem to minimising $\|\tilde y-b\tilde x\|_2^2$ over $b$. Its expansion is
<!-- bilingual-en:end -->

对任意候选斜率 $b$，最优控制系数就是把 $y-xb$ 对 $Z$ 回归的系数，因此剩余误差为 $M_Z(y-xb)=\tilde y-b\tilde x$。先对控制系数最优化，就把原问题化为只对 $b$ 最小化 $\|\tilde y-b\tilde x\|_2^2$。展开得到

$$\tilde y^T\tilde y-2b\tilde x^T\tilde y+b^2\tilde x^T\tilde x.$$

<!-- bilingual-en:start -->
Completing the square gives the stated slope. At that slope, the remaining vector is still the full regression's residual. If the other coefficients are wanted, recover them as $\hat\gamma=(Z^TZ)^{-1}Z^T(y-x\hat b)$. No normality, random sampling, or exogeneity assumption was used: this is a finite-data identity.
<!-- bilingual-en:end -->

配方即可得到上面的斜率；在该斜率下，剩余向量仍是完整回归的残差。若还想求其他系数，可用 $\hat\gamma=(Z^TZ)^{-1}Z^T(y-x\hat b)$ 恢复。推导没有使用正态、随机抽样或外生性假设：它是有限数据的代数恒等式。

<!-- bilingual-en:start -->
### A four-observation calculation

Use education $x=(0,1,1,2)^T$, experience $z=(0,0,1,1)^T$, and wage $y=(2,2,6,10)^T$, as in the multiple-coefficient example. Let $Z$ contain an intercept and $z$. Since $z$ is binary here, regression on $Z$ fits each of its two group means. The fitted education values are $(0.5,0.5,1.5,1.5)^T$, and fitted wages are $(2,2,8,8)^T$.
<!-- bilingual-en:end -->

### 四个观测的完整计算

沿用多元系数卡的教育 $x=(0,1,1,2)^T$、经验 $z=(0,0,1,1)^T$ 和工资 $y=(2,2,6,10)^T$。令 $Z$ 包含截距与 $z$。本例 $z$ 只有两个取值，所以对 $Z$ 的回归拟合出各组的均值。教育拟合值为 $(0.5,0.5,1.5,1.5)^T$，工资拟合值为 $(2,2,8,8)^T$。

> [!example] Compare the remaining variation / 比较剩余变动
> $$\tilde x=(-0.5,0.5,-0.5,0.5)^T,\qquad \tilde y=(0,0,-2,2)^T.$$
>
> $$\tilde x^T\tilde y=2,\quad \tilde x^T\tilde x=1,\quad \hat b=2.$$
>
> <!-- bilingual-en:start -->
> Recovering the controls gives the full fitted equation $\hat y=1+2x+4z$. Both procedures leave $\hat u=(1,-1,-1,1)^T$ and residual sum of squares $\hat u^T\hat u=4$. The raw regression on education alone has slope $4$; removing experience's linear predictions from **both** variables produces the partial slope $2$.
> <!-- bilingual-en:end -->
> 恢复控制系数后，完整拟合式为 $\hat y=1+2x+4z$。两种算法都留下 $\hat u=(1,-1,-1,1)^T$，残差平方和为 $\hat u^T\hat u=4$。只用原始教育回归时斜率为 $4$；从**两个**变量中都移除经验的线性预测，得到偏斜率 $2$。

<!-- bilingual-en:start -->
The residualised $x$ is the part that cannot be linearly fitted by these controls. “Cannot be linearly fitted” does not mean statistically independent of them, and residualising does not remove an omitted variable or establish a causal effect. The theorem concerns the same included columns, transformations, observations, and fitting criterion. For [[加权最小二乘|weighted least squares]], all stages need the corresponding weighted projection (or the same whitening transformation); mixing ordinary residuals with a weighted final regression generally loses the equivalence.
<!-- bilingual-en:end -->

残差化后的 $x$ 是不能由这些控制变量线性拟合的部分。“不能线性拟合”不代表与控制变量统计独立；残差化也不能消除遗漏变量或确立因果效应。定理针对的是同一组已含列、变换形式、观测和拟合准则。对于[[加权最小二乘]]，所有步骤都需要相应的加权投影，或使用同一白化变换；普通残差与加权的最后一步混用，一般会失去等价性。

<!-- bilingual-en:start -->
## Coefficient equality does not set the standard-error defaults

A [[标准误含义|standard error]] estimates sampling variability, so it needs assumptions beyond the identity. Suppose the data follow $y=Z\gamma+xb+\varepsilon$, where $\varepsilon$ is the model error vector. Conditional on the full design, assume $E[\varepsilon\mid Z,x]=0$ and $\operatorname{Cov}(\varepsilon\mid Z,x)=\sigma^2I_n$ for a finite $\sigma^2$. Then $\operatorname{Var}(\hat b\mid Z,x)=\sigma^2/(\tilde x^T\tilde x)$. The [[OLS经典协方差估计|classical estimate]] replaces $\sigma^2$ by $s^2=\hat u^T\hat u/(n-q-1)$ when $n>q+1$. The controls have already used $q$ coefficient degrees of freedom even though they disappear from the last equation.
<!-- bilingual-en:end -->

### 系数相同，不代表标准误默认设置相同

[[标准误含义|标准误]]估计抽样波动，因此需要恒等式以外的假设。设数据满足 $y=Z\gamma+xb+\varepsilon$，其中 $\varepsilon$ 是模型误差向量。在给定完整设计下，假定 $E[\varepsilon\mid Z,x]=0$、$\operatorname{Cov}(\varepsilon\mid Z,x)=\sigma^2I_n$，其中 $\sigma^2$ 有限。则 $\operatorname{Var}(\hat b\mid Z,x)=\sigma^2/(\tilde x^T\tilde x)$。当 $n>q+1$ 时，[[OLS经典协方差估计]]用 $s^2=\hat u^T\hat u/(n-q-1)$ 代替 $\sigma^2$。控制变量虽然从最后的方程中消失，仍已占用 $q$ 个系数自由度。

<!-- bilingual-en:start -->
In the four-observation example, $q=2$, so the full residual degrees of freedom are $4-2-1=1$. Thus $s^2=4$ and the estimated standard error of $\hat b$ is $2$. Software treating the final through-origin regression as a fresh one-coefficient problem would instead use $4-1=3$ degrees of freedom and report $2/\sqrt3$. To reproduce full-model inference, preserve its degrees of freedom and covariance rule, including full-design leverage adjustments when that rule uses them.
<!-- bilingual-en:end -->

四观测例子中 $q=2$，完整模型的残差自由度为 $4-2-1=1$。所以 $s^2=4$，$\hat b$ 的估计标准误为 $2$。若软件把最后的不含截距回归当成一个全新的单系数问题，就会用 $4-1=3$ 个自由度，报出 $2/\sqrt3$。要复现完整模型的推断，必须保留它的自由度与协方差规则；该规则使用完整设计的杠杆值修正时，也必须保留相应修正。

<!-- bilingual-en:start -->
## When little focal variation remains
<!-- bilingual-en:end -->
## 继续

<!-- bilingual-en:start -->
The same variance formula explains a common precision issue. Under the stated covariance model with a fixed positive $\sigma^2$, little residualised education variation means a small $\tilde x^T\tilde x$ and a large coefficient variance. This is the mechanism behind [[近似共线性精度|precision loss under near multicollinearity]], rather than an unconditional claim that adding controls always makes a coefficient imprecise. For the interpretation of the resulting slope, return to [[多元回归系数|multiple-regression coefficients]].
<!-- bilingual-en:end -->

同一方差公式也解释了常见的精度问题。在上述协方差模型及固定正值 $\sigma^2$ 下，教育残差变动少，意味着 $\tilde x^T\tilde x$ 小、系数方差大。这是[[近似共线性精度]]背后的机制，并不是无条件断言加入控制变量总会使系数不精确。所得斜率的含义可回到[[多元回归系数]]阅读。

## 来源与核验

<!-- bilingual-en:start -->
- Hansen, [*Econometrics* (2010)](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf), §3.7 (PDF pp. 42–43): the partitioned-regression formula and equality of full-model residuals; §4.4 (PDF pp. 56–57) and §§4.7–4.9 (PDF pp. 60–64): classical coefficient covariance, residual degrees of freedom, and leverage-sensitive covariance corrections. The focused scalar proof and worked example above are verified directly.
- [MIT 14.382 Lecture 1](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf), PDF pp. 2–5: population and sample partialling out, rank conditions, and the predictive rather than automatic causal interpretation. The least-squares objectives here explicitly include squared norms.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1.3. 系数计算：以二元回归模型为例|Local course: calculating a partial coefficient]]: residualising both the outcome and the focal predictor using the same controls.
<!-- bilingual-en:end -->

- Hansen，[《Econometrics》（2010）](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf)，§3.7（PDF 第 42–43 页）：分块回归公式及完整模型残差相等；§4.4（PDF 第 56–57 页）及 §§4.7–4.9（PDF 第 60–64 页）：经典系数协方差、残差自由度及依赖杠杆值的协方差修正。上文聚焦单系数的证明和算例另作直接核验。
- [MIT 14.382 Lecture 1](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/c62d33e015c910b0d126bcc9344cf2c5_MIT14_382S17_lec1.pdf)，PDF 第 2–5 页：总体与样本残差化、秩条件，以及预测关联不自动具有因果解释的边界。本文最小二乘目标显式保留平方范数。
- [[02_Economy/01_Econometrics/03_多元线性回归.md#2.1.3. 系数计算：以二元回归模型为例|本地课程：FWL 直觉]]：使用同一组控制变量同时残差化结果与焦点预测变量。
