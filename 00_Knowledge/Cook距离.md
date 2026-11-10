---
aliases:
  - "Cook's distance"
  - "Cook's D"
  - "库克距离"
student_os: knowledge-atom
atom_id: ECON-SPEC-013
atom_set: regression-influence-diagnostics
atom_type: definition
status: source-checked
part_of:
  - "[[异常值、杠杆与影响诊断.canvas|异常值、杠杆与影响诊断]]"
requires:
  - "[[Hat杠杆值]]"
  - "[[学生化残差]]"
  - "[[删点影响定义]]"
leads_to:
  - "[[影响诊断阈值]]"
  - "[[异常观测处理原则]]"
related:
  - "[[DFFITS]]"
  - "[[DFBETAS]]"
---

<!-- bilingual-en:start -->
# Cook’s distance
<!-- bilingual-en:end -->

*Cook 距离*

<!-- bilingual-en:start -->
Cook’s distance measures how much **the entire fitted regression changes when one observation is removed**. Fit the model using all observations, fit the same model again without one row, and compare the two sets of fitted values at every original predictor position. Squaring and adding those changes gives a global measure of [[删点影响定义|deletion influence]]; Cook’s distance then puts that total on a common residual scale.
<!-- bilingual-en:end -->

Cook 距离衡量的是：**删去一条观测以后，整个回归拟合改变了多少**。先用全部观测拟合，再去掉一行、重新拟合同一个模型，随后在所有原来的解释变量位置比较两次拟合值。把这些变化平方后相加，就得到一种全局的[[删点影响定义|删点影响]]；Cook 距离进一步用残差尺度把它标准化。

<!-- bilingual-en:start -->
## Follow the movement of a fitted line
<!-- bilingual-en:end -->

*看一条拟合直线怎样移动*

<!-- bilingual-en:start -->
Consider six observations of a predictor $x$ and an outcome $y$. [[普通最小二乘|Ordinary least squares (OLS)]] chooses the intercept and slope that minimise the sum of squared [[回归残差|residuals]], where a residual is observed minus fitted outcome. Here the fitted line is $\hat y=10+2x$.
<!-- bilingual-en:end -->

考虑解释变量 $x$ 与结果 $y$ 的六条观测。[[普通最小二乘|普通最小二乘（OLS）]]选择使[[回归残差|残差]]平方和最小的截距和斜率；残差就是观测结果减去拟合结果。本例的拟合直线为 $\hat y=10+2x$。

| Observation / 观测 $i$ | $x_i$ | $y_i$ | $\hat y_i$ | $e_i=y_i-\hat y_i$ |
|---|---|---|---|---|
| 1 | −1 | 7 | 8 | −1 |
| 2 | −1 | 9 | 8 | 1 |
| 3 | 0 | 9 | 10 | −1 |
| 4 | 0 | 11 | 10 | 1 |
| 5 | 1 | 11 | 12 | −1 |
| 6 | 1 | 13 | 12 | 1 |

<!-- bilingual-en:start -->
Deleting observation 6 changes the line to $\hat y_{(6)}(x)=68/7+(11/7)x$. Subtract the deleted fit from the full fit: the changes are $-1/7$ at each $x=-1$ row, $2/7$ at each $x=0$ row, and $5/7$ at each $x=1$ row. The deleted row is still a location at which to compare the two lines; it simply no longer contributes to fitting the second line.
<!-- bilingual-en:end -->

删去第 6 条后，直线变成 $\hat y_{(6)}(x)=68/7+(11/7)x$。用全样本拟合减去删点拟合：两条 $x=-1$ 观测处的变化各为 $-1/7$，两条 $x=0$ 处各为 $2/7$，两条 $x=1$ 处各为 $5/7$。被删除行的位置仍然参与两条直线的比较，只是不再参与第二条直线的估计。

<!-- bilingual-en:start -->
## Scale the total change
<!-- bilingual-en:end -->

*把总变化放到残差尺度上*

<!-- bilingual-en:start -->
For the general definition, let $X$ be a real $n\times p$ design matrix with [[满列秩与OLS唯一性|full column rank]], so its $p\ge1$ fitted coefficients are unique. Count an intercept in $p$ if present. Let $x_j^T$ be row $j$, with $T$ denoting [[矩阵转置|transpose]]. Write $\hat\beta$ for the full-fit coefficients and $\hat\beta_{(i)}$ for those after deleting row $i$. Define $\hat y_j=x_j^T\hat\beta$, $\hat y_{j(i)}=x_j^T\hat\beta_{(i)}$, and the full-fit residual sum of squares $RSS=\sum_j(y_j-\hat y_j)^2$. For $n>p$, its [[OLS经典协方差估计|residual mean square]] is $s^2=RSS/(n-p)$; $n-p$ is the residual degrees of freedom. Assume $s^2>0$ and that deleting row $i$ leaves full column rank.
<!-- bilingual-en:end -->

一般定义中，设实数设计矩阵 $X$ 有 $n$ 行、$p$ 列并且[[满列秩与OLS唯一性|满列秩]]，因此 $p\ge1$ 个拟合系数唯一；若模型含截距，也计入 $p$。设 $x_j^T$ 是第 $j$ 行，$T$ 表示[[矩阵转置|转置]]。全样本系数记为 $\hat\beta$，删去第 $i$ 行后的系数记为 $\hat\beta_{(i)}$。定义 $\hat y_j=x_j^T\hat\beta$、$\hat y_{j(i)}=x_j^T\hat\beta_{(i)}$，全样本残差平方和为 $RSS=\sum_j(y_j-\hat y_j)^2$。当 $n>p$ 时，[[OLS经典协方差估计|残差均方]]为 $s^2=RSS/(n-p)$，其中 $n-p$ 是残差自由度。假设 $s^2>0$，且删去第 $i$ 行后仍满列秩。

$$
D_i=\frac{\sum_{j=1}^n(\hat y_j-\hat y_{j(i)})^2}{p\,s^2}
=\frac{(\hat\beta-\hat\beta_{(i)})^T X^TX(\hat\beta-\hat\beta_{(i)})}{p\,s^2}.
$$

<!-- bilingual-en:start -->
The second expression is the same total written in coefficient coordinates: multiply the coefficient change by $X$ to recover every fitted-value change. The denominator uses the **full-fit** $s^2$, not the deleted-fit residual mean square. Numerator and denominator both have squared outcome units, so $D_i$ is dimensionless and nonnegative.
<!-- bilingual-en:end -->

第二个表达式只是用系数坐标重写同一个总量：系数变化乘以 $X$，就还原了每个拟合值的变化。分母使用**全样本**的 $s^2$，不是删点后的残差均方。分子和分母都有结果单位的平方，因此 $D_i$ 无单位且非负。

> [!example] Finish the six-observation calculation / 算完六条观测的例子
> <!-- bilingual-en:start -->
> There are $p=2$ coefficients and $RSS=6$, so $s^2=6/(6-2)=3/2$. The total squared movement is $2(1+4+25)/49=60/49$. Thus
> <!-- bilingual-en:end -->
>
> 这里有 $p=2$ 个系数，$RSS=6$，故 $s^2=6/(6-2)=3/2$。总平方移动为 $2(1+4+25)/49=60/49$，所以
> $$D_6=\frac{60/49}{2(3/2)}=\frac{20}{49}\approx0.408.$$
> <!-- bilingual-en:start -->
> This number summarises the movement of both the intercept and slope through their effects on all six fitted values. It is neither a probability nor the proportion of observations affected.
> <!-- bilingual-en:end -->
>
> 这个数通过全部六个拟合值的变化，汇总了截距与斜率的移动；它不是概率，也不是受影响观测的比例。

<!-- bilingual-en:start -->
## Why both residual and leverage matter
<!-- bilingual-en:end -->

*为什么残差和杠杆共同参与*

<!-- bilingual-en:start -->
Let $e_i=y_i-\hat y_i$ and let $h_{ii}=x_i^T(X^TX)^{-1}x_i$ be the full-design [[Hat杠杆值|leverage]], which describes how strongly row $i$ can move its own fit. The OLS deletion identity gives
<!-- bilingual-en:end -->

设 $e_i=y_i-\hat y_i$，$h_{ii}=x_i^T(X^TX)^{-1}x_i$ 是全样本设计的[[Hat杠杆值|杠杆值]]，描述该行推动自身拟合的能力。OLS 的删点恒等式给出

$$
\hat\beta-\hat\beta_{(i)}=(X^TX)^{-1}x_i\frac{e_i}{1-h_{ii}},
\qquad
D_i=\frac{e_i^2}{p\,s^2}\frac{h_{ii}}{(1-h_{ii})^2}.
$$

<!-- bilingual-en:start -->
Substituting the first identity into the coefficient formula gives the second, since $x_i^T(X^TX)^{-1}x_i=h_{ii}$. In the example $e_6=1$ and $h_{66}=1/6+1/4=5/12$, reproducing $20/49$. A high-leverage observation with $e_i=0$ has $D_i=0$ when the stated conditions hold: leverage is potential, while this influence measure depends on the observed outcome as well.
<!-- bilingual-en:end -->

把第一个恒等式代入系数形式，由 $x_i^T(X^TX)^{-1}x_i=h_{ii}$ 就得到第二式。本例 $e_6=1$、$h_{66}=1/6+1/4=5/12$，同样得到 $20/49$。在上述条件满足时，即使杠杆很高，只要 $e_i=0$ 就有 $D_i=0$：杠杆描述潜力，这个影响量还取决于实际结果。

<!-- bilingual-en:start -->
## Conditions and interpretation
<!-- bilingual-en:end -->

## 边界

<!-- bilingual-en:start -->
These are exact identities for unweighted linear OLS; computing them does not require normal or homoskedastic errors. Interpreting $s^2$ as a common error-variance estimate does require the matching [[OLS经典协方差估计|classical covariance assumptions]]. A large value signals sensitivity of this fit, not proof of a data error or a causal effect. [[影响诊断阈值|Screening thresholds]] guide investigation; [[异常观测处理原则|deciding how to handle an observation]] requires its data and model context.
<!-- bilingual-en:end -->

这些是不加权线性 OLS 的精确代数恒等式，计算本身不要求正态或同方差误差；若要把 $s^2$ 解释为共同误差方差的估计，则需要匹配的[[OLS经典协方差估计|经典协方差假设]]。数值大表示当前拟合敏感，不证明数据错误或因果效应。[[影响诊断阈值|筛查阈值]]帮助调查；[[异常观测处理原则|如何处理观测]]还须结合数据与模型背景。

<!-- bilingual-en:start -->
For full-rank $X$, deletion preserves rank exactly when $h_{ii}<1$. At $h_{ii}=1$, the deleted coefficients are not uniquely determined; do not turn a resulting $0/0$ into “zero influence.” A perfect full fit with $s^2=0$ also has no Cook scale. Unlike DFFITS and DFBETAS, this definition does not need a positive deleted residual variance: even $n=p+1$ can permit Cook’s distance when the deleted design remains full rank and the full-fit $s^2$ is positive.
<!-- bilingual-en:end -->

对满列秩 $X$，删点后保持满秩恰好等价于 $h_{ii}<1$。在 $h_{ii}=1$ 时，删点系数不再唯一；不能把产生的 $0/0$ 当作“零影响”。全样本完美拟合导致 $s^2=0$ 时，也没有可用的 Cook 尺度。与 DFFITS、DFBETAS 不同，这一定义不需要正的删点残差方差；即使 $n=p+1$，只要删点后仍满秩且全样本 $s^2>0$，Cook 距离也可定义。

<!-- bilingual-en:start -->
Use [[DFFITS|DFFITS]] for the signed movement at the deleted row’s predictor position and [[DFBETAS|DFBETAS]] for individual coefficient movements. Weighted, nonlinear or generalised linear models need their own definitions or approximations. Single-row deletion can also miss a group whose members mask one another.
<!-- bilingual-en:end -->

若关心被删行的解释变量位置处的有符号变化，使用 [[DFFITS]]；若关心各个系数，使用 [[DFBETAS]]。加权、非线性或广义线性模型需要各自的定义或近似。单行删除也可能漏掉成员互相掩蔽的一组观测。

## 来源与核验

<!-- bilingual-en:start -->
- Penn State STAT 501, [Lesson 11.5](https://online.stat.psu.edu/stat501/Lesson11): deletion comparison, global fitted-value meaning, and the residual–leverage formula.
- Stata, [*regress postestimation*, p. 34](https://www.stata.com/manuals/rregresspostestimation.pdf#page=34): checks Cook’s formula and the parameter count including the intercept.
- R Core, [regression deletion diagnostics](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html) and [official implementation](https://svn.r-project.org/R/trunk/src/library/stats/R/lm.influence.R), `cooks.distance.lm`: checks full-fit variance scaling and distinguishes linear-model calculations from GLM approximations. The six-row example, coefficient identity and boundary cases are worked out directly.
<!-- bilingual-en:end -->

- Penn State 上述 §11.5 支持删点比较、全局拟合变化的含义及残差—杠杆公式。
- Stata 上述第 34 页核对 Cook 公式与含截距的参数计数。
- R Core 上述文档及 `cooks.distance.lm` 官方实现核对全样本方差缩放，并区分线性模型与 GLM 近似。六行例子、系数恒等式和边界情形由正文直接推导。
