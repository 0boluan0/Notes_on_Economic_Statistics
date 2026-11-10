---
aliases:
  - "DFFITS"
  - "Difference in fits"
student_os: knowledge-atom
atom_id: ECON-SPEC-014
atom_set: regression-influence-diagnostics
atom_type: definition
status: source-checked
requires:
  - "[[Hat杠杆值]]"
  - "[[学生化残差]]"
  - "[[删点影响定义]]"
related:
  - "[[Cook距离]]"
  - "[[DFBETAS]]"
leads_to:
  - "[[影响诊断阈值]]"
  - "[[异常观测处理原则]]"
part_of:
  - "[[异常值、杠杆与影响诊断.canvas|异常值、杠杆与影响诊断]]"
---

<!-- bilingual-en:start -->
# DFFITS
<!-- bilingual-en:end -->

*删点后的局部拟合变化*

<!-- bilingual-en:start -->
DFFITS measures the **signed change in the fitted value at one observation’s predictor position when that observation is removed**. Both models are evaluated at that same position. The change is then scaled using the deleted-fit residual variability and the original design. This is a local view of [[删点影响定义|deletion influence]], complementing the all-position summary in [[Cook距离|Cook’s distance]].
<!-- bilingual-en:end -->

DFFITS 衡量的是：**删去一条观测后，在它原来的解释变量位置上，拟合值发生了多大的有符号变化**。两次拟合在同一个位置求值，再用删点后的残差波动和原设计提供的尺度进行缩放。这是[[删点影响定义|删点影响]]的局部视角，与 [[Cook距离|Cook 距离]]对全部位置的汇总互补。

<!-- bilingual-en:start -->
For example, fit an intercept and slope to $x=(-1,-1,0,0,1,1)$ and $y=(7,9,9,11,11,13)$. The full line is $\hat y=10+2x$. Removing observation 6 gives $\hat y_{(6)}(x)=68/7+(11/7)x$. At its original position $x=1$, the fitted value falls from $12$ to $79/7$, a full-minus-deleted difference of $5/7$. The observed outcome $13$ is not one of the two fitted values being subtracted.
<!-- bilingual-en:end -->

例如，对 $x=(-1,-1,0,0,1,1)$、$y=(7,9,9,11,11,13)$ 拟合截距和斜率。全样本直线是 $\hat y=10+2x$；删去第 6 条后，直线变为 $\hat y_{(6)}(x)=68/7+(11/7)x$。在它原来的 $x=1$ 处，拟合值从 $12$ 降到 $79/7$，全样本减去删点的差为 $5/7$。这里相减的是两次拟合值，不是把观测结果 $13$ 拿来作差。

<!-- bilingual-en:start -->
## Specify both parts of the scale
<!-- bilingual-en:end -->

*把尺度的两部分写清楚*

<!-- bilingual-en:start -->
Use unweighted [[普通最小二乘|OLS]] with a real $n\times p$ design $X$, where $p\ge1$ includes an intercept if fitted. Assume [[满列秩与OLS唯一性|full column rank]] before and after deletion and $n>p+1$. Row $i$ is $x_i^T$; $T$ means [[矩阵转置|transpose]]. The full and deleted coefficient vectors are $\hat\beta$ and $\hat\beta_{(i)}$, so $\hat y_i=x_i^T\hat\beta$ and $\hat y_{i(i)}=x_i^T\hat\beta_{(i)}$. Define
<!-- bilingual-en:end -->

采用不加权[[普通最小二乘|OLS]]，实数设计矩阵 $X$ 有 $n$ 行、$p\ge1$ 列，若拟合截距，也计入 $p$。假设全样本和删点后都[[满列秩与OLS唯一性|满列秩]]，且 $n>p+1$。第 $i$ 行为 $x_i^T$，$T$ 表示[[矩阵转置|转置]]。全样本与删点系数分别为 $\hat\beta$、$\hat\beta_{(i)}$，因此 $\hat y_i=x_i^T\hat\beta$、$\hat y_{i(i)}=x_i^T\hat\beta_{(i)}$。定义

$$
s_{(i)}^2=\frac{\sum_{j\ne i}(y_j-x_j^T\hat\beta_{(i)})^2}{n-p-1},
\qquad h_{ii}=x_i^T(X^TX)^{-1}x_i.
$$

<!-- bilingual-en:start -->
The first quantity is the deleted-fit [[OLS经典协方差估计|residual mean square]]: $n-1$ retained observations minus $p$ fitted coefficients leaves $n-p-1$ residual degrees of freedom. Assume $s_{(i)}>0$, taking the positive square root. The second quantity is [[Hat杠杆值|leverage from the full design]], not from the deleted design. For $0<h_{ii}<1$,
<!-- bilingual-en:end -->

第一项是删点后的[[OLS经典协方差估计|残差均方]]：保留 $n-1$ 条观测，再减去 $p$ 个拟合系数，得到 $n-p-1$ 个残差自由度。取正平方根并假设 $s_{(i)}>0$。第二项是**全样本设计**的[[Hat杠杆值|杠杆值]]，不是删点设计的杠杆值。当 $0<h_{ii}<1$ 时，

$$
\operatorname{DFFITS}_i
=\frac{\hat y_i-\hat y_{i(i)}}{s_{(i)}\sqrt{h_{ii}}}.
$$

<!-- bilingual-en:start -->
A positive value means inclusion of the observation raises the fitted value at that position; a negative value means it lowers it. The numerator and denominator have the same outcome units, making DFFITS dimensionless. The denominator deliberately combines a deleted residual scale with full-design geometry.
<!-- bilingual-en:end -->

正值表示纳入这条观测会提高该位置的拟合值，负值表示会降低。分子、分母的结果单位相同，因此 DFFITS 无单位。它的分母按定义将删点残差尺度与全设计几何组合在一起。

> [!important] A diagnostic scale, not the standard error of the difference / 诊断尺度不等于差值标准误
> <!-- bilingual-en:start -->
> Under the classical model $y=X\beta+u$, with fixed $X$, $E(u\mid X)=0$ and $\operatorname{Cov}(u\mid X)=\sigma^2I$, $0<\sigma^2<\infty$, the full fitted mean has variance $\sigma^2h_{ii}$. Here $I$ is the identity matrix and $u$ is the error vector. Thus $s_{(i)}\sqrt{h_{ii}}$ estimates that full-design fitted-mean scale using the deleted variance estimate. The two fits share observations, so their difference has a different variance:
> <!-- bilingual-en:end -->
>
> 在经典模型 $y=X\beta+u$ 中，若固定 $X$、$E(u\mid X)=0$、$\operatorname{Cov}(u\mid X)=\sigma^2I$ 且 $0<\sigma^2<\infty$，全样本拟合均值的方差是 $\sigma^2h_{ii}$。其中 $I$ 是单位矩阵，$u$ 是误差向量。因此，$s_{(i)}\sqrt{h_{ii}}$ 是用删点方差估计来估计全设计拟合均值的尺度。两次拟合共享观测，它们的差有另一种方差：
> $$\operatorname{Var}(\hat y_i-\hat y_{i(i)}\mid X)=\sigma^2\frac{h_{ii}^2}{1-h_{ii}}.$$
> <!-- bilingual-en:start -->
> The deleted fit at $x_i$ also has its own variance, $\sigma^2h_{ii}/(1-h_{ii})$. Neither of these is generally $\sigma^2h_{ii}$. Calling the DFFITS denominator “the deleted prediction standard error” or “the standard error of the difference” changes its meaning.
> <!-- bilingual-en:end -->
>
> 删点拟合在 $x_i$ 处的方差又是 $\sigma^2h_{ii}/(1-h_{ii})$，两者一般都不等于 $\sigma^2h_{ii}$。把 DFFITS 分母称作“删点预测的标准误”或“差值的标准误”，都会改变其含义。

<!-- bilingual-en:start -->
## Calculate it from residual and leverage
<!-- bilingual-en:end -->

*用残差和杠杆算出它*

<!-- bilingual-en:start -->
Let $e_i=y_i-\hat y_i$. The [[学生化残差|externally studentized residual]] is $t_i=e_i/[s_{(i)}\sqrt{1-h_{ii}}]$: it expresses the residual on a scale estimated without that row. The deletion identity gives
<!-- bilingual-en:end -->

令 $e_i=y_i-\hat y_i$。[[学生化残差|外部学生化残差]]为 $t_i=e_i/[s_{(i)}\sqrt{1-h_{ii}}]$，它用不含该行的估计尺度来衡量残差。删点恒等式给出

$$
\hat y_i-\hat y_{i(i)}=\frac{h_{ii}e_i}{1-h_{ii}},
\qquad
\operatorname{DFFITS}_i=t_i\sqrt{\frac{h_{ii}}{1-h_{ii}}}.
$$

<!-- bilingual-en:start -->
This explains why a moderate external residual can accompany large DFFITS at leverage close to one. In the opening example, $h_{66}=5/12$, the deleted residual sum of squares is $30/7$ and its degrees of freedom are $6-2-1=3$, so $s_{(6)}^2=10/7$. Hence
<!-- bilingual-en:end -->

这说明杠杆接近一时，即使外部学生化残差不极端，DFFITS 仍可能很大。开篇例子中，$h_{66}=5/12$，删点残差平方和为 $30/7$，自由度为 $6-2-1=3$，故 $s_{(6)}^2=10/7$。于是

$$
\operatorname{DFFITS}_6=\frac{5/7}{\sqrt{(10/7)(5/12)}}
=\sqrt{\frac67}\approx0.926.
$$

<!-- bilingual-en:start -->
The value is positive because including observation 6 raises the fitted mean at $x=1$. It does not identify which coefficient changed; [[DFBETAS|DFBETAS]] separates the coefficient movements. Nor is it the deleted prediction error: $y_6-\hat y_{6(6)}=13-79/7=12/7$, a different subtraction.
<!-- bilingual-en:end -->

数值为正，因为纳入第 6 条会提高 $x=1$ 处的拟合均值。它不指出哪个系数发生了变化；[[DFBETAS]]负责分开各个系数。它也不是删点预测误差：$y_6-\hat y_{6(6)}=13-79/7=12/7$，这是另一种作差。

<!-- bilingual-en:start -->
## Boundaries of interpretation
<!-- bilingual-en:end -->

## 解释边界

<!-- bilingual-en:start -->
The deletion identities are algebraic and need no Gaussian assumption. The sampling-variance interpretations in the callout use the stated classical covariance model; heteroskedastic or correlated errors do not automatically have that scale. Weighted or generalised linear models also require their corresponding definitions. DFFITS alone is not a calibrated hypothesis test, and a large absolute value prompts [[异常观测处理原则|investigation]], not automatic deletion; see [[影响诊断阈值|screening thresholds]].
<!-- bilingual-en:end -->

删点恒等式是代数关系，不要求 Gaussian 假设。callout 中的抽样方差解释使用所述经典协方差模型；异方差或相关误差不自动具有那种尺度。加权或广义线性模型也需要相应定义。DFFITS 本身不是已经校准的假设检验，绝对值大提示[[异常观测处理原则|进一步调查]]，不自动要求删除，详见[[影响诊断阈值]]。

<!-- bilingual-en:start -->
If $h_{ii}=1$, deletion loses rank. If $n-p-1\le0$ or $s_{(i)}^2=0$, the deleted residual scale is unavailable or zero. The displayed standardisation is then undefined. With an intercept, $h_{ii}\ge1/n>0$. Without an intercept, $h_{ii}=0$ can occur: then $x_i=0$, both fitted values at this row are zero, and the first fraction is $0/0$. Provided the other conditions hold, the residual–leverage expression extends DFFITS to zero; this is an extension, not evaluation of the undefined fraction. Finally, single-row deletion need not expose a group of observations that masks its members.
<!-- bilingual-en:end -->

若 $h_{ii}=1$，删点会失去满秩；若 $n-p-1\le0$ 或 $s_{(i)}^2=0$，删点残差尺度不存在或为零，上面的标准化便没有定义。含截距时，$h_{ii}\ge1/n>0$；无截距时可能 $h_{ii}=0$，此时 $x_i=0$，该行两次拟合值都为零，第一个分式成为 $0/0$。若其余条件满足，可通过残差—杠杆表达式把 DFFITS 延拓为零，这是延拓而非对未定义分式求值。最后，单行删除未必暴露成员互相掩蔽的一组观测。

## 来源与核验

<!-- bilingual-en:start -->
- Penn State STAT 501, [Lesson 11.5](https://online.stat.psu.edu/stat501/Lesson11): the fitted-value difference and displayed DFFITS denominator. The variance interpretations above are derived directly from the OLS deletion identity and classical covariance, rather than from the source’s informal description of that denominator.
- Stata, [*regress postestimation*, p. 34](https://www.stata.com/manuals/rregresspostestimation.pdf#page=34): externally studentized residual, deleted degrees of freedom, and the DFITS identity.
- R Core, [official `dffits.lm` implementation](https://svn.r-project.org/R/trunk/src/library/stats/R/lm.influence.R) and [diagnostics documentation](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html): checks the factor involving leverage and the deleted residual scale. The six-observation example and endpoint cases are direct calculations.
<!-- bilingual-en:end -->

- Penn State 上述 §11.5 支持拟合值作差及其展示的 DFFITS 分母。正文的方差解释由 OLS 删点恒等式和经典协方差直接推导，不沿用原文对分母的非正式描述。
- Stata 上述第 34 页核对外部学生化残差、删点自由度与 DFITS 恒等式。
- R Core 上述 `dffits.lm` 官方实现和文档核对杠杆因子及删点残差尺度。六观测例与端点情形为直接计算。
