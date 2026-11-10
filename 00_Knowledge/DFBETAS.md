---
aliases:
  - "DFBETAS"
  - "Standardized DFBETA"
student_os: knowledge-atom
atom_id: ECON-SPEC-015
atom_set: regression-influence-diagnostics
atom_type: definition
status: source-checked
requires:
  - "[[学生化残差]]"
  - "[[删点影响定义]]"
related:
  - "[[Cook距离]]"
  - "[[DFFITS]]"
leads_to:
  - "[[影响诊断阈值]]"
  - "[[异常观测处理原则]]"
part_of:
  - "[[异常值、杠杆与影响诊断.canvas|异常值、杠杆与影响诊断]]"
---

<!-- bilingual-en:start -->
# DFBETAS
<!-- bilingual-en:end -->

*逐系数的标准化删点变化*

<!-- bilingual-en:start -->
DFBETAS answers a targeted question: **how much does removing one observation change each fitted regression coefficient, relative to a coefficient-specific scale?** A point may move an intercept without moving a slope, or move one slope much more than another. Reporting one value for each observation–coefficient pair makes these differences visible; a single global measure such as [[Cook距离|Cook’s distance]] cannot show them separately.
<!-- bilingual-en:end -->

DFBETAS 回答一个具体问题：**删去一条观测后，每个拟合回归系数相对于该系数的尺度改变了多少？** 某个点可能只改变截距而不改变斜率，也可能对不同斜率的推动差很多。为每一组“观测—系数”分别报告数值，才能把差异展开；[[Cook距离|Cook 距离]]这样的单个全局指标不能分别显示它们。

<!-- bilingual-en:start -->
For a small example, fit an intercept and slope to $x=(-1,-1,0,0,1,1)$ and $y=(7,9,9,11,11,13)$. The full-fit coefficients are $(10,2)$. [[删点影响定义|Deleting observation 6 and refitting the same model]] gives $(68/7,11/7)$. Thus inclusion of this observation raises the intercept by $2/7$ and the slope by $3/7$. Before comparing the magnitudes, account for their different coefficient scales.
<!-- bilingual-en:end -->

看一个小例子：对 $x=(-1,-1,0,0,1,1)$、$y=(7,9,9,11,11,13)$ 拟合截距和斜率，全样本系数是 $(10,2)$。[[删点影响定义|删去第 6 条并重新拟合同一模型]]，得到 $(68/7,11/7)$。因此，纳入这条观测使截距提高 $2/7$、斜率提高 $3/7$。比较大小之前，还要考虑两种系数各自的尺度。

<!-- bilingual-en:start -->
## From raw coefficient movement to DFBETAS
<!-- bilingual-en:end -->

*从原单位系数变化到 DFBETAS*

<!-- bilingual-en:start -->
Let a real design matrix $X$ have $n$ observations and $p\ge1$ columns, counting an intercept if present. Fit unweighted [[普通最小二乘|OLS]]. Assume $X$ and the design after deleting observation $i$ both have [[满列秩与OLS唯一性|full column rank]], and $n>p+1$. Let $\hat\beta_j$ be full-fit coefficient $j$ and $\hat\beta_{j(i)}$ its deleted-fit counterpart. Define the raw, full-minus-deleted difference by
<!-- bilingual-en:end -->

设实数设计矩阵 $X$ 有 $n$ 条观测、$p\ge1$ 列，含截距时也将其计入 $p$，采用不加权[[普通最小二乘|OLS]]。假设 $X$ 以及删去第 $i$ 条后的设计都[[满列秩与OLS唯一性|满列秩]]，且 $n>p+1$。全样本第 $j$ 个系数为 $\hat\beta_j$，删点后对应系数为 $\hat\beta_{j(i)}$。按“全样本减删点”定义原单位变化

$$\operatorname{DFBETA}_{ij}=\hat\beta_j-\hat\beta_{j(i)}.$$

<!-- bilingual-en:start -->
This retains the coefficient’s units: an intercept is in outcome units, whereas a slope is in outcome units per predictor unit. For the standardisation, write $C=(X^TX)^{-1}$, where $T$ denotes [[矩阵转置|transpose]], and let $c_{jj}>0$ be its $j$th diagonal entry. Let $s_{(i)}^2$ be the sum of squared residuals in the deleted fit, divided by its residual degrees of freedom $n-p-1$, and assume its positive square root $s_{(i)}$ is nonzero. Then
<!-- bilingual-en:end -->

它保留系数单位：截距是结果单位，斜率是“结果单位／解释变量单位”。标准化时，记 $C=(X^TX)^{-1}$，其中 $T$ 表示[[矩阵转置|转置]]，$c_{jj}>0$ 是其第 $j$ 个对角元。令 $s_{(i)}^2$ 为删点拟合的残差平方和除以其残差自由度 $n-p-1$，并假设正平方根 $s_{(i)}$ 非零。于是

$$
\operatorname{DFBETAS}_{ij}=\frac{\hat\beta_j-\hat\beta_{j(i)}}{s_{(i)}\sqrt{c_{jj}}}.
$$

<!-- bilingual-en:start -->
The denominator combines the deleted-fit residual scale with the **full-design** coefficient factor $\sqrt{c_{jj}}$. Under the [[OLS经典协方差估计|classical covariance model]], the full-fit coefficient variance is $\sigma^2c_{jj}$; here its unknown error variance $\sigma^2$ is estimated using the deleted residuals. This denominator is generally neither the deleted coefficient’s own standard error nor the standard error of the difference between the two estimates. The algebraic diagnostic can still be computed without normality or homoskedasticity, but those classical scale interpretations do not automatically extend to other error structures.
<!-- bilingual-en:end -->

分母将删点残差尺度与**全设计**的系数因子 $\sqrt{c_{jj}}$ 组合起来。在[[OLS经典协方差估计|经典协方差模型]]下，全样本系数方差为 $\sigma^2c_{jj}$；这里用删点残差来估计未知误差方差 $\sigma^2$。这个分母一般既不是删点系数自身的标准误，也不是两次系数估计之差的标准误。代数诊断仍可在非正态或非同方差时计算，但经典尺度解释不自动推广到其他误差结构。

<!-- bilingual-en:start -->
For every observation, there are $p$ DFBETAS values, one for each coefficient. Positive means that including the observation raises the coefficient relative to its deleted fit; negative means it lowers it. Absolute values describe the size of those standardised changes.
<!-- bilingual-en:end -->

每条观测产生 $p$ 个 DFBETAS，每个系数对应一个。正值表示纳入该观测使系数比删点拟合更高，负值表示更低；绝对值描述这些标准化变化的大小。

<!-- bilingual-en:start -->
## Separate the intercept and slope in the example
<!-- bilingual-en:end -->

*把例子的截距和斜率分开*

<!-- bilingual-en:start -->
The six-row design has $\sum x_i=0$ and $\sum x_i^2=4$, so in coefficient order (intercept, slope), $X^TX=\operatorname{diag}(6,4)$ and $C=\operatorname{diag}(1/6,1/4)$; $\operatorname{diag}(a,b)$ means a diagonal matrix with entries $a,b$. Removing observation 6 leaves residual sum of squares $30/7$ and $3$ residual degrees of freedom, giving $s_{(6)}^2=10/7$. Therefore
<!-- bilingual-en:end -->

六行设计满足 $\sum x_i=0$、$\sum x_i^2=4$。按“截距、斜率”的顺序，$X^TX=\operatorname{diag}(6,4)$、$C=\operatorname{diag}(1/6,1/4)$，其中 $\operatorname{diag}(a,b)$ 表示对角元为 $a,b$ 的对角矩阵。删去第 6 条后，残差平方和为 $30/7$，剩下 $3$ 个残差自由度，所以 $s_{(6)}^2=10/7$。因此

$$
\operatorname{DFBETAS}_{6,\mathrm{intercept}}
=\frac{2/7}{\sqrt{(10/7)(1/6)}}=\sqrt{\frac{12}{35}}\approx0.586,
$$
$$
\operatorname{DFBETAS}_{6,\mathrm{slope}}
=\frac{3/7}{\sqrt{(10/7)(1/4)}}=\sqrt{\frac{18}{35}}\approx0.717.
$$

<!-- bilingual-en:start -->
Both coefficients move upward when observation 6 is included. On these standardised scales its effect is larger for the slope. Now delete observation 3 instead, whose predictor is $0$ and outcome is $9$. The deleted fit is $(51/5,2)$: the slope stays exactly $2$, so its DFBETAS is zero, while the intercept change is $-1/5$. A point can affect the fitted regression without changing every coefficient.
<!-- bilingual-en:end -->

纳入第 6 条时，两个系数都向上移动；按这些标准化尺度，它对斜率的影响较大。改为删去第 3 条（解释变量为 $0$，结果为 $9$），删点系数为 $(51/5,2)$：斜率仍恰好是 $2$，所以斜率 DFBETAS 为零，而截距变化为 $-1/5$。观测能够影响回归拟合，却不必改变每个系数。

> [!note] Keep the coefficient meaning fixed / 先固定系数的含义
> <!-- bilingual-en:start -->
> Merely changing a predictor to a positive multiple of its old units rescales its raw coefficient difference and its denominator equally, preserving that DFBETAS value. Recoding the model can instead change what a coefficient means. For instance, centring a predictor changes the intercept into the fitted value at the new origin; its DFBETAS can change even though the fitted line does not. Compare the same parameterisation, or specify the particular coefficient combination of interest.
> <!-- bilingual-en:end -->
>
> 仅把某解释变量改为原单位的正倍数，会等比例缩放原单位系数变化及其分母，因而该 DFBETAS 不变。但重新编码可能改变系数本身的含义；例如，把解释变量中心化后，截距变成新原点处的拟合值，其 DFBETAS 可以改变，即使拟合直线不变。比较时应使用同一种参数化，或明确指定关心的系数组合。

<!-- bilingual-en:start -->
## Boundaries and software names
<!-- bilingual-en:end -->

## 解释边界

<!-- bilingual-en:start -->
The full-rank deletion requirement is equivalent to $h_{ii}<1$, where $h_{ii}$ is the full-design [[Hat杠杆值|leverage]]. At $h_{ii}=1$, the deleted coefficients are not uniquely defined. With no positive deleted residual degrees of freedom or with $s_{(i)}=0$, there is no finite standardisation of the stated form, even if the raw coefficient difference is available. A raw change of zero does not make $0/0$ a valid diagnostic.
<!-- bilingual-en:end -->

删点后仍满列秩等价于 $h_{ii}<1$，其中 $h_{ii}$ 是全样本设计的[[Hat杠杆值|杠杆值]]。在 $h_{ii}=1$ 时，删点系数不唯一。若删点残差自由度不为正，或 $s_{(i)}=0$，即使能算出原单位系数变化，也不存在上述形式的有限标准化。原单位变化为零，不会使 $0/0$ 变成有效诊断。

<!-- bilingual-en:start -->
In **R’s `stats` package**, `dfbeta()` returns raw changes and `dfbetas()` returns standardised changes. Software naming is not universal: Stata’s `dfbeta` reports standardised changes. Check the definition, not just the spelling, before using a rule such as $2/\sqrt n$, which concerns standardised absolute DFBETAS and is only a [[影响诊断阈值|screening convention]].
<!-- bilingual-en:end -->

在 **R 的 `stats` 包**中，`dfbeta()` 返回原单位变化，`dfbetas()` 返回标准化变化。软件命名并不统一：Stata 的 `dfbeta` 报告标准化变化。因此，使用 $2/\sqrt n$ 之类规则前，要核对定义而不只看名称；该规则针对标准化后的 DFBETAS 绝对值，且只是[[影响诊断阈值|筛查惯例]]。

<!-- bilingual-en:start -->
DFBETAS does not isolate a causal contribution and is not an automatic deletion rule. [[异常观测处理原则|Investigation]] must distinguish errors, model mismatch and genuine unusual observations. Different models may use approximations; multiple observations may mask each other. For changes at a specified predictor position use [[DFFITS|DFFITS]], and for all fitted values jointly use [[Cook距离|Cook’s distance]].
<!-- bilingual-en:end -->

DFBETAS 不分离因果贡献，也不是自动删点规则。[[异常观测处理原则|调查]]需要区分错误、模型失配和真实特殊观测。其他模型可能使用近似，多个观测也可能互相掩蔽。若看特定解释变量位置处的变化，使用 [[DFFITS]]；若看全部拟合值的整体变化，使用 [[Cook距离|Cook 距离]]。

## 来源与核验

<!-- bilingual-en:start -->
- R Core, [regression deletion diagnostics](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/influence.measures.html) and [official `dfbeta.lm` / `dfbetas.lm` implementation](https://svn.r-project.org/R/trunk/src/library/stats/R/lm.influence.R): raw coefficient differences, deleted residual scales, and the diagonal of the inverse full-design cross-product matrix.
- Stata, [*regress postestimation*, pp. 4 and 34](https://www.stata.com/manuals/rregresspostestimation.pdf#page=34): its standardised `dfbeta` convention and equivalent residualised-predictor formula. The examples, coordinate rescaling, and estimability boundaries are checked by direct OLS calculations.
- University of Virginia Library, [Detecting Influential Points with DFBETA(S)](https://library.virginia.edu/data/articles/detecting-influential-points-in-regression-with-dfbetas), definition and cutoff section: supports the retained size-dependent screening convention, rather than a universal test threshold.
<!-- bilingual-en:end -->

- R Core 上述文档与 `dfbeta.lm`、`dfbetas.lm` 官方实现核对原单位系数变化、删点残差尺度及全设计交叉乘积矩阵逆的对角元。
- Stata 上述第 4、34 页核对其标准化 `dfbeta` 命名及等价的解释变量残差化公式。数值例子、坐标缩放和可估边界由 OLS 直接计算核对。

- University of Virginia Library 上述定义与阈值部分支持所保留的样本量相关筛查惯例，而非通用检验临界值。
