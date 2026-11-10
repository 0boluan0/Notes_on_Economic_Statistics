---
aliases:
  - 高杠杆点
  - Hat values and leverage
  - Regression leverage
  - 杠杆值
student_os: knowledge-atom
atom_id: ECON-SPEC-010
atom_set: regression-influence-diagnostics
atom_type: definition
status: source-checked
part_of:
  - "[[异常值、杠杆与影响诊断.canvas|异常值、杠杆与影响诊断]]"
requires:
  - "[[异常杠杆影响区分]]"
  - "[[满列秩与OLS唯一性]]"
leads_to:
  - "[[Cook距离]]"
  - "[[DFFITS]]"
---

<!-- bilingual-en:start -->
# Hat values and leverage
<!-- bilingual-en:end -->

*Hat 值与杠杆*

<!-- bilingual-en:start -->
A hat value measures **how strongly an observation's own outcome can move its own fitted value**, when the predictor design and unweighted least-squares fitting rule are held fixed. It describes the observation's position among the predictors, before asking whether its actual outcome causes a large change to the fitted result.

For a concrete picture, fit an intercept and slope to $x=(-1,0,1,4)$ and $y=(-1,2,5,14)$. The line is $\hat y=2+3x$. If only the last outcome increases by one, refitting raises the fitted value at $x=4$ by $25/28$. Its [[回归残差|residual]]—observed minus fitted outcome—therefore increases by only $3/28$. The fraction $25/28$ is that observation's **hat value**, also called its **leverage**. The fit follows much of the change at this distant predictor position.
<!-- bilingual-en:end -->

Hat 值衡量的是：保持解释变量设计和不加权最小二乘规则不变，**某条观测自己的结果，能多大程度推动它自己的拟合值**。它先描述观测在解释变量中的位置，再另问实际结果是否真的明显改变了拟合。

看一个具体例子：对 $x=(-1,0,1,4)$、$y=(-1,2,5,14)$ 拟合截距和斜率，得到 $\hat y=2+3x$。若只把最后一个结果增加一，重新拟合后，$x=4$ 处的拟合值增加 $25/28$，它的[[回归残差|残差]]（观测结果减拟合结果）因此只增加 $3/28$。$25/28$ 就是这条观测的 **Hat 值**，也称**杠杆值**；在这个较远的解释变量位置，拟合跟随了相当一部分变化。

<!-- bilingual-en:start -->
## The matrix that puts a hat on the outcomes
<!-- bilingual-en:end -->

*把结果变成拟合值的矩阵*

<!-- bilingual-en:start -->
Take $n\ge1$ observations and $p\ge1$ model terms. Let $X$ be their real $n\times p$ design matrix: each of its $n$ rows describes one observation, and its $p$ columns are the included regression terms. An intercept, if present, is a column of ones and counts in $p$. Assume [[满列秩与OLS唯一性|full column rank]], meaning no column can be made from the others; then $p\le n$ and $X'X$ has an inverse. The vector $y$ contains the $n$ outcomes, and a prime means [[矩阵转置|transpose]].

[[普通最小二乘|OLS]] minimises the sum of squared residuals. Its fitted vector is the [[满列秩投影公式|orthogonal projection]] of $y$ onto the combinations of columns of $X$: the nearest vector the model can fit. The **hat matrix** $H$ performs that operation:
<!-- bilingual-en:end -->

取 $n\ge1$ 条观测和 $p\ge1$ 个模型项，设它们的实数设计矩阵为 $X\in\mathbb R^{n\times p}$：$n$ 行各描述一条观测，$p$ 列是模型中纳入的回归项。若含截距，对应的一列全为一，也计入 $p$。假设[[满列秩与OLS唯一性|满列秩]]，即没有列能由其他列组合出来；于是 $p\le n$，$X'X$ 可逆。向量 $y$ 放入 $n$ 个结果，撇号表示[[矩阵转置|转置]]。

[[普通最小二乘|OLS]]最小化残差平方和，得到的拟合向量是 $y$ 向 $X$ 列组合空间的[[满列秩投影公式|正交投影]]，也就是模型能拟合出的最近向量。**Hat 矩阵** $H$ 完成这个操作：

$$
\hat y=Hy,\qquad H=X(X'X)^{-1}X'.
$$

<!-- bilingual-en:start -->
It puts the hat on $y$. In row $i$, the matrix multiplication reads $\hat y_i=\sum_{j=1}^n h_{ij}y_j$: each outcome $y_j$ contributes with coefficient $h_{ij}$. These coefficients are not probabilities; off-diagonal ones can be negative. The diagonal entry $h_{ii}$ multiplies the observation's **own** outcome. If $x_i'$ is row $i$ of $X$ (so $x_i$ is a column vector), then
<!-- bilingual-en:end -->

它把 $y$ 变成带帽子的 $\hat y$。第 $i$ 行的乘法是 $\hat y_i=\sum_{j=1}^n h_{ij}y_j$：每个结果 $y_j$ 以系数 $h_{ij}$ 参与。这些系数不是概率，非对角项可以为负。对角元 $h_{ii}$ 乘的是该观测**自己的**结果。若 $x_i'$ 是 $X$ 第 $i$ 行，因而 $x_i$ 是列向量，则

$$h_{ii}=x_i'(X'X)^{-1}x_i.$$

<!-- bilingual-en:start -->
Changing only $y_i$ by an amount $\delta$ changes $\hat y_i$ by exactly $h_{ii}\delta$. This sensitivity is a fixed-design algebraic identity; it needs no Gaussian-error assumption. It differs from [[删点影响定义|deleting the observation]], which changes the design as well as the available outcomes. Actual deletion influence depends on the observed response and the target of comparison; see [[异常杠杆影响区分|outliers, leverage, and influence]].
<!-- bilingual-en:end -->

只把 $y_i$ 改变 $\delta$，$\hat y_i$ 就恰好改变 $h_{ii}\delta$。这是固定设计的代数恒等式，不需要 Gaussian 误差假设。它不同于[[删点影响定义|删去整条观测]]：后者同时改变设计和可用结果。实际删点影响还依赖已经观测到的响应及所比较的目标，见[[异常杠杆影响区分]]。

<!-- bilingual-en:start -->
## What the values measure in a fitted line
<!-- bilingual-en:end -->

*在一条拟合直线中算出它*

<!-- bilingual-en:start -->
For an intercept and one nonconstant predictor $x$, the formula simplifies to
<!-- bilingual-en:end -->

含截距、一个非常数解释变量 $x$ 时，公式化简为

$$
h_{ii}=\frac1n+\frac{(x_i-\bar x)^2}{\sum_{j=1}^n(x_j-\bar x)^2},
\qquad \bar x=\frac1n\sum_{j=1}^n x_j.
$$

<!-- bilingual-en:start -->
The sample mean $\bar x$ locates the centre; the denominator measures the total squared spread around it and must be positive. Leverage increases with distance from that centre **relative to the spread of this design**. In the opening example, $n=4$, $\bar x=1$, and the squared spread is $14$:
<!-- bilingual-en:end -->

样本均值 $\bar x$ 定位中心，分母是围绕中心的总平方离散程度，必须为正。杠杆随离中心的距离增加，但距离是**相对于这套设计的离散程度**衡量的。开头例子中 $n=4$、$\bar x=1$、平方离散程度为 $14$：

| $x_i$ | $(x_i-\bar x)^2$ | $h_{ii}$ |
|---|---|---|
| $-1$ | $4$ | $15/28$ |
| $0$ | $1$ | $9/28$ |
| $1$ | $0$ | $7/28$ |
| $4$ | $9$ | $25/28$ |

<!-- bilingual-en:start -->
Their sum is $2$, the number of fitted coefficients. All four values stay the same if any outcomes change. With several predictors, leverage concerns their joint layout, not the largest raw coordinate or the outcome's magnitude. For example, a rare high-income/low-education combination can be unusual relative to a strong income–education relationship even when both coordinates separately lie within familiar ranges.
<!-- bilingual-en:end -->

四个值相加为 $2$，正好是拟合系数个数；无论结果值怎样改变，这四个杠杆都不变。多个解释变量下，应看联合布局，而非哪个原始坐标最大、结果值有多大。例如，收入与教育强相关时，罕见的“高收入、低教育”组合即使两坐标分别都在常见范围内，也可能相对联合关系显得特殊。

> [!note] Range and average / 范围与平均值
> <!-- bilingual-en:start -->
> For this full-column-rank unweighted OLS design,
> <!-- bilingual-en:end -->
>
> 对这里的满列秩、不加权 OLS 设计，
> $$
> 0\le h_{ii}\le1,\qquad \sum_{i=1}^n h_{ii}=p,\qquad \text{average leverage}=p/n.
> $$
> <!-- bilingual-en:start -->
> The [[迹与特征值总和|trace]] is the sum of diagonal entries. A projection keeps the $p$ fitted directions and removes the others, so its trace is $p$. Symmetry and $H^2=H$ give $h_{ii}=\sum_j h_{ij}^2\ge0$; the complementary projection $I-H$ (where $I$ is the identity matrix) gives $1-h_{ii}\ge0$. If the design contains an intercept, the constant direction contributes $1/n$ to every diagonal and the remaining projection contributes a nonnegative amount, hence $h_{ii}\ge1/n$. Without an intercept this stronger lower bound need not hold.
> <!-- bilingual-en:end -->
>
> [[迹与特征值总和|迹]]是对角元之和。投影保留 $p$ 个拟合方向、去掉其余方向，因此迹为 $p$。对称性与 $H^2=H$ 给出 $h_{ii}=\sum_j h_{ij}^2\ge0$；互补投影 $I-H$（$I$ 为单位矩阵）给出 $1-h_{ii}\ge0$。若设计含截距，常数方向为每个对角元贡献 $1/n$，剩余投影再贡献非负量，因此 $h_{ii}\ge1/n$。无截距时，不保证这个更强的下界。

<!-- bilingual-en:start -->
## Conditions and interpretation
<!-- bilingual-en:end -->

## 边界

<!-- bilingual-en:start -->
A large hat value does not prove a data error or large actual influence. A point can have high leverage and lie exactly on the fitted trend. Rules based on a multiple of the average leverage are [[影响诊断阈值|screening conventions]], not part of the definition and not universal significance tests.

At the endpoint $h_{ii}=1$, row $i$ of the hat matrix has zero off-diagonal entries: $h_{ii}=\sum_j h_{ij}^2$ already uses the full sum of squares on its diagonal. Therefore $\hat y_i=y_i$ for every outcome vector and its residual is forced to zero. When $p=n$, $H=I$ and this happens to every row. A perfect training fit in that situation does not demonstrate predictive adequacy. [[学生化残差|Residual diagnostics using $1-h_{ii}$ in a denominator]] need their nonzero-denominator conditions checked.

If $X$ is rank deficient, $(X'X)^{-1}$ does not exist. The fitted vector is still the unique projection onto its column space. Use an independent basis for that space or [[AA+列空间投影|the pseudoinverse projector]] $H=XX^+$, where $X^+$ is the Moore–Penrose pseudoinverse. The diagonal then sums to $\operatorname{rank}(X)$, the number of independent fitted directions, rather than the number of listed columns. Individual coefficients need not be unique. The formulas above describe unweighted OLS; weighted or nonlinear fitting needs its corresponding leverage definition.
<!-- bilingual-en:end -->

Hat 值大，不证明数据错误或实际影响大；高杠杆点可以恰好落在拟合趋势上。平均杠杆若干倍之类规则属于[[影响诊断阈值|筛查惯例]]，不是定义的一部分，也不是通用显著性检验。

端点 $h_{ii}=1$ 时，Hat 矩阵第 $i$ 行的非对角项全为零：在 $h_{ii}=\sum_j h_{ij}^2$ 中，对角项已经用完平方和。因此对任何结果向量都有 $\hat y_i=y_i$，该点残差被强制为零。若 $p=n$，则 $H=I$，每一行都如此。这时训练拟合完美，不代表预测表现充分可靠。[[学生化残差|分母使用 $1-h_{ii}$ 的残差诊断]]必须核对非零分母条件。

若 $X$ 秩亏，$(X'X)^{-1}$ 不存在，但拟合向量仍是到其列空间的唯一投影。可取该空间的一组独立基，或用[[AA+列空间投影|伪逆投影]] $H=XX^+$，其中 $X^+$ 是 Moore–Penrose 伪逆。此时对角和是 $\operatorname{rank}(X)$，即独立拟合方向的个数，而非列出的列数；单个系数未必唯一。以上公式描述不加权 OLS，加权或非线性拟合需要采用相应的杠杆定义。

## 来源与核验

<!-- bilingual-en:start -->
- Penn State STAT 501, [Lesson 11.2](https://online.stat.psu.edu/stat501/Lesson11.html), definition and properties of leverages: supports the hat matrix, diagonal interpretation, design dependence, range and parameter-count sum. The one-predictor formula, four-row calculation and response-perturbation identity are direct consequences of OLS.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.2sum.pdf|MIT 18.06SC Session 2.2 summary]], PDF pp.1–2, supports the orthogonal projection formula, symmetry and idempotence; [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.8sum.pdf|Session 3.8]], PDF pp.1–2, supports the pseudoinverse construction and projection interpretation. The endpoint and intercept-bound arguments are derived from these properties.
- NIST Dataplot, [Regression Diagnostics](https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/regrdiag.htm), opening and leverage discussion: supports separating leverage from influence. Its later “Column 1” hat-matrix display omits an inverse; the correct formula is verified against Penn State and MIT.
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#2.1. OLS|Permanent course section 2.1]] supplies the local matrix-OLS context.
<!-- bilingual-en:end -->

- Penn State 上述 §11.2 定义与性质：支持 Hat 矩阵、对角元含义、设计依赖、范围与参数个数之和；一元公式、四行计算及结果扰动恒等式由 OLS 直接推导。
- MIT 上述 Session 2.2 PDF 第1–2页、Session 3.8 PDF 第1–2页：支持正交投影公式、对称幂等性、伪逆构造与投影解释；端点和含截距下界由这些性质推导。
- NIST 上述开头与杠杆讨论：支持区分杠杆与影响；后文 Column 1 的 Hat 公式漏逆号，正确式已对照 Penn State 与 MIT。
- 永久课程上述 §2.1：保留本地矩阵 OLS 语境。
