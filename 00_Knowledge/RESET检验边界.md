---
aliases:
  - "Ramsey RESET"
  - "RESET specification test"
  - "RESET 检验"
student_os: knowledge-atom
atom_id: ECON-SPEC-007
atom_type: diagnostic
status: source-checked
part_of:
  - "[[回归模型设定与函数形式.canvas|回归模型设定与函数形式]]"
requires:
  - "[[残差图边界]]"
contrasts_with:
  - "[[LM设定检验]]"
---

<!-- bilingual-en:start -->
# Ramsey RESET
<!-- bilingual-en:end -->
*RESET 检验及其解释边界*

<!-- bilingual-en:start -->
A fitted straight line may miss a curve even when we cannot name a particular omitted variable. **Ramsey RESET** uses a few nonlinear transformations of the original fitted values as probes: can these additional directions explain part of the remaining variation? It is a diagnostic question about the specification. The test does not identify a unique replacement model.
<!-- bilingual-en:end -->

拟合的直线可能漏掉曲线，而我们未必能说出具体遗漏了哪个变量。**Ramsey RESET** 用原拟合值的几个非线性变换作探针：这些额外方向，能否解释原来剩下的一部分变化？这是一个关于模型设定的诊断问题；检验不会给出唯一的替代模型。

<!-- bilingual-en:start -->
## What is added to the model?
<!-- bilingual-en:end -->
*模型里添加的究竟是什么？*

<!-- bilingual-en:start -->
Start with [[普通最小二乘|OLS]], which minimizes squared fitting errors. For $n$ observations, let $X$ have $k$ [[线性无关|independent columns]], including an intercept, and obtain $\hat y=X\hat\beta$ from fitting $y$ on $X$. Each $\hat y_i$ is a prediction made from the observed regressors. The common fitted-value version of RESET chooses powers in advance, for example $\hat y_i^2$ and $\hat y_i^3$, and fits the auxiliary model
<!-- bilingual-en:end -->

先用最小化拟合偏差平方和的[[普通最小二乘|OLS]]。有 $n$ 个观测，$X$ 包含 $k$ 个[[线性无关]]的列及截距；把 $y$ 对 $X$ 拟合，得到 $\hat y=X\hat\beta$。每个 $\hat y_i$ 都是用已有回归量作出的预测。常见的拟合值版本 RESET 事先选定幂次，例如 $\hat y_i^2$、$\hat y_i^3$，再拟合辅助模型

$$y_i=x_i^{\mathsf T}b+\gamma_2\hat y_i^2+\gamma_3\hat y_i^3+v_i.$$

<!-- bilingual-en:start -->
Here $x_i^{\mathsf T}$ is row $i$ of $X$; $b$ re-estimates the original coefficients rather than holding them at $\hat\beta$. The term $v_i$ is the error in this augmented auxiliary model; after fitting, its estimated value is the remaining residual. The [[零假设与备择假设|null hypothesis]] sets all additional coefficients to zero. The powers are calculated from the **original** fit and used together with the retained original regressors on the same sample. They are not new measurements of a mechanism: if $\hat y=a+bx$, then $\hat y^2=a^2+2abx+b^2x^2$, a combination of the original terms and a square. With several predictors, products between predictors can also appear.
<!-- bilingual-en:end -->

这里 $x_i^{\mathsf T}$ 是 $X$ 的第 $i$ 行；$b$ 重新估计原有系数，而不是把它们固定在 $\hat\beta$。$v_i$ 是这个扩展辅助模型的误差项；拟合后对它的估计值就是剩余残差。[[零假设与备择假设|零假设]]把所有新增系数设为零。幂项取自**原来的**拟合，在同一样本中与保留的原回归量一起使用。它们不是对某种机制的新测量：若 $\hat y=a+bx$，则 $\hat y^2=a^2+2abx+b^2x^2$，即原项与平方项的组合。有多个解释变量时，还会出现它们的乘积。

<!-- bilingual-en:start -->
Let $Z$ collect the chosen powers. Count the increase in [[矩阵秩|rank]], not just the written columns: $q=\operatorname{rank}([X\ Z])-k$. Remove redundant added columns. If $q=0$, no additional hypothesis remains testable by these probes. Otherwise, require $n>k+q$. Write $RSS_R$ for the original squared [[回归残差|residual]] total and $RSS_U$ for that after augmentation. When $RSS_U>0$, the usual homoskedastic statistic compares the improvement per new direction with the remaining error per residual degree of freedom:
<!-- bilingual-en:end -->

令 $Z$ 收集选定的幂项。应数[[矩阵秩|秩]]的增量，而不只是写了几列：$q=\operatorname{rank}([X\ Z])-k$。去掉冗余新增列后，若 $q=0$，这些探针没有留下可检验的额外假设。否则要求 $n>k+q$。把原[[回归残差|残差]]平方和记为 $RSS_R$，扩展拟合后的记为 $RSS_U$。当 $RSS_U>0$ 时，常见同方差统计量比较“每个新增方向带来的改善”与“每个剩余自由度承受的误差”：

$$F=\frac{(RSS_R-RSS_U)/q}{RSS_U/(n-k-q)}.$$

> [!example] Two powers need not add two directions · 两个幂项未必新增两个方向
> <!-- bilingual-en:start -->
> Use $x=(0,0,1,1,2,2)$ and $y=(11,9,16,14,27,25)$. The original fit is $\hat y=9+8x$, with $RSS_R=18$. Since $\hat y^2=81+144x+64x^2$, adding its square is equivalent to adding $x^2$. The expanded residual total is $6$, so $k=2$, $q=1$ and $F=6$. Adding $\hat y^3$ as well creates no second direction here: with only three distinct $x$ values, the columns $1,x,x^2$ already represent every function on those three values. The rank remains $3$.
> <!-- bilingual-en:end -->
>
> 取 $x=(0,0,1,1,2,2)$、$y=(11,9,16,14,27,25)$。原拟合为 $\hat y=9+8x$，$RSS_R=18$。由于 $\hat y^2=81+144x+64x^2$，加入拟合值平方等价于加入 $x^2$。扩展残差平方和为 $6$，所以 $k=2$、$q=1$、$F=6$。此时再加 $\hat y^3$ 并不产生第二个新方向：$x$ 只有三个不同值，$1,x,x^2$ 已能表示这三个值上的任意函数，秩仍为 $3$。

<!-- bilingual-en:start -->
## The reference distribution has conditions
<!-- bilingual-en:end -->
*参考分布需要条件*

<!-- bilingual-en:start -->
The formula alone does not guarantee an exact [[F分布|F distribution]]. A sufficient finite-sample setting for this fitted-power version is a fixed full-rank $X$, a correct null mean $X\beta$, independent $N(0,\sigma^2)$ errors with $\sigma^2>0$, and a power list chosen in advance, with no further selection using the residuals. Conditional on the original fitted vector and a positive effective rank $q<n-k$, the statistic has reference $F_{q,n-k-q}$.
<!-- bilingual-en:end -->

单凭公式不能保证精确的 [[F分布|F 分布]]。对这个拟合值幂版本，一组有限样本充分条件是：固定且满列秩的 $X$、正确的零假设均值 $X\beta$、独立 $N(0,\sigma^2)$ 误差且 $\sigma^2>0$，以及事先选定的幂次列表，没有再根据残差选择探针。在给定原拟合向量、有效秩满足 $0<q<n-k$ 时，统计量的参考分布是 $F_{q,n-k-q}$。

> [!note]- Why generated fitted values can work here · 为什么这里可以使用样本生成的拟合值？
> <!-- bilingual-en:start -->
> Under these Gaussian conditions, the original fitted vector and original residual vector are independent. Once the fitted vector is held fixed, the power probes are fixed too. Within the original residual space, the improvement and the remaining residual are projections onto orthogonal spaces of dimensions $q$ and $n-k-q$. Their squared lengths divided by $\sigma^2$ are independent [[卡方分布|chi-square variables]], giving the F ratio. Selecting probes after inspecting those residuals would break this argument. The point is the specific independence structure, not a general permission to treat every data-generated regressor as fixed.
> <!-- bilingual-en:end -->
>
> 在这些 Gaussian 条件下，原拟合向量与原残差向量独立。固定拟合向量后，幂项探针也固定了。在原残差空间内，改善部分与剩余残差分别是向两个正交空间的投影，维数为 $q$ 和 $n-k-q$。它们的平方长度除以 $\sigma^2$ 是独立的[[卡方分布|卡方变量]]，从而给出 F 比值。若看过这些残差才选择探针，这个论证就会失效。关键是特定独立结构，不是可以把所有数据生成的回归量都视为固定。

<!-- bilingual-en:start -->
In large samples, a [[Wald联合检验|joint Wald test]] can use a [[卡方分布|chi-square]] reference when the chosen augmented regressors have nondegenerate limiting variation, the tested coefficient estimates have a valid asymptotically normal limit, and their covariance estimate is consistent under the sampling model. [[异方差|Heteroskedasticity]] or [[聚类标准误边界|cluster dependence]] requires an appropriate robust procedure and its own regularity conditions. A software option requesting an F statistic or a robust covariance does not by itself establish exact finite-sample calibration.
<!-- bilingual-en:end -->

大样本下，若所选扩展回归量有非退化的极限变化，受检验系数估计具有有效的渐近正态极限，且其协方差估计在抽样模型下具有一致性，[[Wald联合检验|Wald 联合检验]]可以使用[[卡方分布|卡方]]参考。[[异方差]]或[[聚类标准误边界|簇内相关]]需要相应的稳健程序及其正则条件。软件中选择 F 统计量或稳健协方差的选项，本身并不能确立精确的有限样本校准。

<!-- bilingual-en:start -->
## Interpreting rejection and non-rejection
<!-- bilingual-en:end -->
## 拒绝与不拒绝分别说明什么

<!-- bilingual-en:start -->
With a valid calibration and the other maintained assumptions, rejection is evidence against the original mean specification along the selected probes. An omitted square, an interaction, a transformation or another structure associated with the probes could explain it. If the assumed error dependence or variance model is wrong, the conventional calibration itself may be wrong. A small [[p值条件解释|p-value]] therefore does not uniquely diagnose “the functional form is wrong,” much less name the missing variable. Separate the [[设定问题分层|mean, uncertainty and causal-identification layers]] before choosing a remedy.
<!-- bilingual-en:end -->

在校准有效、其余维持假设成立时，拒绝是在所选探针方向上反对原均值设定的证据。遗漏平方项、交互项、变量变换，或其他与探针有关的结构，都可能解释它。若假定的误差相关或方差模型不对，常规校准本身也可能出错。因此，小 [[p值条件解释|p 值]]不能唯一诊断“函数形式错误”，更不能指出究竟漏了哪个变量。选补救方法前，应先区分[[设定问题分层|均值、不确定性和因果识别的不同层次]]。

<!-- bilingual-en:start -->
Non-rejection is insufficient evidence against these probes, not certification of the model. [[检验功效|Power]] may be low, or the chosen directions may miss the relevant departure. For a visible limiting case, take $x=(-1,-1,0,0,1,1)$ and noiseless $y=x^2$. A straight-line fit gives the constant $\hat y=2/3$. Every fitted-value power is then another constant, already represented by the intercept: $q=0$. This RESET is **not a valid “pass”**; it has no testable added direction even though a targeted $x^2$ term describes the curvature exactly.
<!-- bilingual-en:end -->

不拒绝只表示没有足够证据反对这些探针，不是给模型发合格证。[[检验功效|功效]]可能较低，所选方向也可能没碰到真正的偏离。一个直观的极端例子是：取 $x=(-1,-1,0,0,1,1)$，无噪声结果 $y=x^2$。直线拟合给出常数 $\hat y=2/3$，其任意幂仍是常数，已被截距表示，故 $q=0$。这次 RESET **不能算“通过”**：它根本没有可检验的新增方向，尽管明确加入 $x^2$ 就能精确描述曲率。

<!-- bilingual-en:start -->
After rejection, automatically retaining $\hat y^2$ does not establish a unique substantive explanation. Return to the original variables and research mechanism, propose interpretable terms, and validate the revision. Neither rejection nor non-rejection establishes a causal effect.
<!-- bilingual-en:end -->

拒绝后自动把 $\hat y^2$ 留进最终模型，并不能确立唯一的实质解释。应回到原始变量和研究机制，提出可解释的项，再验证修订。拒绝和不拒绝都不能确立因果效应。

<!-- bilingual-en:start -->
## Continue
<!-- bilingual-en:end -->
## 继续

- [[LM设定检验]]：已有具体候选方向时，直接检验那些方向。
- [[设定修复循环]]：把诊断转成有依据的模型修订和验证。

## 来源与核验

<!-- bilingual-en:start -->
- [Hansen, Econometrics (2010)](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf), §7.7, PDF pp. 121–122: fitted-value powers chosen in advance and the asymptotic Wald formulation. §4.13, pp. 68–70 supplies Gaussian fitted-value/residual independence; the conditional F argument above combines that fact with orthogonal projections and the F definition.
- [Stata, regress postestimation](https://www.stata.com/manuals/rregresspostestimation.pdf), PDF p. 34, `estat ovtest`: fitted-value-power and regressor-power implementations. [statsmodels, linear_reset documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_reset.html) and [implementation](https://www.statsmodels.org/stable/_modules/statsmodels/stats/diagnostic.html): augmentation choices and covariance/test options. These options do not replace rank and calibration checks.
- [[02_Economy/01_Econometrics/04_模型设定.md#3.3. 拉姆齐 RESET 检验|Course §3.3]] supports course scope and the joint zero restriction. Both small-data examples are direct calculations.
- Historical reference: Ramsey (1969), [Tests for Specification Errors in Classical Linear Least-Squares Regression Analysis](https://doi.org/10.1111/j.2517-6161.1969.tb00796.x). Further reading: Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 9. Neither full text was reopened for this verification; they do not serve as the verification evidence above.
<!-- bilingual-en:end -->

- Hansen §7.7 支持事先选定拟合值幂与渐近 Wald 表述；§4.13 支持 Gaussian 拟合值与残差独立，上面的条件 F 论证另结合正交投影和 F 分布定义直接得出。
- Stata `estat ovtest` 与 statsmodels 文档、实际实现支持不同探针及协方差选项；软件选项不能替代有效秩和校准检查。
- 永久课程 §3.3 保留课程范围与联合零限制；两个小数据例直接计算核对。Ramsey 原论文保留为历史出处，Wooldridge 保留为延伸阅读；本次未重开二者全文，不把它们当作上述核验的证据。
