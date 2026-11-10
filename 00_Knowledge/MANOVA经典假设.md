---
aliases:
  - "Classical MANOVA assumptions"
  - "MANOVA assumption boundary"
student_os: knowledge-atom
atom_id: STAT-MAN-005
atom_set: manova
atom_type: assumption-boundary
status: source-checked
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
requires:
  - "[[MANOVA]]"
  - "[[矩阵正态误差结构]]"
  - "[[Wishart抽样假设]]"
related:
  - "[[误差SSP秩边界]]"
  - "[[相关行不容独立推断]]"
  - "[[多响应联合显著边界]]"
  - "[[多响应因果边界]]"
---

<!-- bilingual-en:start -->
# Classical MANOVA assumptions
<!-- bilingual-en:end -->

*MANOVA 的经典假设*

<!-- bilingual-en:start -->
A [[MANOVA|MANOVA]] calculation compares variation explained by a specified effect with variation left within the observations. The **classical assumptions** say what kind of repeated-sampling experiment gives that comparison its usual probability distribution. A table of means and cross-products can be calculated without these assumptions; interpreting its test statistic using a classical reference law needs more.
<!-- bilingual-en:end -->

[[MANOVA]] 比较指定效应解释的变动与观测中留下的变动。**经典假设**描述的是：在怎样的重复抽样实验下，这种比较具有通常采用的概率分布。均值与交叉乘积表可以先算出来，但要用经典参考分布解释检验统计量，还需要模型依据。

<!-- bilingual-en:start -->
This card describes a standard sufficient model for classical multivariate linear-model inference. It does not claim that every method called MANOVA requires these assumptions: robust, resampling, heterogeneous-covariance, repeated-measurement, and high-dimensional methods must justify their own models and calibration.
<!-- bilingual-en:end -->

本卡解释经典多响应线性模型推断的一套标准充分条件，并没有声称所有称为 MANOVA 的方法都必须满足它们。稳健、重抽样、异质协方差、重复测量或高维方法，需要各自说明模型与校准依据。

<!-- bilingual-en:start -->
## What is random, and what is shared?
<!-- bilingual-en:end -->

*什么在随机变化，什么被各组共享？*

<!-- bilingual-en:start -->
For one-way group comparison, $Y_{ij}$ is the $p$-coordinate response vector for unit $j$ in group $i$. Its [[均值向量|mean vector]] $\mu_i$ can vary with the group. The classical model is
<!-- bilingual-en:end -->

单因素组间比较中，$Y_{ij}$ 是第 $i$ 组第 $j$ 个单位的 $p$ 坐标响应向量，[[均值向量]] $\mu_i$ 可以随组改变。经典模型为

$$Y_{ij}=\mu_i+\varepsilon_{ij},\qquad
\{\varepsilon_{ij}\}\ \text{independent},\qquad
\varepsilon_{ij}\sim N_p(0,\Sigma),\qquad \Sigma\succ0.$$

<!-- bilingual-en:start -->
The notation $N_p(0,\Sigma)$ means a [[多元正态分布|joint Gaussian distribution]] with zero mean and [[协方差矩阵|covariance matrix]] $\Sigma$. Positive definiteness, $\Sigma\succ0$, means that every nonzero response combination has positive variance; see [[正定矩阵|positive-definite matrices]]. The errors are independent and identically distributed. The observations themselves can have different means across groups, so they need not be identically distributed.
<!-- bilingual-en:end -->

$N_p(0,\Sigma)$ 表示均值为零、[[协方差矩阵]]为 $\Sigma$ 的[[多元正态分布|联合 Gaussian 分布]]。正定条件 $\Sigma\succ0$ 表示每个非零响应组合的方差都为正，见[[正定矩阵]]。这里独立同分布的是误差；观测值可以具有不同的组均值，因此不必同分布。

<!-- bilingual-en:start -->
In the broader [[多响应线性回归|multivariate linear model]] $Y=XB+U$, treat the design $X$ as fixed, or state the model conditional on $X$. The mean must be $XB$, and the rows of $U$ must have this independent, common-covariance Gaussian law. This is the independent-row case of [[矩阵正态误差结构|matrix-normal errors]]. A relevant predictor omitted from the mean or a dependence ignored by the model is not repaired by fitting more response columns.
<!-- bilingual-en:end -->

在更一般的[[多响应线性回归|多响应线性模型]] $Y=XB+U$ 中，把设计 $X$ 看作固定，或明确给定 $X$ 后的条件模型。均值必须为 $XB$，$U$ 的各行须具有上述独立、共同协方差的 Gaussian 分布。这是[[矩阵正态误差结构]]中的独立行特例。遗漏应进入均值的解释变量，或忽略模型中的依赖，不会因为增加响应列而自动修复。

<!-- bilingual-en:start -->
## Independence concerns units, normality concerns the whole vector
<!-- bilingual-en:end -->

*独立性针对单位，正态性针对整个向量*

<!-- bilingual-en:start -->
A student's mathematics and reading errors may be correlated; that is exactly what the off-diagonal entries of $\Sigma$ describe. Independence is required between distinct observational units in this model. If the same student's repeated measurements are put in separate rows, or students share unmodelled classroom effects, those rows are generally dependent. The study design must justify the row model; see [[相关行不容独立推断|the inference boundary for dependent rows]].
<!-- bilingual-en:end -->

同一名学生的数学与阅读误差可以相关，这正是 $\Sigma$ 非对角元素描述的内容。这个模型要求不同观测单位之间独立。若把同一学生的重复测量放在不同行，或学生共享未建模的班级效应，这些行通常就相关。行模型需要研究设计的支持，见[[相关行不容独立推断]]。

<!-- bilingual-en:start -->
For example, three visits by one student are not automatically three independent students. A possible multivariate repeated-measurement design instead puts that student's measurements in one response vector and uses independent students as rows. Its mean contrasts and within-person covariance still need to be specified; changing the arrangement alone does not establish the assumptions.
<!-- bilingual-en:end -->

例如，同一学生来三次，并不会自动变成三个独立学生。一种多变量重复测量设计可把该学生的测量放进同一个响应向量，以独立学生作为各行；其中的均值对比与个体内协方差仍须明确。仅改变排列方式不能证明假设成立。

<!-- bilingual-en:start -->
Normality is also a joint statement: within each group, every fixed linear combination of the responses must be Gaussian. Separate bell-shaped histograms do not establish this; see [[边际与马氏图不证联合Gaussian|the limits of marginal and Mahalanobis plots]]. Conversely, pooling groups with different means can produce a non-Gaussian mixture even when every group's error vector is Gaussian. Diagnose variation around the appropriate fitted mean, not only the histogram of the pooled raw responses.
<!-- bilingual-en:end -->

正态性也是联合命题：每组内，响应的任意固定线性组合都须为 Gaussian。分别画出钟形直方图不能证明这一点，见[[边际与马氏图不证联合Gaussian]]。反过来，把均值不同的组混在一起，即使各组的误差向量都为 Gaussian，混合后的总体也可能非 Gaussian。诊断应围绕正确拟合的均值考察误差，而非只看合并原始响应的直方图。

<!-- bilingual-en:start -->
## A common covariance includes cross-response variation
<!-- bilingual-en:end -->

*共同协方差也包含响应之间的共同变化*

<!-- bilingual-en:start -->
Common covariance means the whole within-group matrix is the same: $\Sigma_1=\cdots=\Sigma_g=\Sigma$. It does not require the group means to agree, and it is stronger than equal variances considered one response at a time.
<!-- bilingual-en:end -->

共同协方差要求整个组内矩阵相同，即 $\Sigma_1=\cdots=\Sigma_g=\Sigma$。它不要求组均值相等，也强于逐个响应检查方差相等。

> [!example] Equal variances, different covariance · 方差相同，协方差不同
> <!-- bilingual-en:start -->
> These two positive-definite covariance matrices have the same diagonal entries:
> <!-- bilingual-en:end -->
>
> 下面两个正定协方差矩阵的对角元素完全相同：
>
> $$\Sigma_1=\begin{bmatrix}1&0.8\\0.8&1\end{bmatrix},\qquad
> \Sigma_2=\begin{bmatrix}1&-0.8\\-0.8&1\end{bmatrix}.$$
>
> <!-- bilingual-en:start -->
> Each individual response has variance $1$ in either group. But the difference of the two responses has variance $1+1-2(0.8)=0.4$ in group 1 and $1+1-2(-0.8)=3.6$ in group 2, by [[线性投影方差|the variance of a linear combination]]. The same observed mean difference along this direction therefore faces different noise. Pooling these matrices as though a common covariance had been established loses a condition used by the classical law.
> <!-- bilingual-en:end -->
>
> 两组中每个响应的方差都是 $1$；但根据[[线性投影方差|线性组合的方差公式]]，两响应之差在第 1 组的方差为 $1+1-2(0.8)=0.4$，第 2 组为 $1+1-2(-0.8)=3.6$。因此沿这个方向，相同的均值差面对不同的噪声。把它们当作已满足共同协方差条件而直接合并，会丢掉经典抽样律使用的一项前提。

<!-- bilingual-en:start -->
## Which coefficient comparisons can the design identify?
<!-- bilingual-en:end -->

*设计究竟能识别哪些系数对比？*

<!-- bilingual-en:start -->
A coefficient label is not automatically an identifiable target. In a one-way model written $\mu+\tau_i$, replacing $\mu$ by $\mu+c$ and all $\tau_i$ by $\tau_i-c$ leaves every group mean unchanged. Unconstrained data cannot distinguish those coefficient representations. The differences $\tau_i-\tau_j$, however, stay unchanged and are meaningful group comparisons. Constraints can choose a representation without changing those comparisons.
<!-- bilingual-en:end -->

有了系数名称，不等于该系数就可识别。把单因素模型写成 $\mu+\tau_i$ 时，将 $\mu$ 改为 $\mu+c$，同时把所有 $\tau_i$ 改为 $\tau_i-c$，各组均值完全不变。没有约束时，数据无法区分这些系数表示；但 $\tau_i-\tau_j$ 保持不变，仍是有意义的组间比较。约束可以选定表示，而不改变这些对比。

<!-- bilingual-en:start -->
For $Y=XB+U$, a coefficient contrast $CB$ is **estimable** when it can be recovered as a linear function of the model mean $XB$. Concretely, there must be a matrix $K$ with $C=KX$, so $CB=K(XB)$. Equivalently, each row of $C$ belongs to the [[行空间|row space of $X$]], the linear combinations of its rows. Full column rank makes every coefficient contrast estimable, but redundant design coding does not destroy all estimable comparisons. The requested [[多响应一般线性假设|coefficient and response hypothesis]] must be checked on its actual estimable space.
<!-- bilingual-en:end -->

对 $Y=XB+U$，若系数对比 $CB$ 能由模型均值 $XB$ 的线性函数恢复，就称它**可估**。具体要求存在矩阵 $K$ 使 $C=KX$，于是 $CB=K(XB)$。等价地，$C$ 的各行属于 $X$ 的[[行空间]]，即 $X$ 各行的线性组合所成空间。满列秩保证所有系数对比都可估，但冗余编码不会毁掉所有可估对比；应在实际可估空间中检查所问的[[多响应一般线性假设|系数与响应假设]]。

<!-- bilingual-en:start -->
## Are there enough residual directions?
<!-- bilingual-en:end -->

*残差还有足够多的独立方向吗？*

<!-- bilingual-en:start -->
Let $N$ be the number of rows and $r=\operatorname{rank}(X)$ the number of independent design directions, as measured by [[矩阵秩|matrix rank]]. Fitting the mean uses $r$ directions, leaving residual degrees of freedom $\nu=N-r$. If all $g$ groups are observed, the one-way group-indicator design has $r=g$. Adding an intercept to all $g$ indicators adds a redundant column; it does not change the rank to $g+1$.
<!-- bilingual-en:end -->

令 $N$ 为行数，$r=\operatorname{rank}(X)$ 为独立设计方向的个数，即设计矩阵的[[矩阵秩|秩]]。拟合均值消耗 $r$ 个方向，留下残差自由度 $\nu=N-r$。若 $g$ 组都有观测，单因素组别指示设计的秩为 $r=g$；在全部 $g$ 个指示量之外再加截距，只增加一列冗余列，不会使秩变成 $g+1$。

<!-- bilingual-en:start -->
Let $\hat B$ be a least-squares coefficient estimate. The [[多响应残差与误差SSP|residual matrix]] $R=Y-X\hat B$ is $N\times p$. Its error SSCP $E=R^TR$ is $p\times p$. Residual columns lie in a space of dimension $\nu$, so
<!-- bilingual-en:end -->

令 $\hat B$ 为最小二乘系数估计。[[多响应残差与误差SSP|残差矩阵]] $R=Y-X\hat B$ 为 $N\times p$，误差 SSCP $E=R^TR$ 为 $p\times p$。残差列只能落在一个 $\nu$ 维空间中，因此

$$\operatorname{rank}(E)=\operatorname{rank}(R)\le\min(p,\nu).$$

<!-- bilingual-en:start -->
If $\nu=0$, every residual is zero and $E=0$. For $\nu\ge1$, under the stated independent Gaussian model with positive-definite $\Sigma$, $E$ has the [[误差SSP抽样律|Wishart residual law]] with $\nu$ degrees of freedom. It reaches rank $\min(p,\nu)$ with probability one; in particular, when $\nu\ge p$, it is positive definite with probability one. The deterministic dimension bound and this model-based probability-one statement are different claims; see [[Wishart秩与可逆性|Wishart rank and invertibility]]. If observed residual responses are exactly collinear, their actual $E$ is singular and cannot be inverted, regardless of the intended model.
<!-- bilingual-en:end -->

若 $\nu=0$，所有残差均为零，$E=0$。当 $\nu\ge1$ 时，在上述独立 Gaussian 且 $\Sigma$ 正定的模型下，$E$ 具有自由度为 $\nu$ 的[[误差SSP抽样律|残差 Wishart 律]]，以概率一达到秩 $\min(p,\nu)$；特别地，$\nu\ge p$ 时以概率一正定。确定性的维数上界与依赖概率模型的“概率一”结论是不同命题，见[[Wishart秩与可逆性]]。若实测残差响应精确共线，实际 $E$ 仍然奇异，不能因为原本打算采用某个模型就把它求逆。

> [!warning] Positive residual degrees of freedom may still be insufficient · 自由度为正仍可能不够
> <!-- bilingual-en:start -->
> With $N=5$, $g=3$, and $p=3$, only $\nu=2$ residual directions remain: the $3\times3$ matrix $E$ must be singular. With $N=9$, $g=3$, and $p=2$, $\nu=6$ clears the dimension threshold; the Gaussian positive-definite model supplies the further probability-one conclusion. The full-response requirement concerns pooled residual degrees of freedom, not a rule that every group must contain more than $p$ units. A prespecified lower-dimensional response hypothesis has its own effective dimension and rank check. A pseudoinverse or regularisation does not automatically retain the original full-response reference law; see [[误差SSP秩边界|the error-SSCP rank boundary]].
> <!-- bilingual-en:end -->
>
> 当 $N=5$、$g=3$、$p=3$ 时，只剩 $\nu=2$ 个残差方向，$3\times3$ 的 $E$ 必然奇异。若 $N=9$、$g=3$、$p=2$，则 $\nu=6$ 越过维数门槛，Gaussian 正定模型进一步给出概率一可逆结论。全响应条件看的是合并后的残差自由度，并非要求每组都必须多于 $p$ 个单位。预先选定的低维响应假设要检查它自己的有效维数与秩；使用伪逆或正则化不会自动保留原全响应参考分布，见[[误差SSP秩边界]]。

<!-- bilingual-en:start -->
## What model checking can establish
<!-- bilingual-en:end -->

*模型检查能提供什么结论？*

<!-- bilingual-en:start -->
Residual plots, scatterplots, and outlier checks can expose conflicts with the model; they cannot prove the population assumptions from a finite sample. Fitted residuals also obey constraints, such as summing to zero within each fitted group, so they should not be treated as a fresh independent sample of the unobserved errors. [[联合Gaussian诊断|Joint Gaussian diagnostics]] explain the difference between exploratory checks and calibrated evidence.
<!-- bilingual-en:end -->

残差图、散点图与异常值检查能够暴露模型冲突，却不能凭有限样本证明总体假设。拟合残差还满足约束，例如每个拟合组内残差和为零，不能把它们当作一份新抽到的独立真实误差样本。[[联合Gaussian诊断]]进一步区分探索性检查与经过校准的证据。

<!-- bilingual-en:start -->
A covariance-homogeneity test such as Box's $M$ also has assumptions and limited detection power. Failure to reject is not proof that the covariance matrices agree. If a model is inappropriate, choose a method justified for the actual design and state its target and calibration. Even within the Gaussian model, not every software-reported $F$ conversion is exact; see [[MANOVA检验校准|MANOVA calibration]]. Neither these assumptions nor a significant test establish random treatment assignment or remove confounding; that is the separate [[多响应因果边界|causal-inference boundary]].
<!-- bilingual-en:end -->

Box's $M$ 这类协方差齐性检验本身也有假设和有限的检出能力；未拒绝不能证明协方差相同。模型不合适时，应选择有真实设计依据的方法，并明确它检验什么、如何校准。即便在 Gaussian 模型内，软件报告的 $F$ 换算也并非一律精确，见[[MANOVA检验校准]]。这些假设或显著结果同样不会创造随机分配或消除混杂，这属于独立的[[多响应因果边界]]。

## 来源与核验

<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 8, §§8.2, 8.4](https://online.stat.psu.edu/stat505/Lesson08) supports the independent-unit, within-group normal, common-covariance model and the role of residual and covariance diagnostics.
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm) specifies independent observation vectors with correlated responses and residual degrees of freedom $N-\operatorname{rank}(X)$. [SAS, Estimable Functions](https://support.sas.com/documentation/cdl/en/statug/66859/HTML/default/statug_intromod_sect042.htm) gives the row-space criterion $C=KX$.
- [Stanley Sawyer, Washington University, *Multivariate Linear Models*, §§1, 3, 6, pp. 1, 3, 7](https://www.math.wustl.edu/~sawyer/handouts/manovas.pdf) supports independent Gaussian errors and the residual Wishart law. The residual SSCP is the $p\times p$ product $R^TR$ as in equation (3.2b); the later transposition typo is not used. [[误差SSP秩边界|The residual rank result]] and [[Wishart秩与可逆性|Wishart rank]] supply the deterministic and probability-one rank arguments.
- [R, `summary.manova`, Details](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html) distinguishes approximate $F$ conversions and checks for residual response rank deficiency. The numerical covariance and design examples above are direct calculations.
<!-- bilingual-en:end -->

- Penn State §§8.2、8.4 支持独立单位、组内联合正态、共同协方差模型及残差、协方差诊断的用途。
- SAS MANOVA 文档明确区分观测间独立与响应间相关，并使用 $N-\operatorname{rank}(X)$ 的残差自由度；其可估函数文档给出 $C=KX$ 的行空间判据。
- Washington University 讲义支持独立 Gaussian 误差与残差 Wishart 律；SSCP 按式 (3.2b) 使用 $p\times p$ 的 $R^TR$，不沿用后文的转置笔误。残差秩与 Wishart 秩条目提供确定性及概率一论证。
- R 官方文档支持近似 $F$ 换算与残差响应秩检查；协方差、设计与自由度算例均直接计算。
