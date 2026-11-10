---
aliases:
  - "Multivariate Analysis of Variance"
  - "多元方差分析"
student_os: knowledge-atom
atom_id: STAT-MAN-001
atom_set: manova
atom_type: definition
status: source-checked
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
requires:
  - "[[多响应线性回归]]"
  - "[[多响应一般线性假设]]"
related:
  - "[[多响应与多解释变量回归]]"
  - "[[pooled Hotelling T²]]"
  - "[[多响应联合显著边界]]"
  - "[[多响应因果边界]]"
---

<!-- bilingual-en:start -->
# MANOVA
<!-- bilingual-en:end -->

*Multivariate analysis of variance · 多元方差分析*

<!-- bilingual-en:start -->
**MANOVA asks whether a specified effect changes a collection of response means jointly.** Suppose three teaching programmes are compared using mathematics and reading scores from each student. We want to compare the programmes' two-coordinate mean profiles while accounting for how the scores vary together. Two separate tests, one per subject, do not by themselves perform this joint test.
<!-- bilingual-en:end -->

**MANOVA 联合检验一个指定效应是否改变一组响应的均值。** 例如，比较三种教学方案，每名学生同时记录数学与阅读成绩。我们想比较三组的二坐标平均成绩，同时考虑两科怎样共同波动。分别做一次数学检验和一次阅读检验，并不会自动构成这个联合检验。

<!-- bilingual-en:start -->
A [[均值向量|mean vector]] keeps the population mean of each response in its own coordinate. A [[协方差矩阵|covariance matrix]] describes both the spread of each response and their joint variation. MANOVA uses both: the means specify the question, and the variation determines how much evidence a difference supplies. Group comparison is the entry point below; the same framework also handles specified effects in a shared-design [[多响应线性回归|multivariate linear model]].
<!-- bilingual-en:end -->

[[均值向量]]把每个响应的总体均值放在各自坐标中；[[协方差矩阵]]同时描述各响应的分散程度和共同变化。MANOVA 两者都用：均值确定所问的问题，波动决定观察到的差异有多强的证据。下面从组间比较入手，同一框架还可检验共享设计的[[多响应线性回归|多响应线性模型]]中的指定效应。

<!-- bilingual-en:start -->
## What does equality of group means mean?
<!-- bilingual-en:end -->

*组均值相等具体指什么？*

<!-- bilingual-en:start -->
Let $g\ge2$ be the number of groups and $p\ge1$ the number of responses recorded for each unit. Group $i$ has $n_i\ge1$ observations. The column vector $Y_{ij}\in\mathbb R^p$ records the $j$th unit in group $i$; $\mu_i$ is its population mean vector and $\varepsilon_{ij}$ its deviation from that mean. The one-way model and hypothesis are
<!-- bilingual-en:end -->

令 $g\ge2$ 为组数，$p\ge1$ 为每个单位记录的响应数，第 $i$ 组有 $n_i\ge1$ 个观测。列向量 $Y_{ij}\in\mathbb R^p$ 记录第 $i$ 组第 $j$ 个单位，$\mu_i$ 为该组的总体均值向量，$\varepsilon_{ij}$ 为偏离该均值的误差。单因素模型与假设为

$$Y_{ij}=\mu_i+\varepsilon_{ij},\qquad E[\varepsilon_{ij}]=0,$$
$$H_0:\mu_1=\mu_2=\cdots=\mu_g.$$

<!-- bilingual-en:start -->
Here $E[·]$ denotes [[期望|expectation]], the model average. Equality here means equality in **every** response coordinate. The alternative is that at least one pair of group mean vectors differs. For three programmes and three scores, such as mathematics, reading, and writing, the hypothesis compares three three-coordinate profiles. It does not claim that all observed students have equal scores, or that their within-group variation is zero. The classical probability model used to calibrate the test adds conditions on the errors; those conditions are explained in [[MANOVA经典假设|classical MANOVA assumptions]].
<!-- bilingual-en:end -->

这里 $E[·]$ 表示[[期望]]，即模型中的平均。相等要求**每一个**响应坐标都相等；备择则是至少一对组均值向量不同。若有三种方案以及数学、阅读、写作三科，检验比较的就是三个三坐标平均档案。它没有声称每名学生的实测成绩相同，也没有声称组内波动为零。把统计量换成检验结论还需要误差的概率模型，见[[MANOVA经典假设]]。

<!-- bilingual-en:start -->
## Why compare between-group and within-group variation?
<!-- bilingual-en:end -->

*为什么要同时看组间与组内变动？*

<!-- bilingual-en:start -->
The [[样本均值向量|sample mean vector]] $\bar y_i$ averages each response across the units in group $i$. Let $N=\sum_i n_i$ and let $\bar y=N^{-1}\sum_i n_i\bar y_i$ be the overall sample mean. Write $y_{ij}$ for the observed value of $Y_{ij}$. Two kinds of deviations matter: $y_{ij}-\bar y_i$ describes scatter around a group's own centre; $\bar y_i-\bar y$ describes the displacement of that centre from the overall centre. Their outer products collect squares on the diagonal and cross-products off the diagonal:
<!-- bilingual-en:end -->

[[样本均值向量]] $\bar y_i$ 沿第 $i$ 组的观测逐坐标取平均。令 $N=\sum_i n_i$，总样本均值为 $\bar y=N^{-1}\sum_i n_i\bar y_i$。用 $y_{ij}$ 表示 $Y_{ij}$ 的实测值。需要区分两种偏离：$y_{ij}-\bar y_i$ 描述围绕本组中心的散布；$\bar y_i-\bar y$ 描述组中心偏离总中心多少。它们的外积在对角线上记录平方，在非对角线上记录交叉乘积：

$$E=\sum_{i=1}^g\sum_{j=1}^{n_i}(y_{ij}-\bar y_i)(y_{ij}-\bar y_i)^T,\qquad
H=\sum_{i=1}^g n_i(\bar y_i-\bar y)(\bar y_i-\bar y)^T.$$

<!-- bilingual-en:start -->
Here $T$ denotes [[矩阵转置|transpose]]. Both $E$ and $H$ are $p\times p$ **sums-of-squares-and-cross-products matrices**, abbreviated SSCP: $E$ records within-group error variation, and $H$ records between-group variation for this hypothesis. They are unnormalised sums, not covariance estimates. The [[MANOVA SSCP分解|SSCP decomposition]] explains their entries and why total scatter is $E+H$.
<!-- bilingual-en:end -->

上标 $T$ 表示[[矩阵转置|转置]]。$E$ 与 $H$ 都是 $p\times p$ 的**平方和与交叉乘积矩阵**，简称 SSCP；$E$ 记录组内误差变动，$H$ 记录该假设对应的组间变动。它们是尚未除以自由度的和，不是协方差估计量。各元素的意义与总散布为何等于 $E+H$，见[[MANOVA SSCP分解]]。

> [!example] Three groups, two responses · 三组、两个响应
> <!-- bilingual-en:start -->
> Consider these nine complete observations. Each ordered pair is one student's mathematics and reading scores; the numbers are an illustration, not evidence that the statistical assumptions hold.
> <!-- bilingual-en:end -->
>
> 考虑下面九个完整观测。每个有序对是一名学生的数学与阅读成绩；这些数值用于说明计算，并不证明统计假设成立。
>
> | Group · 组 | Three observations · 三个观测 | Mean · 均值 |
> |---|---|---|
> | A | $(59,70),(60,69),(61,71)$ | $(60,70)$ |
> | B | $(61,70),(62,69),(63,71)$ | $(62,70)$ |
> | C | $(59,72),(60,71),(61,73)$ | $(60,72)$ |
>
> <!-- bilingual-en:start -->
> Within every group the deviations are $(-1,0)^T,(0,-1)^T,(1,1)^T$. Thus each group contributes $\begin{bmatrix}2&1\\1&2\end{bmatrix}$ to $E$. The overall mean is $(182/3,212/3)^T$, giving
> <!-- bilingual-en:end -->
>
> 每组的组内偏离都是 $(-1,0)^T,(0,-1)^T,(1,1)^T$，因此每组为 $E$ 贡献 $\begin{bmatrix}2&1\\1&2\end{bmatrix}$。总均值为 $(182/3,212/3)^T$，从而
>
> $$E=\begin{bmatrix}6&3\\3&6\end{bmatrix},\qquad
> H=\begin{bmatrix}8&-4\\-4&8\end{bmatrix}.$$
>
> <!-- bilingual-en:start -->
> The off-diagonal entries matter: within-group deviations move together overall, whereas the group centres trade off mathematics and reading around the overall centre. Retaining only two separate sums of squares would discard this information. Whether the between-group pattern is unusually large relative to the within-group noise still requires a chosen statistic and its reference distribution.
> <!-- bilingual-en:end -->
>
> 非对角元素也有意义：组内偏离总体上同向变化，组中心则围绕总中心呈现数学与阅读一升一降的格局。只保留两个各自的平方和会丢掉这部分信息。组间格局相对于组内噪声是否异常大，还需要选定统计量及其参考分布才能判断。

<!-- bilingual-en:start -->
For one response, ordinary one-way ANOVA compares $H/(g-1)$ with $E/(N-g)$ through an $F$ ratio. For several responses these are matrices, so there is no single entrywise division that plays the same role. [[MANOVA四种统计量|The four classical MANOVA statistics]] summarise the comparison in different ways. The observed statistic alone is not a $p$-value; [[MANOVA检验校准|reference-distribution calibration]] also uses the dimensions, degrees of freedom, and model.
<!-- bilingual-en:end -->

只有一个响应时，普通单因素 ANOVA 用 $H/(g-1)$ 与 $E/(N-g)$ 的 $F$ 比率作比较。多个响应时，对象变成矩阵，不能靠逐元素相除得到同一种检验。[[MANOVA四种统计量]]用不同方式汇总比较；统计量本身还不是 $p$ 值，[[MANOVA检验校准|参考分布校准]]还要用维数、自由度和模型。

<!-- bilingual-en:start -->
## The shared-design model behind the group comparison
<!-- bilingual-en:end -->

*组间比较背后的共享设计模型*

<!-- bilingual-en:start -->
Stack the $N$ observed response vectors as rows of the [[多元数据矩阵|data matrix]] $Y$. A common design matrix $X$ describes group membership, factors, or covariates for those same units. In the [[多响应线性回归|multivariate linear model]],
<!-- bilingual-en:end -->

把 $N$ 个响应向量按行排列成[[多元数据矩阵]] $Y$，共同的设计矩阵 $X$ 描述这些单位的组别、因子或协变量。[[多响应线性回归|多响应线性模型]]写成

$$Y_{N\times p}=X_{N\times k}B_{k\times p}+U_{N\times p}.$$

<!-- bilingual-en:start -->
The $k$ columns of $X$ are design variables; $B$ contains their coefficients for each response; $U$ contains the errors. In the three-group example, three group-indicator columns suffice: an A student has design row $(1,0,0)$, a B student $(0,1,0)$, and a C student $(0,0,1)$. The rows of $B$ are then $\mu_1^T,\mu_2^T,\mu_3^T$.
<!-- bilingual-en:end -->

$X$ 的 $k$ 列是设计变量，$B$ 收集它们对各响应的系数，$U$ 收集误差。三组例子可以使用三列组别指示量：A 组学生的设计行为 $(1,0,0)$，B 组为 $(0,1,0)$，C 组为 $(0,0,1)$。此时 $B$ 的三行分别是 $\mu_1^T,\mu_2^T,\mu_3^T$。

<!-- bilingual-en:start -->
A prespecified [[多响应一般线性假设|general linear hypothesis]] has the form $H_0:CBA=D$. Here $C$ selects coefficient comparisons, $A$ selects response combinations, and $D$ gives their proposed values. If $C$ is $r\times k$ and $A$ is $p\times s$, then $D$ is $r\times s$. For equality of the three group means, take $C=\begin{bmatrix}1&0&-1\\0&1&-1\end{bmatrix}$, take $A=I_p$, the identity matrix with ones on the diagonal and zeros elsewhere, to keep all responses, and set $D$ to the zero matrix: this compares each of the first two groups with the third. This broader formulation explains why MANOVA is not restricted to categorical predictors.
<!-- bilingual-en:end -->

预先规定的[[多响应一般线性假设]]可写成 $H_0:CBA=D$。$C$ 选择系数对比，$A$ 选择响应组合，$D$ 指定它们的假定值。若 $C$ 为 $r\times k$、$A$ 为 $p\times s$，则 $D$ 为 $r\times s$。检验三组均值相等时，取 $C=\begin{bmatrix}1&0&-1\\0&1&-1\end{bmatrix}$，取 $A=I_p$（对角线为一、其余元素为零的单位矩阵）保留全部响应，并令 $D$ 为零矩阵，就是把前两组分别与第三组比较。这一更一般的形式也说明，MANOVA 并不限制预测变量只能是分类变量。

<!-- bilingual-en:start -->
Another one-way notation is $Y_{ij}=\mu+\tau_i+\varepsilon_{ij}$. Without an identifying constraint, increasing $\mu$ by a vector and subtracting it from every $\tau_i$ leaves all means unchanged. A constraint such as $\sum_i n_i\tau_i=0$ chooses a unique representation; under that constraint, equality of group means is equivalent to every $\tau_i=0$. The substantive target is the estimable group comparison, not arbitrary unconstrained coefficients; see [[MANOVA经典假设|the design and estimability conditions]].
<!-- bilingual-en:end -->

另一种单因素写法是 $Y_{ij}=\mu+\tau_i+\varepsilon_{ij}$。没有识别约束时，给 $\mu$ 加上一个向量，再从每个 $\tau_i$ 减掉它，所有组均值都不变。$\sum_i n_i\tau_i=0$ 这类约束选定唯一表示；在该约束下，组均值相等就等价于所有 $\tau_i=0$。真正要检验的是可估的组间比较，而非任意的无约束系数，见[[MANOVA经典假设|设计与可估性条件]]。

> [!important] Interpreting a joint test · 怎样解释联合检验
> <!-- bilingual-en:start -->
> In the one-way setting, $p=1$ reduces to ordinary ANOVA. With $g=2$ and the usual common-covariance model and invertibility conditions, the group-mean test is equivalent to [[pooled Hotelling T²|pooled two-sample Hotelling $T^2$]]; three groups are not a definition threshold. Rejection says that some tested mean comparison differs in some response direction. It does not show that every response differs, identify which groups differ, or establish a causal effect. Those follow-up questions require the appropriate [[多响应联合显著边界|coordinate or contrast inference]] and [[多响应因果边界|causal design]].
> <!-- bilingual-en:end -->
>
> 单因素情形中，$p=1$ 回到普通 ANOVA；当 $g=2$ 且满足通常的共同协方差模型与可逆性条件时，组均值检验等价于 [[pooled Hotelling T²|pooled 两样本 Hotelling $T^2$]]，三组不是定义门槛。拒绝原假设说明某个受检均值对比在某个响应方向上不同，不会证明每个响应都不同、自动指出哪几组不同，或建立因果效应。后续问题需要相应的[[多响应联合显著边界|坐标或对比推断]]以及[[多响应因果边界|因果设计]]。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.5. 多个总体均值向量比较：单因子 MANOVA|Local multivariate-statistics course, §1.5]] sets the one-way course scope and notation.
- [Penn State STAT 505, Lesson 8, §§8.2–8.3](https://online.stat.psu.edu/stat505/Lesson08) supports the group-mean hypothesis, within/between SSCP construction, and the route from a statistic to a test. The nine-observation example above is calculated directly from the displayed data.
- [SAS GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm) supports the shared design and simultaneous coefficient/response transformations. [statsmodels, multivariate-OLS source documentation, “Hypothesis test”](https://www.statsmodels.org/stable/_modules/statsmodels/multivariate/multivariate_ols.html) explicitly specifies the nonzero-target form $LBM=C$; this card renames those matrices $CBA=D$.
- [Stanley Sawyer, Washington University, *Multivariate Linear Models*, §§1, 3, 4, 6, pp. 1, 3–5, 7–8](https://www.math.wustl.edu/~sawyer/handouts/manovas.pdf) supports the shared-response model, matrix comparison, and two-group Hotelling equivalence; matrix products are checked against their dimensions.
<!-- bilingual-en:end -->

- 本地课程 §1.5 确定单因素课程范围与记号；Penn State §§8.2–8.3 支持均值向量假设、组内/组间 SSCP 及检验思路。九个观测的算例由表中数据直接计算。
- SAS 支持共享设计及系数、响应两个方向的变换；statsmodels 原始文档明确给出非零右端 $LBM=C$，本卡统一改记为 $CBA=D$。
- Washington University 讲义的上述章节支持多响应模型、矩阵比较与两组 Hotelling 等价性；矩阵乘法另按尺寸核对。
