---
aliases:
  - VIF threshold boundary
  - Interpreting VIF thresholds
  - VIF 截断值的使用边界
student_os: knowledge-atom
atom_id: ECON-COL-005
atom_set: multicollinearity-design-diagnostics
atom_type: diagnostic-boundary
status: source-checked
requires:
  - "[[方差膨胀因子]]"
  - "[[模型选择目标]]"
related:
  - "[[显著性解释边界]]"
  - "[[共线性补救原则]]"
part_of:
  - "[[多重共线性.canvas|多重共线性与设计矩阵诊断]]"
---

<!-- bilingual-en:start -->
# Interpreting VIF thresholds
<!-- bilingual-en:end -->

*VIF 阈值边界*

<!-- bilingual-en:start -->
A regression reports VIF $12$ for an important control variable. Should you remove it because a guide says “keep VIF below $10$”? The number tells you how much that column overlaps with the other predictors. It does not tell you whether removing the variable preserves your research question, or whether the coefficient is already estimated precisely enough. A **VIF threshold is a screening convention**: it directs attention to a possible difficulty, which must then be evaluated for the intended comparison.
<!-- bilingual-en:end -->

回归结果中，一个重要控制变量的 VIF 为 $12$。看到指南说“VIF 应低于 $10$”，是否就应删除它？这个数告诉你该列与其他预测列有多少重叠，却没有告诉你删项是否保留原研究问题，也没有告诉你系数是否已经估计得足够精确。**VIF 阈值是一种筛查约定**：它提醒你调查某种困难，随后还要根据预定比较判断其影响。

<!-- bilingual-en:start -->
## What the cutoff leaves out
<!-- bilingual-en:end -->

*阈值遗漏了哪些信息？*

<!-- bilingual-en:start -->
For an unweighted [[普通最小二乘|OLS regression]] with an intercept and [[线性无关|linearly independent design columns]], choose a nonconstant predictor $x_j$. Its [[方差膨胀因子|variance inflation factor]] is $1/(1-R_j^2)$, where $R_j^2$ is the fraction of its centered variation explained by regressing it on the other columns, including the intercept. This auxiliary regression uses the same observations as the main regression; its response is $x_j$, not the outcome $y$.
<!-- bilingual-en:end -->

对含截距、设计列[[线性无关]]的不加权[[普通最小二乘|OLS 回归]]，选一个非常数预测变量 $x_j$。它的[[方差膨胀因子]]是 $1/(1-R_j^2)$；$R_j^2$ 表示把该列对其他列（包括截距）回归时，解释了多少中心化变动。辅助回归与主回归使用相同观测，其响应是 $x_j$，不是结果 $y$。

<!-- bilingual-en:start -->
To translate this design ratio into a sampling variance, specify $y=X\beta+u$: $X$ is the fixed [[设计矩阵|design matrix]] with $n$ rows, $\beta$ the coefficient vector and $u$ the error vector. Assume $E[u\mid X]=0$ and $\operatorname{Cov}(u\mid X)=\sigma^2I_n$, with $0<\sigma^2<\infty$. Here $I_n$ is the identity matrix, with ones on its diagonal and zeros elsewhere. Thus errors have zero [[条件期望|conditional mean]], a common finite variance and zero cross-covariances in their [[协方差矩阵|covariance matrix]]. If $T_j=\sum_i(x_{ij}-\bar x_j)^2$ is the predictor's total centered squared variation, then the [[FWL残差化定理|FWL calculation]] gives
<!-- bilingual-en:end -->

要把设计比值解释为抽样方差，还需指定 $y=X\beta+u$：$X$ 是有 $n$ 行的固定[[设计矩阵]]，$\beta$ 是系数向量，$u$ 是误差向量。假定 $E[u\mid X]=0$、$\operatorname{Cov}(u\mid X)=\sigma^2I_n$，且 $0<\sigma^2<\infty$，其中 $I_n$ 为对角元是一、其余为零的单位矩阵；即误差的[[条件期望|条件均值]]为零，[[协方差矩阵]]中的有限方差相同、交叉协方差为零。若 $T_j=\sum_i(x_{ij}-\bar x_j)^2$ 是预测变量的总中心化变动平方和，[[FWL残差化定理|FWL 计算]]给出

$$\operatorname{Var}(\hat\beta_j\mid X)=\frac{\sigma^2}{T_j}\,\mathrm{VIF}_j.$$

<!-- bilingual-en:start -->
Here $\hat\beta_j$ is the OLS estimate and $\bar x_j$ the sample mean. VIF supplies only one factor. Noise variance and total predictor spread also matter; sample size matters through the design rather than through a universal cutoff. The square root of this variance is the sampling standard deviation, which a reported [[标准误含义|standard error]] estimates. Under heteroskedastic or correlated errors, use the appropriate [[协方差失效下的OLS|covariance calculation]]; the same VIF is generally not the actual variance multiplier.
<!-- bilingual-en:end -->

这里 $\hat\beta_j$ 是 OLS 估计，$\bar x_j$ 是样本均值。VIF 只提供其中一个因子；噪声方差和预测变量总变动同样重要，样本量通过设计影响精度，而不是决定一个万能阈值。上述方差开平方是抽样标准差，报告的[[标准误含义|标准误]]估计的正是它。误差存在异方差或相关时，应使用适当的[[协方差失效下的OLS|协方差计算]]；同一个 VIF 一般不再是实际方差的放大倍数。

> [!example] Same VIF, different precision · 相同 VIF，不同精度
> <!-- bilingual-en:start -->
> Compare two designs for the same slope in the **same units**, both with $\mathrm{VIF}_j=10$ and $\sigma^2=4$. If $T_j=40$, the coefficient variance is $1$ and its sampling standard deviation is $1$. If $T_j=4000$, they are $0.01$ and $0.1$. The latter design could arise from repeating the predictor patterns $100$ times while collecting new observations whose errors still satisfy the stated covariance model. The VIF stays the same while the information grows. Merely copying the old responses would not satisfy that new-observation argument.
> <!-- bilingual-en:end -->
>
> 比较**相同单位**下同一斜率的两个设计，两者都满足 $\mathrm{VIF}_j=10$、$\sigma^2=4$。若 $T_j=40$，系数方差为 $1$，抽样标准差为 $1$；若 $T_j=4000$，则分别为 $0.01$ 和 $0.1$。后一设计可以来自重复预测变量组合 $100$ 次，同时获取误差仍满足上述协方差模型的新观测。VIF 不变，信息却增加了。仅复制旧响应，并不能使用这个新观测论证。

<!-- bilingual-en:start -->
Penn State's course uses $4$ and $10$ as rules of thumb for investigation and concern. These values are not breakpoints in the variance formula: moving from $9.9$ to $10.1$ changes variance continuously. A VIF of $3$ can still accompany unacceptable imprecision; a VIF of $20$ need not prevent useful prediction. Neither number establishes a causal interpretation or determines its validity. That needs a separate model and identification argument.
<!-- bilingual-en:end -->

Penn State 课程把 $4$ 和 $10$ 用作值得调查、需要关注的经验界限。这些数不是方差公式中的断点：从 $9.9$ 变到 $10.1$，方差连续变化。VIF 为 $3$ 时仍可能不够精确，VIF 为 $20$ 时也未必不能有效预测。两个数都不能建立或否定因果解释；因果解释需要另行论证模型与识别。

<!-- bilingual-en:start -->
## What to investigate after a high VIF
<!-- bilingual-en:end -->

## 高 VIF 之后应问什么

<!-- bilingual-en:start -->
Begin with the [[模型选择目标|target of the model]]. If it is one slope, examine that slope's uncertainty in meaningful units. If it is a joint contrast or prediction, evaluate that quantity directly: uncertainty in separate coefficients can partly cancel in a combination. Check prediction on relevant new observations, especially whether their predictor combinations resemble the sample. Good prediction in the observed region does not license extrapolation or causal claims about individual coefficients.
<!-- bilingual-en:end -->

先明确[[模型选择目标]]。若关注一个斜率，就用有实际意义的单位检查其不确定性；若关注联合对比或预测，就直接评价该量，因为单独系数的不确定性可能在某种组合中部分抵消。应在相关的新观测上检查预测，尤其关注预测变量组合是否接近样本。观测区域内预测良好，不代表可以任意外推或对单独系数作因果解释。

<!-- bilingual-en:start -->
Next inspect the actual design: confirm the common sample, intercept and coding, and distinguish an exact coding identity from close overlap or deliberately constructed powers and interactions. An exact identity makes a separate coefficient unidentified; a merely high finite VIF raises a precision question. A high value for a scientifically necessary control is not permission to delete it. Removal can change the coefficient's meaning or introduce [[遗漏变量偏差|omitted-variable bias]] relative to the intended target. Retaining the variable and honestly reporting a wide interval can be the appropriate outcome.
<!-- bilingual-en:end -->

然后检查实际设计：核实共同样本、截距和编码，区分精确编码恒等式、接近重叠，以及有意构造的幂次项和交互项。精确恒等式使单独系数无法识别；仅仅较大的有限 VIF 则提出精度问题。科学上必要的控制项 VIF 较高，不构成删除它的理由；删项可能改变系数含义，或相对预定目标引入[[遗漏变量偏差]]。保留变量并如实报告较宽区间，可以是适当结果。

<!-- bilingual-en:start -->
A repair should therefore have a stated purpose; see [[共线性补救原则|principles for addressing collinearity]]. If repeated model selection also uses the observed outcome, significance or fit, ordinary final-model p-values and intervals do not automatically retain their prespecified guarantees; this is a [[选择后推断|post-selection inference]] issue. A rule using only fixed $X$ is not automatically outcome-driven selection, but its resulting model still needs a justified target and error assumptions.
<!-- bilingual-en:end -->

因此，修复应有明确目的，见[[共线性补救原则]]。若反复选模型还使用了观察到的结果、显著性或拟合效果，最终模型的普通 p 值和区间并不自动保留预先指定分析的保证，这是[[选择后推断]]问题。只使用固定 $X$ 的规则并不自动属于结果驱动选择，但所得模型仍需论证其目标和误差假设。

## 来源与核验

<!-- bilingual-en:start -->
- [Penn State STAT 501, Lesson 12.4](https://online.stat.psu.edu/stat501/Lesson12): the centered VIF formula, the matched variance baseline and the explicitly heuristic $4/10$ cutoffs. The two precision examples are direct calculations under the stated covariance model, not empirical cutoff recommendations.
- [SAS/STAT, REG: Collinearity Diagnostics](https://support.sas.com/documentation/cdl/en/statug/67523/HTML/default/statug_reg_details24.htm): VIF/tolerance and the lack of a formal VIF cutoff for effects on predictions.
- [Berk et al. (2013), Valid post-selection inference, §1, PDF pp. 1–3](https://arxiv.org/pdf/1306.1059#page=1): outcome-dependent model selection, changing submodel targets and the loss of automatic classical inference guarantees; p. 2 explicitly distinguishes selection involving the stochastic response when predictors are fixed.
- [[02_Economy/01_Econometrics/06_多重共线性.md#4.4. 方差膨胀因子检验|Course §4.4]] and its subsequent remedies discussion: local scope, the variance/standard-deviation distinction and preserving necessary variables. [[模型选择目标|Model-selection targets]], [[遗漏变量偏差|omitted-variable bias]] and [[选择后推断|post-selection inference]] provide the separate downstream arguments.
<!-- bilingual-en:end -->

- Penn State 第 12.4 节核对中心化 VIF 公式、匹配的方差基准及明确称为经验法则的 $4/10$ 界限；两个精度例子是所述协方差模型下的直接计算，不是经验阈值建议。
- SAS/STAT 的共线性诊断说明 VIF、容忍度及预测影响不存在正式的通用 VIF 阈值。
- Berk 等第 1 节、PDF 第 1–3 页核对结果驱动选择、子模型目标变化和经典推断保证的边界；第 2 页明确区分固定预测变量时涉及随机响应的选择。
- 本地课程第 4.4 节及其后的补救部分核对课程范围、方差与标准差倍数区别，以及保留必要变量的选择；相关原子分别解释模型目标、遗漏变量偏差与选择后推断。
