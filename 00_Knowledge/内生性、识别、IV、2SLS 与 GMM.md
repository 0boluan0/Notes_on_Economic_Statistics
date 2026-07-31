---
aliases:
  - "Endogeneity and Identification"
  - "Instrumental Variables"
  - "IV"
  - "2SLS"
  - "GMM"
  - "内生性与识别"
status: source-checked
---

# 内生性、识别、IV、2SLS 与 GMM
<!-- bilingual-en:start -->
*Endogeneity, identification, IV, 2SLS, and GMM*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 当解释变量和回归中未观察到的影响一起变化时，OLS 会把两者作用混在一起；本主题寻找能隔离外生变化的设计。
> **具体锚点：** 研究教育对工资时，能力影响教育也影响工资。OLS 可能把能力差异误算成教育回报；增加样本或换稳健标准误都无济于事。
> **核心难点：** 工具变量必须既推动处理（relevance），又只能通过处理影响结果且与结构误差无关（exogeneity/exclusion）；前者可部分检验，后者主要靠设计论证。
> **为什么重要：** 它直接决定一个回归系数能否解释为因果效应，而不是只影响精度。
> **继续：** 先弄清内生性来源和目标参数，再学 IV/2SLS；有多个矩条件时进入 GMM。
> <!-- bilingual-en:start -->
> **What it solves:** When an explanatory variable moves together with unobserved determinants of the outcome, OLS mixes their effects. This topic seeks designs that isolate exogenous variation.
> **Concrete anchor:** Ability affects both schooling and wages. OLS may therefore mistake wage differences caused by ability for returns to education; neither a larger sample nor robust standard errors resolves this problem.
> **Central difficulty:** An instrument must change the treatment (relevance) and affect the outcome only through that treatment while remaining unrelated to the structural error (exogeneity and exclusion). Relevance is partly testable; exogeneity and exclusion rely mainly on the research design.
> **Why it matters:** Endogeneity determines whether a regression coefficient can be interpreted causally, not merely how precisely it is estimated.
> **Continue with:** First identify the source of endogeneity and the target parameter, then study IV and 2SLS. Multiple moment conditions lead to GMM.
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - [[02_Economy/01_Econometrics/太白金星v计量.pdf]]：支持课程范围、初级计量顺序与示例；核心结论另与权威教材或原始论文交叉核验。
> - Wooldridge, *Introductory Econometrics: A Modern Approach*：核验线性模型、推断、内生性、面板与时序计量。
> <!-- bilingual-en:start -->
> - [[02_Economy/01_Econometrics/太白金星v计量.pdf|Local econometrics course notes]] support the course scope, introductory sequence, and examples; the core results were also cross-checked against authoritative textbooks or original research.
> - Wooldridge, *Introductory Econometrics: A Modern Approach*, supports the treatment of linear models, inference, endogeneity, panel data, and time-series econometrics.
> <!-- bilingual-en:end -->

## 内生性是什么
<!-- bilingual-en:start -->
*What endogeneity means*
<!-- bilingual-en:end -->

在 $Y=\beta X+u$ 中，若 $E[u\mid X]\ne0$ 或至少 $E[Xu]\ne0$，X 内生。u 汇总未进入模型、却影响 Y 的因素；X 与它共同变化时，OLS 无法把 X 的变化单独解释。内生性是相对某个结构方程和目标参数而言，不是变量天生属性。
<!-- bilingual-en:start -->
In $Y=\beta X+u$, $X$ is endogenous if $E[u\mid X]\ne0$, or at least if $E[Xu]\ne0$. The error $u$ collects factors that affect $Y$ but are omitted from the model. When $X$ moves with those factors, OLS cannot isolate variation in $X$. Endogeneity is defined relative to a particular structural equation and target parameter; it is not an intrinsic property of a variable.
<!-- bilingual-en:end -->

## 三类常见来源
<!-- bilingual-en:start -->
*Three common sources*
<!-- bilingual-en:end -->

遗漏变量：未观测因素同时影响 X 和 Y；同时性：X 与 Y 在同一系统共同决定；测量误差：观测 X 含噪声并进入复合误差。样本选择和动态反馈也可产生相关。不同来源对应不同设计，不应统称后随便“加控制”。
<!-- bilingual-en:start -->
Omitted variables arise when an unobserved factor affects both $X$ and $Y$. Simultaneity arises when $X$ and $Y$ are jointly determined within one system. Measurement error arises when observed $X$ contains noise that becomes part of the composite error. Sample selection and dynamic feedback can also create correlation. Different sources require different designs; they should not be lumped together and addressed by indiscriminately “adding controls.”
<!-- bilingual-en:end -->

## 识别与工具变量条件
<!-- bilingual-en:start -->
*Identification and the conditions for an instrument*
<!-- bilingual-en:end -->

工具 Z 需满足相关性 $Cov(Z,X)\ne0$，以及排除/外生性使 $E[Zu]=0$。排除限制意味着 Z 不通过 X 之外路径影响 Y；这常需要制度、时序和机制证据。若效应异质，标准 IV 可能识别 compliers 的 LATE，而非总体平均效应。
<!-- bilingual-en:start -->
An instrument $Z$ must be relevant, $Cov(Z,X)\ne0$, and satisfy exclusion and exogeneity so that $E[Zu]=0$. The exclusion restriction means that $Z$ does not affect $Y$ through any path other than $X$; establishing it usually requires institutional, temporal, and mechanistic evidence. With heterogeneous effects, standard IV may identify the LATE for compliers rather than the population average effect.
<!-- bilingual-en:end -->

## Wald、IV 与 2SLS
<!-- bilingual-en:start -->
*Wald, IV, and 2SLS*
<!-- bilingual-en:end -->

单工具单内生变量时，Wald/IV 比率为 reduced form（Z 对 Y）除以 first stage（Z 对 X）。2SLS 第一步用全部外生变量和工具预测内生 X，第二步用其被工具解释的部分估计结果。标准误必须用联合 2SLS 公式，不能把第二步当普通 OLS。
<!-- bilingual-en:start -->
With one instrument and one endogenous variable, the Wald or IV ratio is the reduced-form effect of $Z$ on $Y$ divided by the first-stage effect of $Z$ on $X$. In 2SLS, the first stage predicts endogenous $X$ from all exogenous variables and instruments; the second stage estimates the outcome relationship using the part of $X$ explained by the instruments. Standard errors must come from the joint 2SLS calculation, not from treating the second stage as an ordinary OLS regression.
<!-- bilingual-en:end -->

## 弱工具与有限样本
<!-- bilingual-en:start -->
*Weak instruments and finite samples*
<!-- bilingual-en:end -->

第一阶段弱时 2SLS 分布偏斜、常规 t 检验失真，估计可向 OLS 偏。first-stage F 只是诊断之一，多个内生变量和异方差需用合适的弱识别统计与 Anderson–Rubin 等稳健推断。
<!-- bilingual-en:start -->
When the first stage is weak, the finite-sample distribution of 2SLS is skewed, conventional $t$ tests are distorted, and estimates may be biased toward OLS. The first-stage $F$ statistic is only one diagnostic. With multiple endogenous variables or heteroskedasticity, use appropriate weak-identification statistics and robust procedures such as Anderson–Rubin inference.
<!-- bilingual-en:end -->

## 过度识别与工具有效性
<!-- bilingual-en:start -->
*Overidentification and instrument validity*
<!-- bilingual-en:end -->

工具数超过内生变量数时可做 Hansen J/Sargan 检验矩条件整体相容，但未拒绝不能证明所有工具有效；若所有工具以相同方式无效，检验也可能无力。过多工具会过拟合第一阶段并损害推断。
<!-- bilingual-en:start -->
When there are more instruments than endogenous variables, Hansen's $J$ or Sargan's test can assess whether the moment conditions are jointly compatible with the data. Failure to reject does not prove that every instrument is valid, and the test may have little power when all instruments fail in the same way. Too many instruments can overfit the first stage and damage inference.
<!-- bilingual-en:end -->

## GMM
<!-- bilingual-en:start -->
*Generalized method of moments*
<!-- bilingual-en:end -->

GMM 选择参数使样本矩 $g_n(\theta)$ 接近零，并用权重矩阵衡量各矩条件。恰好识别时权重不影响点估计；过度识别时有效权重与矩条件协方差逆相关。两步 GMM 标准误和有限样本修正必须匹配估计。
<!-- bilingual-en:start -->
GMM chooses parameters that make the sample moments $g_n(\theta)$ close to zero, using a weighting matrix to combine the moment conditions. Under exact identification, the weight does not affect the point estimate. Under overidentification, the efficient weight is related to the inverse covariance matrix of the moments. Standard errors and finite-sample corrections for two-step GMM must match the estimator actually used.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 用教育—工资例子解释为什么 X 与误差相关会污染 OLS。
<!-- bilingual-en:start -->
*Use the education–wage example to explain why correlation between $X$ and the error contaminates OLS.*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 能力未观测而进入误差，同时提高教育和工资；教育高的人工资高的一部分来自能力，OLS 会把它混入教育系数。
> <!-- bilingual-en:start -->
> Ability is unobserved and therefore enters the error, while also increasing both education and wages. Part of the wage advantage among more educated people is due to ability, and OLS mixes that contribution into the education coefficient.
> <!-- bilingual-en:end -->
### 工具变量的相关性显著，是否足以证明工具有效？
<!-- bilingual-en:start -->
*Is a statistically significant relevance relationship enough to establish that an instrument is valid?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不足。还要论证工具与结构误差独立且没有绕过处理直接影响结果；这通常不是单靠数据检验能证明。
> <!-- bilingual-en:start -->
> No. One must also argue that the instrument is independent of the structural error and has no direct path to the outcome that bypasses the treatment. Data tests alone generally cannot establish those claims.
> <!-- bilingual-en:end -->
### 弱工具为什么不仅是‘标准误大’？
<!-- bilingual-en:start -->
*Why are weak instruments more than a problem of “large standard errors”?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 它会使 2SLS 的有限样本分布非正态、偏向 OLS，并让常规 t/F 推断严重失真。
> <!-- bilingual-en:start -->
> Weak instruments make the finite-sample distribution of 2SLS nonnormal, can bias it toward OLS, and severely distort conventional $t$ and $F$ inference.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[02_Economy/01_Econometrics/太白金星v计量.pdf]]：支持课程范围、初级计量顺序与示例；核心结论另与权威教材或原始论文交叉核验。
- Wooldridge, *Introductory Econometrics: A Modern Approach*：核验线性模型、推断、内生性、面板与时序计量。
<!-- bilingual-en:start -->
- [[02_Economy/01_Econometrics/太白金星v计量.pdf|Local econometrics course notes]] support the course scope, introductory sequence, and examples; the core results were also cross-checked against authoritative textbooks or original research.
- Wooldridge, *Introductory Econometrics: A Modern Approach*, was used to verify the linear model, inference, endogeneity, panel data, and time-series econometrics.
<!-- bilingual-en:end -->
