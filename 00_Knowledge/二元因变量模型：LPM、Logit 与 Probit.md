---
aliases:
  - "Binary Outcome Models"
  - "Binary Response Models"
  - "线性概率模型"
  - "LPM"
  - "Logit"
  - "Probit"
status: source-checked
---

# 二元因变量模型：LPM、Logit 与 Probit
<!-- bilingual-en:start -->
*Binary-outcome models: LPM, logit, and probit*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 当结果只有 0/1——是否就业、是否违约、是否采用某项技术——回归要解释的是事件发生概率 $P(Y=1\mid X)$，而不是把 0 和 1 当作普通连续数值。
> **具体锚点：** 若研究培训是否提高就业概率，LPM 的培训系数可直接读成概率百分点；logit/probit 则先估计非线性概率曲线，再比较“培训”和“未培训”的预测概率。
> **核心难点：** LPM 简单但可能预测越界且必然异方差；logit/probit 保证概率在 $[0,1]$，但原始系数不是概率变化，必须转成预测概率、离散变化或边际效应。
> **为什么重要：** “系数显著”并不说明实际概率变化多大；选择模型、解释尺度、标准误结构和识别条件必须分开处理。
> **继续：** 先看 [[#共同目标：条件概率]]，再比较 [[#LPM：直接在线性概率尺度上估计]] 与 [[#Logit 与 Probit：把线性指数映射为概率]]；重复观察的二元结果另见 [[面板数据：Pooled OLS、固定效应与随机效应#二元结果与动态面板的边界]]。
> <!-- bilingual-en:start -->
> **What it solves:** When the outcome is 0/1—employed or not, default or not, adopted a technology or not—the regression target is the event probability $P(Y=1\mid X)$, not an ordinary continuous measurement taking the values zero and one.
> **Concrete anchor:** To study whether training raises employment, the training coefficient in an LPM is directly a percentage-point change. Logit and probit instead estimate a nonlinear probability curve and compare predicted probabilities with and without training.
> **Central difficulty:** The LPM is simple but can predict outside the unit interval and is inherently heteroskedastic. Logit and probit keep probabilities in $[0,1]$, but their raw coefficients are not probability changes and must be translated into predictions, discrete changes, or marginal effects.
> **Why it matters:** A statistically significant coefficient does not reveal the practical change in probability. Model choice, interpretation scale, standard-error structure, and identification conditions must be handled separately.
> **Continue with:** Begin with the [[#共同目标：条件概率|common probability target]], then compare the [[#LPM：直接在线性概率尺度上估计|LPM]] with [[#Logit 与 Probit：把线性指数映射为概率|logit and probit]]. For repeated binary outcomes, see the [[面板数据：Pooled OLS、固定效应与随机效应#二元结果与动态面板的边界|boundary between binary outcomes and panel methods]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., Sections 7.5 and 17.1：核验 LPM、二元响应的 logit/probit、极大似然与边际效应。
> - Stata 官方 [logit](https://www.stata.com/manuals/rlogit.pdf)、[probit](https://www.stata.com/manuals/rprobit.pdf) 与 [margins](https://www.stata.com/manuals/rmargins.pdf) 手册：核验概率函数、odds ratio、连续/离散边际效应和平均边际效应的计算尺度。
> - Stata 官方 [`xtlogit`](https://www.stata.com/manuals/xtxtlogit.pdf) 手册：核验条件固定效应 logit 与结果不发生组内变化时整组观测被排除的边界。
> - [[02_Economy/01_Econometrics/10_虚拟变量.md]] 与 [[02_Economy/01_Econometrics/太白金星v计量.pdf]] 第 13 章：核对本课程的 LPM—潜变量—probit/logit 学习顺序。
> <!-- bilingual-en:start -->
> - Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., Sections 7.5 and 17.1, supports the LPM, logit and probit binary-response models, maximum likelihood, and marginal effects.
> - The official Stata manuals for [logit](https://www.stata.com/manuals/rlogit.pdf), [probit](https://www.stata.com/manuals/rprobit.pdf), and [margins](https://www.stata.com/manuals/rmargins.pdf) support the probability functions, odds ratios, continuous and discrete marginal effects, and average marginal effects.
> - Stata's official [`xtlogit` manual](https://www.stata.com/manuals/xtxtlogit.pdf) supports the conditional fixed-effects logit boundary and the omission of groups with no within-group outcome variation.
> - [[02_Economy/01_Econometrics/10_虚拟变量.md|Course note on indicator variables]] and Chapter 13 of the [[02_Economy/01_Econometrics/太白金星v计量.pdf|local econometrics course text]] establish the course's LPM–latent-variable–probit/logit sequence.
> <!-- bilingual-en:end -->

## 共同目标：条件概率
<!-- bilingual-en:start -->
*The common target: a conditional probability*
<!-- bilingual-en:end -->

当 $Y\in\{0,1\}$ 时，条件期望与成功概率相同：
<!-- bilingual-en:start -->
When $Y\in\{0,1\}$, its conditional expectation equals its success probability:
<!-- bilingual-en:end -->

$$
E(Y\mid X)=1\cdot P(Y=1\mid X)+0\cdot P(Y=0\mid X)=P(Y=1\mid X).
$$

因此 LPM、logit 和 probit 都在描述同一个对象 $p(X)=P(Y=1\mid X)$，区别是给 $p(X)$ 施加什么函数形式、如何估计，以及如何把参数转回概率尺度。二元因变量不要与二元解释变量混淆：`employed` 可以是结果，`trained` 可以是解释变量，两者虽然都编码为 0/1，却扮演不同角色。
<!-- bilingual-en:start -->
Thus, the LPM, logit, and probit all describe the same object, $p(X)=P(Y=1\mid X)$. They differ in the functional form imposed on $p(X)$, their estimation method, and how parameters are translated back to the probability scale. Do not confuse a binary outcome with a binary regressor: `employed` may be the outcome and `trained` a regressor. Both are coded 0/1 but play different roles.
<!-- bilingual-en:end -->

这些模型本身只描述条件概率。把某个 $X$ 的变化解释成因果效应，仍需说明无混杂、随机分配、工具变量或其他识别设计；把 OLS 换成 logit 并不会修复内生性。
<!-- bilingual-en:start -->
These models by themselves describe a conditional probability. Interpreting a change in $X$ causally still requires an argument based on no confounding, random assignment, instruments, or another identification design. Replacing OLS with logit does not repair endogeneity.
<!-- bilingual-en:end -->

## LPM：直接在线性概率尺度上估计
<!-- bilingual-en:start -->
*The LPM: estimation directly on the probability scale*
<!-- bilingual-en:end -->

线性概率模型设定
<!-- bilingual-en:start -->
The linear probability model specifies
<!-- bilingual-en:end -->

$$
P(Y=1\mid X)=E(Y\mid X)=X'\beta.
$$

用 OLS 估计后，连续变量 $X_j$ 的系数 $\beta_j$ 表示 $X_j$ 增加 1 单位时概率改变 $\beta_j$；若 $\beta_j=0.08$，就是增加 0.08，即 8 个百分点，而不是增加 8%。二元解释变量应比较从 0 到 1 的离散变化；在线性且无交互的 LPM 中，这个变化恰好也是其系数。
<!-- bilingual-en:start -->
After OLS estimation, the coefficient $\beta_j$ on a continuous regressor is the change in probability associated with a one-unit increase in $X_j$. If $\beta_j=0.08$, the change is 0.08, or eight percentage points—not an 8 percent increase. A binary regressor should be interpreted through its discrete change from zero to one; in a linear LPM without interactions, that change is exactly its coefficient.
<!-- bilingual-en:end -->

LPM 的优点是目标透明、交互项和固定效应容易解释、估计快速，而且平均部分效应常可直接读出。它也有三个结构性限制：
<!-- bilingual-en:start -->
The LPM has a transparent target, makes interactions and fixed effects easy to interpret, is computationally simple, and often reports average partial effects directly. It also has three structural limitations:
<!-- bilingual-en:end -->

1. $X'\hat\beta$ 可能小于 0 或大于 1；这些数仍是线性投影的拟合值，却不能作为合法概率。
2. 由于 $Var(Y\mid X)=p(X)[1-p(X)]$，误差方差随 $X$ 改变，经典同方差标准误不成立。
3. 每个连续变量的边际效应被强制为常数，难以表达接近 0 或 1 时趋于平缓的概率曲线。
<!-- bilingual-en:start -->

&nbsp;
**1.** $X'\hat\beta$ can be below zero or above one. Such values remain fitted values from a linear projection but are not valid probabilities.<br>
**2.** Because $Var(Y\mid X)=p(X)[1-p(X)]$, the error variance changes with $X$, invalidating homoskedastic standard errors.<br>
**3.** The marginal effect of each continuous regressor is forced to be constant, making it difficult to represent a probability curve that flattens near zero or one.<br>
<!-- bilingual-en:end -->

异方差稳健标准误是 LPM 推断的最低要求；有重复单位、学校或地区层级时还要按相关层级聚类。但稳健标准误只修复标准误估计，不会把越界预测拉回 $[0,1]$，也不会修复错误的条件均值形状。
<!-- bilingual-en:start -->
Heteroskedasticity-robust standard errors are the minimum for LPM inference; repeated units or school- and region-level dependence additionally call for appropriate clustering. Robust standard errors only repair variance estimation. They neither bring out-of-range fitted values back into $[0,1]$ nor correct a misspecified conditional-mean shape.
<!-- bilingual-en:end -->

## Logit 与 Probit：把线性指数映射为概率
<!-- bilingual-en:start -->
*Logit and probit: mapping a linear index into a probability*
<!-- bilingual-en:end -->

两者都写成
<!-- bilingual-en:start -->
Both models take the form
<!-- bilingual-en:end -->

$$
P(Y=1\mid X)=G(X'\beta),
$$

其中 logit 使用 logistic CDF，probit 使用标准正态 CDF：
<!-- bilingual-en:start -->
where logit uses the logistic CDF and probit uses the standard normal CDF:
<!-- bilingual-en:end -->

$$
G_{logit}(z)=\Lambda(z)=\frac{e^z}{1+e^z},
\qquad
G_{probit}(z)=\Phi(z).
$$

两个函数都严格递增并把任意线性指数映射到 $(0,1)$。因此 $\beta_j$ 的符号给出 $X_j$ 对成功概率影响的方向，但其大小通常不是概率变化。logit 和 probit 通常用极大似然估计；有效推断要求观测独立结构或聚类结构被正确处理，并且均值模型、链接函数与抽样过程足以支持所用似然或稳健推断。
<!-- bilingual-en:start -->
Both functions are strictly increasing and map any linear index into $(0,1)$. The sign of $\beta_j$ therefore gives the direction of $X_j$'s effect on the success probability, but its magnitude is generally not a probability change. Logit and probit are usually estimated by maximum likelihood. Valid inference requires the observation-level independence or clustering structure to be handled correctly and the mean model, link, and sampling process to justify the likelihood or robust inference being used.
<!-- bilingual-en:end -->

潜变量表达 $Y^*=X'\beta+\varepsilon$、$Y=1[Y^*>0]$ 可以解释两种模型从何而来：若 $\varepsilon$ 是 logistic 分布得到 logit，若是标准正态得到 probit。但 $Y^*$ 的尺度不可观察，所以不同模型或样本中的原始系数不能机械比较；实质解释应回到预测概率和边际效应。
<!-- bilingual-en:start -->
The latent-variable representation $Y^*=X'\beta+\varepsilon$, $Y=1[Y^*>0]$, explains where the two models come from: a logistic error yields logit, while a standard normal error yields probit. Because the scale of $Y^*$ is unobserved, raw coefficients should not be compared mechanically across models or samples. Substantive interpretation should return to predicted probabilities and marginal effects.
<!-- bilingual-en:end -->

logit 还允许 odds 表达：$odds=p/(1-p)$，$e^{\beta_j}$ 是其他变量不变时 $X_j$ 增加 1 单位对应的条件 odds ratio。它不是风险比，更不是概率增加 $e^{\beta_j}-1$；当基准概率不同，相同 odds ratio 对应的概率变化也不同。
<!-- bilingual-en:start -->
Logit also permits an odds interpretation. With $odds=p/(1-p)$, $e^{\beta_j}$ is the conditional odds ratio associated with a one-unit increase in $X_j$, holding other variables fixed. It is neither a risk ratio nor a probability increase of $e^{\beta_j}-1$; the same odds ratio corresponds to different probability changes at different baseline probabilities.
<!-- bilingual-en:end -->

## 怎样解释 Logit 与 Probit
<!-- bilingual-en:start -->
*How to interpret logit and probit*
<!-- bilingual-en:end -->

对近似连续的 $X_j$，个体 $i$ 处的边际效应是
<!-- bilingual-en:start -->
For an approximately continuous regressor $X_j$, the marginal effect for observation $i$ is
<!-- bilingual-en:end -->

$$
\frac{\partial P(Y_i=1\mid X_i)}{\partial X_{ij}}
=g(X_i'\beta)\beta_j,
$$

其中 $g$ 是相应 CDF 的密度。它随 $X_i$ 改变；同一个 $\beta_j$ 在概率曲线中部和尾部对应不同的概率变化。
<!-- bilingual-en:start -->
where $g$ is the density associated with the CDF. This effect varies with $X_i$; the same $\beta_j$ corresponds to different probability changes in the middle and tails of the probability curve.
<!-- bilingual-en:end -->

对二元解释变量 $D$，不要用导数代替从 0 到 1 的真实变化，而应计算
<!-- bilingual-en:start -->
For a binary regressor $D$, do not substitute a derivative for its actual zero-to-one change. Compute
<!-- bilingual-en:end -->

$$
G(X_{-D}'\beta+\beta_D)-G(X_{-D}'\beta).
$$

常用汇总有三种：在特定案例或政策相关协变量处报告预测概率；把每个样本点的边际效应求平均得到 average marginal effect（AME）；或在平均协变量处计算 marginal effect at the mean（MEM）。AME 回答“样本中平均改变多少”，通常比构造一个未必真实存在的“平均人”更容易解释。
<!-- bilingual-en:start -->
Three summaries are common: report predicted probabilities for a particular case or policy-relevant covariate profile; average observation-specific marginal effects to obtain the average marginal effect (AME); or evaluate the effect at mean covariates to obtain the marginal effect at the mean (MEM). The AME answers “what is the average change in this sample?” and is often easier to interpret than an artificial “average person” who may not exist.
<!-- bilingual-en:end -->

若模型含交互项，概率尺度上的组差或差异之差应由完整预测计算。非线性指数中的交互系数只描述 latent index 上的交互，不能直接当作概率尺度的交互效应。
<!-- bilingual-en:start -->
When the model contains interactions, group contrasts or differences in differences on the probability scale should be computed from complete predictions. The interaction coefficient in a nonlinear index describes interaction on the latent-index scale; it is not automatically the interaction effect on the probability scale.
<!-- bilingual-en:end -->

> [!source] 解释尺度核验
> Wooldridge Section 17.1 给出 $g(X'\beta)\beta_j$ 和二元解释变量的离散变化；Stata `margins` 手册明确区分连续变量的导数、factor variable 的离散变化、AME 与在指定协变量值处的预测。
> <!-- bilingual-en:start -->
> Wooldridge Section 17.1 derives $g(X'\beta)\beta_j$ and the discrete change for a binary regressor. Stata's `margins` manual explicitly distinguishes derivatives for continuous variables, discrete changes for factor variables, AMEs, and predictions at specified covariate values.
> <!-- bilingual-en:end -->

## 完整例子：培训与就业概率
<!-- bilingual-en:start -->
*Worked example: training and employment probability*
<!-- bilingual-en:end -->

假设 LPM 估计为
<!-- bilingual-en:start -->
Suppose the estimated LPM is
<!-- bilingual-en:end -->

$$
\widehat P(employed=1)=0.18+0.12\,trained+0.015\,experience.
$$

在相同经验下，培训与就业概率高 12 个百分点相关。10 年经验、未培训者的预测概率是 $0.18+0.015\times10=0.33$，培训者是 0.45。若经验很高，线性式可能超过 1；这暴露的是 LPM 的函数形式边界，不应把结果截断后假装模型已修复。
<!-- bilingual-en:start -->
At the same experience level, training is associated with an employment probability that is 12 percentage points higher. For an untrained worker with ten years of experience, the fitted probability is $0.18+0.015\times10=0.33$; for a trained worker it is 0.45. At high experience levels, the linear expression may exceed one. This exposes the LPM's functional-form boundary; truncating the prediction does not repair the model.
<!-- bilingual-en:end -->

再假设 logit 估计的线性指数为
<!-- bilingual-en:start -->
Now suppose the estimated logit index is
<!-- bilingual-en:end -->

$$
\hat\eta=-2+0.90\,trained+0.08\,experience.
$$

在 10 年经验处，未培训者的预测概率为 $\Lambda(-1.2)\approx0.231$，培训者为 $\Lambda(-0.3)\approx0.426$，离散变化约为 19.5 个百分点。培训的 odds ratio 是 $e^{0.90}\approx2.46$，但这绝不等于就业概率增加 146%。在 25 年经验处，同一个 0.90 系数对应的概率变化约为 $\Lambda(0.9)-\Lambda(0)\approx0.211$，说明概率效应依赖基准位置。
<!-- bilingual-en:start -->
At ten years of experience, the predicted probability is $\Lambda(-1.2)\approx0.231$ without training and $\Lambda(-0.3)\approx0.426$ with training, a discrete change of about 19.5 percentage points. The training odds ratio is $e^{0.90}\approx2.46$, but this emphatically does not mean that employment probability rises by 146 percent. At 25 years of experience, the same coefficient of 0.90 corresponds to a probability change of approximately $\Lambda(0.9)-\Lambda(0)\approx0.211$, showing that the probability effect depends on the starting point.
<!-- bilingual-en:end -->

以上三种数字回答不同问题：logit 系数 0.90 在 log-odds 尺度上，odds ratio 2.46 在 odds 尺度上，而 19.5 个百分点在概率尺度上。面向实质问题的报告通常应优先给预测概率或 AME，并说明比较的协变量取值。
<!-- bilingual-en:start -->
These three numbers answer different questions: the logit coefficient 0.90 is on the log-odds scale, the odds ratio 2.46 is on the odds scale, and 19.5 percentage points is on the probability scale. Substantive reporting should usually prioritize predicted probabilities or AMEs and state the covariate values used in the comparison.
<!-- bilingual-en:end -->

## 如何选择，以及模型会在哪里失败
<!-- bilingual-en:start -->
*How to choose, and where each model can fail*
<!-- bilingual-en:end -->

若目标是容易解释的平均概率差、模型含大量固定效应，或需要把交互项直接读成概率差，LPM 常是有价值的基准。若合法概率、明显的 S 形关系、个体风险预测或 likelihood-based 比较很重要，logit/probit 更自然。logit 与 probit 的主要差别是链接函数和尾部形状；很多数据中两者预测接近，选择应依据机制、惯例和预测表现，而不是比较原始系数大小。
<!-- bilingual-en:start -->
The LPM is often a valuable benchmark when the target is an easily interpreted average probability difference, the model contains many fixed effects, or interactions should be read directly as probability contrasts. Logit or probit is more natural when valid probability bounds, a visibly S-shaped relationship, individual risk prediction, or likelihood-based comparison matters. Logit and probit mainly differ in link and tail shape; their predictions are often similar, so choose based on mechanism, convention, and predictive performance rather than raw coefficient magnitudes.
<!-- bilingual-en:end -->

建模前检查事件比例与协变量支持。某个类别中全是 0 或全是 1 时会出现 complete or quasi-complete separation，logit/probit 的某些极大似然估计会向无穷发散；这不是“特别显著”，而是数据不足以给出有限系数。稀有事件、极端权重点和过多参数也会让概率估计不稳定。
<!-- bilingual-en:start -->
Inspect the event rate and covariate support before fitting. If a category contains only zeros or only ones, complete or quasi-complete separation can make some logit or probit maximum-likelihood estimates diverge toward infinity. This is not “extreme significance”; the data do not support a finite coefficient. Rare events, extreme leverage, and too many parameters can likewise destabilize probability estimates.
<!-- bilingual-en:end -->

诊断不能只看分类正确率，因为严重类别不平衡时“永远预测 0”也可能看似准确。应结合目标查看校准、概率分布、残差或影响点、out-of-sample 表现，以及实质相关的阈值指标。若研究目标是因果效应，还必须单独审查遗漏变量、反向因果和样本选择。
<!-- bilingual-en:start -->
Diagnostics should not rely solely on classification accuracy: with severe class imbalance, always predicting zero may appear accurate. Depending on the goal, inspect calibration, the distribution of fitted probabilities, residuals or influential observations, out-of-sample performance, and threshold metrics relevant to the application. For causal questions, separately assess omitted variables, reverse causality, and sample selection.
<!-- bilingual-en:end -->

## 二元结果与动态面板的边界
<!-- bilingual-en:start -->
*The boundary with binary and dynamic panel models*
<!-- bilingual-en:end -->

同一人或企业被重复观察时，简单横截面 logit/probit 的独立性标准误通常不再合适。至少要处理单位内相关；若加入单位固定效应，还会出现新的估计边界：短面板中的普通 fixed-effects logit 可用条件似然消去个体截距但会丢弃结果从不变化的单位，fixed-effects probit 没有同样简单的消元法，而直接加入大量个体 dummy 可能有 incidental-parameters bias。
<!-- bilingual-en:start -->
When the same person or firm is observed repeatedly, independence-based standard errors from a simple cross-sectional logit or probit are usually inappropriate. Within-unit dependence must at least be handled. Unit fixed effects introduce further boundaries: in short panels, conditional fixed-effects logit can eliminate individual intercepts but drops units whose outcomes never change; fixed-effects probit has no equally simple elimination; and directly adding many unit indicators can suffer from incidental-parameters bias.
<!-- bilingual-en:end -->

若模型还含滞后因变量，当前结果与过去冲击之间的机械联系使普通 FE 进一步产生短 $T$ 偏误。此时问题已超出本文件，应先明确估计对象，再进入动态面板或事件史方法，而不是在 LPM、logit、probit 三者中盲选。
<!-- bilingual-en:start -->
If the model also contains a lagged dependent variable, the mechanical link between the current outcome and past shocks creates additional short-$T$ bias in ordinary fixed effects. This is beyond the present note: define the estimand first, then move to dynamic-panel or event-history methods rather than choosing mechanically among LPM, logit, and probit.
<!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 用自己的话解释：为什么 $Y\in\{0,1\}$ 时回归目标可以写成概率？
<!-- bilingual-en:start -->
*Explain in your own words: why can the regression target be written as a probability when $Y\in\{0,1\}$?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 因为条件期望把 1 乘以成功概率、0 乘以失败概率，所以 $E(Y\mid X)=P(Y=1\mid X)$。三类模型都在估计这个条件概率，只是函数形式和解释方式不同。
> <!-- bilingual-en:start -->
> The conditional expectation multiplies one by the success probability and zero by the failure probability, so $E(Y\mid X)=P(Y=1\mid X)$. All three models estimate this conditional probability using different functional forms and interpretations.
> <!-- bilingual-en:end -->

### LPM 系数为 0.06 应怎样表述？稳健标准误解决了 LPM 的哪些问题？
<!-- bilingual-en:start -->
*How should an LPM coefficient of 0.06 be stated, and which LPM problems do robust standard errors solve?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 在给定模型下，解释变量增加 1 单位与成功概率增加 6 个百分点相关。稳健标准误处理由 $p(1-p)$ 带来的异方差推断，但不解决越界预测、常数边际效应或内生性。
> <!-- bilingual-en:start -->
> Under the fitted model, a one-unit increase in the regressor is associated with a six-percentage-point increase in success probability. Robust standard errors address heteroskedastic inference arising from $p(1-p)$; they do not fix out-of-range predictions, constant marginal effects, or endogeneity.
> <!-- bilingual-en:end -->

### Logit 中 $e^{\beta}=2$ 是否表示概率翻倍？
<!-- bilingual-en:start -->
*In a logit model, does $e^{\beta}=2$ mean that probability doubles?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不是。它表示其他变量不变时 odds 翻倍。概率变化取决于原来的概率，应通过两组预测概率或边际效应计算。
> <!-- bilingual-en:start -->
> No. It means that the odds double, holding other variables fixed. The probability change depends on the initial probability and should be calculated from two predicted probabilities or a marginal effect.
> <!-- bilingual-en:end -->

### 对二元解释变量，为什么应计算从 0 到 1 的离散变化，而不是只报告导数？
<!-- bilingual-en:start -->
*For a binary regressor, why should its zero-to-one discrete change be calculated rather than reporting only a derivative?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 二元变量没有可观察的无穷小变化；研究问题就是把状态从 0 切换到 1。非线性模型中这个完整变化为两个预测概率之差，并会随其他协变量而变。
> <!-- bilingual-en:start -->
> A binary regressor has no observable infinitesimal change; the substantive intervention switches its state from zero to one. In a nonlinear model, this complete change is the difference between two predicted probabilities and varies with the other covariates.
> <!-- bilingual-en:end -->

### 某组的结果全为 1，logit 给出极大系数。应如何诊断？
<!-- bilingual-en:start -->
*Every outcome in one group equals one, and logit returns an enormous coefficient. What should be diagnosed?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 检查 complete 或 quasi-complete separation、该组样本量和共同支持。极大系数可能意味着有限 MLE 不存在，而不是发现了无限大的真实效应。
> <!-- bilingual-en:start -->
> Check for complete or quasi-complete separation, the group's sample size, and common support. The enormous coefficient may indicate that a finite MLE does not exist, not that the true effect is infinite.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Wooldridge, Jeffrey M. *Introductory Econometrics: A Modern Approach*, 6th ed., Section 7.5 and Chapter 17.1：核验二元结果的条件期望、LPM 的异方差与越界、logit/probit 的 CDF 表达、极大似然、连续边际效应和二元变量离散变化。
- StataCorp, [Logistic regression manual](https://www.stata.com/manuals/rlogit.pdf)：核验 logit 的概率、log-odds、odds ratio、稳健与聚类推断选项。
- StataCorp, [Probit regression manual](https://www.stata.com/manuals/rprobit.pdf)：核验 probit 的标准正态 CDF 设定及相应推断。
- StataCorp, [Margins and marginal effects manual](https://www.stata.com/manuals/rmargins.pdf)：核验导数、factor 的离散变化、AME、MEM 和指定协变量处预测之间的区别。
- StataCorp, [`xtlogit` manual](https://www.stata.com/manuals/xtxtlogit.pdf)：核验 conditional FE logit 的定义，以及结果始终为 0 或始终为 1 的单位会因没有组内结果变化而被排除。
- [[02_Economy/01_Econometrics/10_虚拟变量.md]] 与 [[02_Economy/01_Econometrics/太白金星v计量.pdf]] 第 13 章（扫描页 161–164）：支持课程内 LPM、潜变量、probit 与 logit 的组织；核心定义和公式已对照 Wooldridge 与 Stata 手册复核。
<!-- bilingual-en:start -->
- Wooldridge, Jeffrey M. *Introductory Econometrics: A Modern Approach*, 6th ed., Section 7.5 and Chapter 17.1, supports the conditional expectation for a binary outcome, LPM heteroskedasticity and out-of-range predictions, the CDF form of logit and probit, maximum likelihood, continuous marginal effects, and discrete changes for binary regressors.
- StataCorp's [logistic regression manual](https://www.stata.com/manuals/rlogit.pdf) supports logit probabilities, log-odds, odds ratios, and robust or clustered inference options.
- StataCorp's [probit regression manual](https://www.stata.com/manuals/rprobit.pdf) supports the standard-normal CDF specification and its inference.
- StataCorp's [margins and marginal-effects manual](https://www.stata.com/manuals/rmargins.pdf) supports the distinction among derivatives, discrete changes for factors, AMEs, MEMs, and predictions at specified covariate values.
- StataCorp's [`xtlogit` manual](https://www.stata.com/manuals/xtxtlogit.pdf) supports the definition of conditional FE logit and the omission of units whose outcomes are always zero or always one because they contain no within-unit outcome variation.
- [[02_Economy/01_Econometrics/10_虚拟变量.md|Course note on indicator variables]] and Chapter 13 of the [[02_Economy/01_Econometrics/太白金星v计量.pdf|local econometrics course text]] (scan pages 161–164) support the course organization of LPM, latent-variable, probit, and logit material. Core definitions and formulas were rechecked against Wooldridge and the Stata manuals.
<!-- bilingual-en:end -->
