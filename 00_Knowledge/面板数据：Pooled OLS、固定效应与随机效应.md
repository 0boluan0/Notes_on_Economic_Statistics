---
aliases:
  - "Panel Data Models"
  - "Longitudinal Data Models"
  - "Fixed Effects and Random Effects"
  - "面板数据模型"
  - "FE and RE"
status: source-checked
---

# 面板数据：Pooled OLS、固定效应与随机效应
<!-- bilingual-en:start -->
*Panel data: pooled OLS, fixed effects, and random effects*
<!-- bilingual-en:end -->

> [!summary] 快速恢复
> **它解决什么：** 面板数据重复观察同一个人、企业或地区，使研究者能把“单位之间本来就不同”与“同一单位后来发生了变化”区分开。
> **具体锚点：** 比较不同人的工资与培训经历，能力差异会混进结果；固定效应改为比较“同一个人培训前后相对自身平均值的变化”，从而消去不随时间变化的能力。
> **核心难点：** pooled OLS、fixed effects（FE）与 random effects（RE）不是三个可随意替换的按钮。它们使用不同的组内/组间变异，并对未观测个体效应与解释变量的关系作出不同假设。
> **为什么重要：** FE 只能消去加性且时间不变的混杂，不能修复时变遗漏、反向因果或测量误差；普通逐行标准误也会忽略同一单位内相关。
> **继续：** 先认清 [[#面板结构与误差分解]]，再用 [[#三种估计量究竟在比较什么]] 选择估计量；政策处理随组别和时间变化时继续到 [[双重差分法（DID）]]。
> <!-- bilingual-en:start -->
> **What it solves:** Panel data observe the same person, firm, or region repeatedly, allowing researchers to separate “units were different to begin with” from “the same unit changed over time.”
> **Concrete anchor:** Comparing wages and training across different people mixes in ability differences. Fixed effects instead compare each person's before–after movement relative to their own mean, removing ability that does not change over time.
> **Central difficulty:** Pooled OLS, fixed effects (FE), and random effects (RE) are not interchangeable buttons. They use different combinations of within- and between-unit variation and make different assumptions about the relation between unobserved unit effects and regressors.
> **Why it matters:** FE removes only additive, time-invariant confounding. It does not repair time-varying omissions, reverse causality, or measurement error. Ordinary row-level standard errors also ignore dependence within a unit.
> **Continue with:** First understand the [[#面板结构与误差分解|panel structure and error decomposition]], then use [[#三种估计量究竟在比较什么|what the three estimators actually compare]] to choose an estimator. When policy treatment varies across groups and time, continue to [[双重差分法（DID）|difference-in-differences]].
> <!-- bilingual-en:end -->

> [!source] 本节依据
> - Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., Chapters 13–14：核验 pooled OLS、first difference、within FE、RE、correlated random effects 和聚类推断。
> - MIT OpenCourseWare, [14.382 Lecture 8: Linear Panel Data Models Under Strict and Weak Exogeneity](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/049489cf2ae5d050409ee2c5cbd5346f_MIT14_382S17_lec8.pdf)：核验严格外生、个体效应、差分与 FE 的识别边界。
> - Stata 官方 [`xtreg` manual](https://www.stata.com/manuals/xtxtreg.pdf)：核验 pooled/BE/FE/RE/CRE 的估计对象、变换和输出。
> - [[02_Economy/01_Econometrics/13_面板数据模型.md]] 与 [[02_Economy/01_Econometrics/太白金星v计量.pdf]] 第 15 章：核对课程符号、讲授顺序与实践语境。
> <!-- bilingual-en:start -->
> - Wooldridge, *Introductory Econometrics: A Modern Approach*, 6th ed., Chapters 13–14, supports pooled OLS, first differences, within FE, RE, correlated random effects, and clustered inference.
> - MIT OpenCourseWare, [14.382 Lecture 8: Linear Panel Data Models Under Strict and Weak Exogeneity](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/049489cf2ae5d050409ee2c5cbd5346f_MIT14_382S17_lec8.pdf), supports strict exogeneity, individual effects, differencing, and the identifying boundaries of FE.
> - Stata's official [`xtreg` manual](https://www.stata.com/manuals/xtxtreg.pdf) supports the estimands, transformations, and output for pooled, between, FE, RE, and CRE estimators.
> - [[02_Economy/01_Econometrics/13_面板数据模型.md|Course note on panel-data models]] and Chapter 15 of the [[02_Economy/01_Econometrics/太白金星v计量.pdf|local econometrics course text]] establish the course notation, sequence, and applied context.
> <!-- bilingual-en:end -->

## 面板结构与误差分解
<!-- bilingual-en:start -->
*Panel structure and error decomposition*
<!-- bilingual-en:end -->

最常见的线性面板模型写成
<!-- bilingual-en:start -->
The standard linear panel model is
<!-- bilingual-en:end -->

$$
y_{it}=\alpha+x_{it}'\beta+a_i+\lambda_t+u_{it},
\qquad i=1,\ldots,N,\;t=1,\ldots,T.
$$

$a_i$ 是单位 $i$ 不随时间变化、但研究者没有完全观察到的特征，例如稳定能力、企业文化或地理禀赋；$\lambda_t$ 是同一时期冲击所有单位的共同时间效应，例如全国通胀或宏观衰退；$u_{it}$ 是随单位和时间变化的剩余冲击。$x_{it}$ 可以含随时间变化和不变化的变量。
<!-- bilingual-en:start -->
Here $a_i$ is an incompletely observed characteristic of unit $i$ that remains constant over time, such as stable ability, firm culture, or geographic endowment. The common time effect $\lambda_t$ captures shocks affecting every unit in a period, such as national inflation or a macroeconomic recession. The idiosyncratic error $u_{it}$ varies across both units and time. The regressors $x_{it}$ may include time-varying and time-invariant variables.
<!-- bilingual-en:end -->

balanced panel 中每个单位都有相同的 $T$ 个时期；unbalanced panel 允许进入、退出或缺失。非平衡本身不必然造成偏误，但缺失机制若与未观察结果或冲击相关，就会引入选择问题，不能仅靠 FE 解决。
<!-- bilingual-en:start -->
In a balanced panel, every unit is observed for the same $T$ periods; an unbalanced panel permits entry, exit, or missing observations. Imbalance alone need not create bias, but if missingness depends on unobserved outcomes or shocks, it creates a selection problem that FE alone cannot solve.
<!-- bilingual-en:end -->

面板数据的价值不在于“样本行更多”，而在于能利用单位内部的变化并显式处理同一单位内相关。只有当关键解释变量在单位内部确实变化、且这种变化满足相应外生条件时，组内信息才能识别 $\beta$。
<!-- bilingual-en:start -->
The value of panel data is not merely “more rows.” It lies in using within-unit changes and explicitly handling within-unit dependence. Within variation identifies $\beta$ only when the key regressors actually change within units and those changes satisfy the relevant exogeneity conditions.
<!-- bilingual-en:end -->

## Pooled OLS：把所有行堆在一起
<!-- bilingual-en:start -->
*Pooled OLS: stacking all rows together*
<!-- bilingual-en:end -->

pooled OLS 直接把所有 $(i,t)$ 观测合并回归，等价于把复合误差写成 $v_{it}=a_i+u_{it}$。若 $a_i$ 与 $x_{it}$ 相关，例如能力高的人更容易接受培训，那么 $E(v_{it}\mid x_{it})\neq0$，pooled OLS 会把不变能力的一部分误算成培训作用。
<!-- bilingual-en:start -->
Pooled OLS simply stacks all $(i,t)$ observations and treats the composite error as $v_{it}=a_i+u_{it}$. If $a_i$ is correlated with $x_{it}$—for example, more able people are more likely to receive training—then $E(v_{it}\mid x_{it})\neq0$, and pooled OLS attributes part of stable ability to training.
<!-- bilingual-en:end -->

即使复合误差与解释变量不相关，$a_i$ 在同一单位每一期都出现，使 $v_{it}$ 在单位内相关。忽略这一点的普通 OLS 标准误通常不可信；至少应按单位聚类，或使用与抽样和处理分配层级相匹配的推断方法。
<!-- bilingual-en:start -->
Even if the composite error is uncorrelated with regressors, the same $a_i$ appears in every period for a unit, inducing within-unit dependence in $v_{it}$. Conventional OLS standard errors that ignore this dependence are generally unreliable. At a minimum, cluster by unit or use inference aligned with the sampling and treatment-assignment level.
<!-- bilingual-en:end -->

pooled OLS 并非永远错误。若研究目标只是清楚标记为描述性的总体关系，或有可信理由认为单位效应与所有解释变量正交，它可作为基准；但必须说明它同时使用组间和组内变异，不能自动获得 FE 的“控制不变异质性”解释。
<!-- bilingual-en:start -->
Pooled OLS is not always wrong. It can be a benchmark for a clearly descriptive population relationship or when there is a credible reason to regard unit effects as orthogonal to every regressor. But it combines between- and within-unit variation and does not automatically inherit FE's interpretation of controlling for time-invariant heterogeneity.
<!-- bilingual-en:end -->

## 固定效应：只利用单位内部变化
<!-- bilingual-en:start -->
*Fixed effects: using only within-unit variation*
<!-- bilingual-en:end -->

对每个单位取时间均值并相减，得到 within transformation：
<!-- bilingual-en:start -->
Taking each unit's time mean and subtracting it gives the within transformation:
<!-- bilingual-en:end -->

$$
y_{it}-\bar y_i=(x_{it}-\bar x_i)'\beta+(\lambda_t-\bar\lambda_i)+(u_{it}-\bar u_i).
$$

因为 $a_i-\bar a_i=0$，任何加性且时间不变的未观测特征都被消去。线性模型中，直接加入 $N-1$ 个单位 dummy 的 least-squares dummy-variable（LSDV）估计与 within FE 对 $\beta$ 给出相同结果；两者只是计算和展示方式不同。
<!-- bilingual-en:start -->
Because $a_i-\bar a_i=0$, every additive time-invariant unobserved characteristic disappears. In a linear model, least-squares dummy variables (LSDV), which include $N-1$ unit indicators, and the within FE transformation produce the same estimate of $\beta$. They differ only in computation and presentation.
<!-- bilingual-en:end -->

FE 允许 $a_i$ 与任意时期的 $x_{it}$ 相关，但通常要求 idiosyncratic error 对整个解释变量路径严格外生：
<!-- bilingual-en:start -->
FE permits $a_i$ to be correlated with $x_{it}$ in any period, but ordinarily requires the idiosyncratic error to be strictly exogenous with respect to the entire regressor history:
<!-- bilingual-en:end -->

$$
E(u_{it}\mid x_{i1},x_{i2},\ldots,x_{iT},a_i)=0
\quad\text{for every }t.
$$

这不仅排除同期遗漏，还排除“今天的工资冲击导致明天去培训”之类反馈。若只满足当期外生而未来 $x$ 会响应当前 $u$，普通 FE 也可能不一致。
<!-- bilingual-en:start -->
This rules out not only contemporaneous omissions but also feedback such as “today's wage shock causes training tomorrow.” If only contemporaneous exogeneity holds and future $x$ responds to current $u$, ordinary FE can still be inconsistent.
<!-- bilingual-en:end -->

FE 的系数来自 within variation，所以单位内不变化的变量（出生地区、固定性别编码、长期地理属性）在去均值后恒为 0，无法与 $a_i$ 分开估计。一个变量若几乎不在单位内变化，虽然形式上可估计，信息也会很弱，测量误差的相对影响往往更大。
<!-- bilingual-en:start -->
FE coefficients come from within variation. A variable that never changes within a unit—birth region, a fixed gender code, or a permanent geographic attribute—becomes zero after demeaning and cannot be estimated separately from $a_i$. If a regressor changes very little within units, it may be formally estimable but weakly informed, often magnifying the relative impact of measurement error.
<!-- bilingual-en:end -->

加入时间 dummy 得到 two-way fixed effects，可消去 $\lambda_t$ 这类共同时间冲击。但它不能消去因单位而异的时变混杂，例如某企业在接受培训同年更换管理层；也不能自动解决 staggered treatment 下效应异质性带来的加权问题，相关设计应进入 [[双重差分法（DID）]]。
<!-- bilingual-en:start -->
Adding period indicators yields two-way fixed effects and removes common shocks such as $\lambda_t$. It does not remove time-varying confounders that differ by unit—for example, a firm replacing management in the same year it adopts training. Nor does it automatically solve weighting problems from heterogeneous effects under staggered treatment; those designs belong in [[双重差分法（DID）|difference-in-differences]].
<!-- bilingual-en:end -->

> [!source] FE 条件核验
> MIT 14.382 Lecture 8 和 Wooldridge Chapter 14 都把严格外生写在整个 $X_i$ 路径上，并区分个体效应 $a_i$ 与 idiosyncratic error $u_{it}$。两者共同支持：FE 允许 $a_i$ 与 $X_i$ 相关，但不能容忍任意的 $u_{it}$—未来解释变量反馈。
> <!-- bilingual-en:start -->
> MIT 14.382 Lecture 8 and Wooldridge Chapter 14 both formulate strict exogeneity with respect to the entire $X_i$ path and distinguish the individual effect $a_i$ from the idiosyncratic error $u_{it}$. Together they support the conclusion that FE permits correlation between $a_i$ and $X_i$ but not unrestricted feedback from $u_{it}$ to future regressors.
> <!-- bilingual-en:end -->

## 随机效应：在更强正交条件下结合组内与组间信息
<!-- bilingual-en:start -->
*Random effects: combining within and between information under stronger orthogonality*
<!-- bilingual-en:end -->

RE 不是简单地说“样本中的人是随机抽来的”。关键条件是未观测单位效应与整个解释变量历史不相关，典型写法为 $E(a_i\mid X_i)=0$，并配合 idiosyncratic error 的外生和方差结构。RE 用 GLS 对变量做 quasi-demeaning：只减去单位均值的一部分，而不像 FE 完全减去。
<!-- bilingual-en:start -->
RE does not merely mean that “people were randomly sampled.” Its key condition is that the unobserved unit effect is uncorrelated with the entire regressor history, commonly written $E(a_i\mid X_i)=0$, together with exogeneity and variance conditions for the idiosyncratic error. RE uses GLS to quasi-demean variables, subtracting a fraction of each unit mean rather than removing it completely as FE does.
<!-- bilingual-en:end -->

因此 RE 同时利用单位之间和单位内部的差异，条件成立时通常比 FE 更有效率，并能估计时间不变变量的系数；代价是 $E(a_i\mid X_i)=0$ 在许多社会科学问题中很强。若稳定能力既影响教育又影响工资，RE 的正交条件就可疑。
<!-- bilingual-en:start -->
RE therefore uses both between- and within-unit differences. When its conditions hold, it is often more efficient than FE and can estimate coefficients on time-invariant regressors. The cost is that $E(a_i\mid X_i)=0$ is a strong assumption in many social-science applications. If stable ability affects both education and wages, the RE orthogonality condition is doubtful.
<!-- bilingual-en:end -->

correlated random effects（Mundlak approach）在 RE 方程中加入时变解释变量的单位均值，使 $a_i$ 与 $X_i$ 的相关性通过这些均值参数化。在线性平衡面板中，时变变量的系数可复现 FE 结果，同时还能展示 between components；检验单位均值项是否共同为 0 可作为 RE 正交条件的诊断，但不能把模型选择简化为一次 p-value。
<!-- bilingual-en:start -->
Correlated random effects, or the Mundlak approach, adds unit means of time-varying regressors to the RE equation, parameterizing correlation between $a_i$ and $X_i$ through those means. In a linear balanced panel, coefficients on time-varying regressors reproduce the FE result while retaining between components. Testing whether the unit-mean terms are jointly zero helps diagnose the RE orthogonality condition, but model choice should not be reduced to a single p-value.
<!-- bilingual-en:end -->

传统 Hausman test 比较在原假设下 FE 与 RE 都一致、但 RE 更有效时两组系数的系统差异。拒绝常被解释为反对 RE；不拒绝却不证明 $E(a_i\mid X_i)=0$，尤其在样本小、模型设定不同或协方差估计不匹配时。优先从研究机制判断相关性，再用 Hausman 或 CRE 作为诊断。
<!-- bilingual-en:start -->
The traditional Hausman test compares FE and RE coefficients under a null in which both are consistent but RE is more efficient. Rejection is commonly interpreted as evidence against RE. Failure to reject does not prove $E(a_i\mid X_i)=0$, especially with a small sample, differing specifications, or incompatible covariance estimates. Start with a substantive argument about correlation, then use Hausman or CRE as a diagnostic.
<!-- bilingual-en:end -->

## 三种估计量究竟在比较什么
<!-- bilingual-en:start -->
*What the three estimators actually compare*
<!-- bilingual-en:end -->

| 方法 | 使用的变异 | 对 $a_i$ 与 $X_i$ 的关键要求 | 能否估计时间不变变量 | 常见风险 |
|---|---|---|---|---|
| pooled OLS | 混合组间与组内 | 复合误差与解释变量正交 | 可以 | 遗漏相关单位效应；普通 SE 忽略组内相关 |
| FE / within | 只用组内偏离 | 允许 $a_i$ 与 $X_i$ 相关；要求对整条 $X_i$ 路径严格外生 | 不可以 | 时变混杂、反馈、弱组内变异、测量误差 |
| RE / GLS | 加权结合组内与组间 | 进一步要求 $E(a_i\mid X_i)=0$ | 可以 | 正交条件不可信时系数不一致 |
<!-- bilingual-en:start -->
| Method | Variation used | Key requirement on $a_i$ and $X_i$ | Estimates time-invariant regressors? | Common risk |
|---|---|---|---|---|
| Pooled OLS | Mixes between and within | Composite error orthogonal to regressors | Yes | Correlated unit effects omitted; ordinary SE ignores within dependence |
| FE / within | Within-unit deviations only | Allows $a_i$–$X_i$ correlation; requires strict exogeneity over the full $X_i$ path | No | Time-varying confounding, feedback, weak within variation, measurement error |
| RE / GLS | Weighted combination of within and between | Additionally requires $E(a_i\mid X_i)=0$ | Yes | Inconsistent coefficients if orthogonality is implausible |
<!-- bilingual-en:end -->

选择方法时先问“哪个变化识别我想要的效应”，再问“使这种比较有效的假设是否可信”。若研究问题是“同一个人接受培训后工资怎样变”，FE 的 within estimand 更贴近问题；若问题包含教育这类时间不变变量，则需要可信的 RE/CRE 结构、between design 或其他识别策略，不能要求 FE 凭空估计。
<!-- bilingual-en:start -->
Choose a method by first asking which variation identifies the desired effect, then whether the assumptions validating that comparison are credible. If the question is “how does the same person's wage change after training?”, the FE within estimand is closer to the target. If the question concerns a time-invariant variable such as education, a credible RE/CRE structure, a between-unit design, or another identification strategy is required; FE cannot estimate it from nonexistent within variation.
<!-- bilingual-en:end -->

## 完整例子：为什么 within comparison 会改写结论
<!-- bilingual-en:start -->
*Worked example: why a within comparison changes the conclusion*
<!-- bilingual-en:end -->

考虑两个工人、两个时期的极简数据。高能力工人 A 的培训状态从 0 变 1，工资从 20 变 22；低能力工人 B 始终未培训，工资保持 12。若把四行 pooled 在一起，唯一的“已培训”工资是 22，三行“未培训”平均工资是 $(20+12+12)/3\approx14.67$，表面差距约 7.33。
<!-- bilingual-en:start -->
Consider a minimal dataset with two workers and two periods. High-ability worker A moves from untrained to trained and wages rise from 20 to 22. Low-ability worker B remains untrained and wages stay at 12. If all four rows are pooled, the only trained wage is 22, while the three untrained wages average $(20+12+12)/3\approx14.67$, producing an apparent gap of about 7.33.
<!-- bilingual-en:end -->

within FE 先去掉每个人的平均水平。A 的训练偏离是 $(-0.5,0.5)$、工资偏离是 $(-1,1)$；B 两个变量的偏离都为 0。FE 斜率因此是
<!-- bilingual-en:start -->
Within FE first removes each person's mean. Worker A's training deviations are $(-0.5,0.5)$ and wage deviations are $(-1,1)$; both deviations are zero for worker B. The FE slope is therefore
<!-- bilingual-en:end -->

$$
\hat\beta_{FE}
=\frac{(-0.5)(-1)+(0.5)(1)}{(-0.5)^2+(0.5)^2}
=2.
$$

这个 2 来自 A 自身的变化，不再把 A 与 B 的稳定工资水平差异算成培训效应。它仍不是无条件的因果证明：这个玩具样本只有一个 switching unit；若 A 同期晋升，时变遗漏仍会污染结果。
<!-- bilingual-en:start -->
This estimate of two comes from A's own change and no longer attributes the stable wage-level difference between A and B to training. It is still not unconditional causal proof: this toy sample has only one switching unit, and a simultaneous promotion for A would remain a time-varying omitted variable.
<!-- bilingual-en:end -->

若第二期所有人的工资都因通胀上升 1，再只做单位 FE 会把共同时间变化留在误差中；加入第二期 dummy 可控制该共同冲击。若通胀冲击对行业不同，则单一时间效应仍不足，必须建模行业×时期冲击或采用更可信的比较设计。
<!-- bilingual-en:start -->
If every worker's wage rises by one in period two because of inflation, unit FE alone leaves the common time movement in the error; adding a period-two indicator controls for that common shock. If inflation differs by industry, a single time effect remains insufficient, requiring industry-by-period shocks or a more credible comparison design.
<!-- bilingual-en:end -->

## 推断、诊断与失败边界
<!-- bilingual-en:start -->
*Inference, diagnostics, and failure boundaries*
<!-- bilingual-en:end -->

面板中同一单位的 $u_{it}$ 常有任意序列相关和异方差，因此通常按单位聚类标准误。若政策在州层面分配、数据却是个人面板，应认真考虑按州而非个人聚类；聚类层级取决于处理分配和误差相关来源。聚类数很少时，普通 cluster-robust 渐近近似也可能失真，需要小样本修正或适合研究设计的随机化推断。
<!-- bilingual-en:start -->
Within a panel unit, $u_{it}$ often exhibits arbitrary serial correlation and heteroskedasticity, so standard errors are commonly clustered by unit. If policy is assigned at the state level while the panel observes individuals, clustering by state rather than person may be necessary. The cluster level follows treatment assignment and the source of error dependence. With few clusters, ordinary cluster-robust asymptotics can also be unreliable, calling for small-sample corrections or design-appropriate randomization inference.
<!-- bilingual-en:end -->

报告 FE 前检查关键变量的 within variation、switchers 数量、面板是否失衡以及结果对时间效应和趋势控制的敏感性。只报告 overall $R^2$ 会混合组间水平差异；解释 FE 时更相关的是 within fit 和目标系数来自哪些单位内变化。
<!-- bilingual-en:start -->
Before reporting FE, inspect within variation in key regressors, the number of switchers, panel imbalance, and sensitivity to period effects or trend controls. Reporting only overall $R^2$ mixes between-unit level differences. For FE, the within fit and the unit-level changes identifying the target coefficient are more informative.
<!-- bilingual-en:end -->

FE 会放大经典测量误差，因为去均值后真实信号可能很小；单位特有趋势若同时影响 $x$ 和 $y$ 也会造成时变混杂。加入过多趋势可以吸收几乎全部识别变异，所以每个控制都要说明它阻断了什么混杂路径，以及还留下多少支持。
<!-- bilingual-en:start -->
FE can magnify classical measurement error because demeaning may leave little true signal. Unit-specific trends that affect both $x$ and $y$ also create time-varying confounding. Adding too many trends can absorb nearly all identifying variation, so each control should be justified by the confounding path it blocks and the support that remains.
<!-- bilingual-en:end -->

严格外生失败的典型诊断是反馈：企业在本期遭遇负面销售冲击后，下期增加广告；工人本期工资异常下降后，下期参加培训。普通 FE 不能修复这种 $u_{it}\to x_{i,t+1}$ 关系，需要预定变量假设、工具变量或动态模型。
<!-- bilingual-en:start -->
A canonical failure of strict exogeneity is feedback: a firm increases advertising next period after a negative sales shock this period, or a worker enters training next period after an unusually low wage. Ordinary FE does not repair the path $u_{it}\to x_{i,t+1}$; predetermined-variable assumptions, instruments, or dynamic models are required.
<!-- bilingual-en:end -->

## 二元结果与动态面板的边界
<!-- bilingual-en:start -->
*The boundary with binary outcomes and dynamic panels*
<!-- bilingual-en:end -->

若 $y_{it}$ 为 0/1，线性 FE 是固定效应 LPM，系数仍是组内概率百分点变化并应按单位聚类；它保留易解释性但仍可能越界。非线性 fixed-effects logit/probit 面临条件似然、无结果变化单位被丢弃和 incidental-parameters bias 等额外问题，详见 [[二元因变量模型：LPM、Logit 与 Probit#二元结果与动态面板的边界]]。
<!-- bilingual-en:start -->
If $y_{it}$ is binary, linear FE becomes a fixed-effects LPM. Its coefficient remains a within-unit percentage-point change and standard errors should be clustered by unit, but fitted values can still leave the unit interval. Nonlinear fixed-effects logit or probit introduces additional issues involving conditional likelihood, the loss of units whose outcomes never change, and incidental-parameters bias; see [[二元因变量模型：LPM、Logit 与 Probit#二元结果与动态面板的边界|binary and dynamic panel boundaries]].
<!-- bilingual-en:end -->

若右侧含 $y_{i,t-1}$，去均值后的滞后因变量会与去均值后的误差相关，短 $T$ 下产生 Nickell bias。增加单位和时间固定效应不能消除这种机械相关；需要根据 $N,T$ 和外生性结构选择 Arellano–Bond 等动态面板方法或其他估计策略。
<!-- bilingual-en:start -->
If $y_{i,t-1}$ appears on the right-hand side, the demeaned lagged outcome is correlated with the demeaned error, producing Nickell bias when $T$ is short. Adding unit and time fixed effects does not eliminate this mechanical correlation. Depending on $N$, $T$, and the exogeneity structure, a dynamic-panel method such as Arellano–Bond or another estimator is required.
<!-- bilingual-en:end -->

> [!source] 动态边界核验
> MIT 14.382 Lecture 8 明确指出，滞后因变量作为解释变量会违反普通 FE 所用的严格外生条件；Nickell, “Biases in Dynamic Models with Fixed Effects,” *Econometrica* 49(6), 1981, pp. 1417–1426，给出短 $T$ 动态固定效应估计的偏误结果。
> <!-- bilingual-en:start -->
> MIT 14.382 Lecture 8 explicitly notes that a lagged dependent variable violates the strict-exogeneity condition used by ordinary FE. Nickell, “Biases in Dynamic Models with Fixed Effects,” *Econometrica* 49(6), 1981, pp. 1417–1426, derives the bias of dynamic fixed-effects estimation when $T$ is short.
> <!-- bilingual-en:end -->

## 最小自检
<!-- bilingual-en:start -->
*Minimum self-check*
<!-- bilingual-en:end -->

### 用自己的话解释：FE 为什么能消去稳定能力，却不能消去一次晋升？
<!-- bilingual-en:start -->
*Explain in your own words: why can FE remove stable ability but not a one-time promotion?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 稳定能力作为 $a_i$ 每期相同，去单位均值后恰好为 0；晋升随时间变化，仍留在去均值误差中。若晋升又与培训相关，就会形成时变混杂。
> <!-- bilingual-en:start -->
> Stable ability enters as the same $a_i$ every period and becomes exactly zero after unit demeaning. A promotion changes over time and remains in the demeaned error; if it is also related to training, it creates time-varying confounding.
> <!-- bilingual-en:end -->

### 为什么 RE 的关键不在于“个体是否随机抽样”？
<!-- bilingual-en:start -->
*Why is RE not primarily about whether individuals were randomly sampled?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> RE 的核心识别要求是未观测单位效应 $a_i$ 与整个解释变量路径 $X_i$ 正交。随机抽样本身并不能保证能力、企业文化等 $a_i$ 与教育、培训或投资不相关。
> <!-- bilingual-en:start -->
> The core RE identification condition is orthogonality between the unobserved unit effect $a_i$ and the full regressor history $X_i$. Random sampling alone does not ensure that ability, firm culture, or other components of $a_i$ are unrelated to education, training, or investment.
> <!-- bilingual-en:end -->

### 一个解释变量从不在单位内部变化，为什么 FE 不能估计它？
<!-- bilingual-en:start -->
*Why can FE not estimate a regressor that never changes within a unit?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> FE 只使用 $x_{it}-\bar x_i$。时间不变变量的这个差恒为 0，与单位固定效应无法分开，因此数据中没有识别其独立系数的组内变化。
> <!-- bilingual-en:start -->
> FE uses only $x_{it}-\bar x_i$. This difference is always zero for a time-invariant regressor, so it cannot be separated from the unit fixed effect. The data contain no within-unit variation identifying its coefficient.
> <!-- bilingual-en:end -->

### 工资本期意外下降会促使工人下期参加培训。哪项 FE 条件失败，为什么？
<!-- bilingual-en:start -->
*An unexpected wage decline this period causes a worker to enter training next period. Which FE condition fails, and why?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 严格外生失败，因为当前误差 $u_{it}$ 影响未来解释变量 $x_{i,t+1}$。单位去均值不会切断这条反馈路径，普通 FE 可能不一致。
> <!-- bilingual-en:start -->
> Strict exogeneity fails because the current error $u_{it}$ affects the future regressor $x_{i,t+1}$. Unit demeaning does not break this feedback path, so ordinary FE may be inconsistent.
> <!-- bilingual-en:end -->

### 为什么“不拒绝 Hausman test”不等于已经证明 RE 正确？
<!-- bilingual-en:start -->
*Why does failure to reject a Hausman test not prove that RE is correct?*
<!-- bilingual-en:end -->

> [!answer]- 答案
> 不拒绝可能来自检验力不足、协方差估计不稳或两模型设定不完全可比；它只是没有发现足够大的系统差异。$E(a_i\mid X_i)=0$ 仍需实质机制支持，并可用 CRE 等方式进一步诊断。
> <!-- bilingual-en:start -->
> Nonrejection may reflect low power, unstable covariance estimates, or incompletely comparable specifications; it only means that a sufficiently large systematic difference was not detected. The condition $E(a_i\mid X_i)=0$ still requires substantive support and can be further diagnosed with CRE.
> <!-- bilingual-en:end -->

## 来源与核验
<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- Wooldridge, Jeffrey M. *Introductory Econometrics: A Modern Approach*, 6th ed., Chapters 13–14：核验面板误差分解、一阶差分、within FE、严格外生、聚类标准误、RE quasi-demeaning、Hausman 与 correlated random effects。
- MIT OpenCourseWare, [14.382 Econometrics, Lecture 8](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/049489cf2ae5d050409ee2c5cbd5346f_MIT14_382S17_lec8.pdf)：核验严格/弱外生下线性面板模型、个体效应与解释变量相关时的 FE/差分逻辑，以及反馈造成的边界。
- StataCorp, [`xtreg` manual](https://www.stata.com/manuals/xtxtreg.pdf)：核验 FE、BE、RE 与 CRE 的软件定义、within/between/overall 分解、面板变换和 cluster-robust 推断选项。
- Nickell, Stephen. “Biases in Dynamic Models with Fixed Effects.” *Econometrica* 49(6), 1981, 1417–1426：核验短 $T$、含滞后因变量的固定效应估计偏误。
- [[02_Economy/01_Econometrics/13_面板数据模型.md]] 与 [[02_Economy/01_Econometrics/太白金星v计量.pdf]] 第 15 章（扫描页 169–176）：支持本课程的 pooled OLS—FE—RE 结构；核心假设、公式和解释已与 Wooldridge、MIT 和 Stata 手册逐项复核。
<!-- bilingual-en:start -->
- Wooldridge, Jeffrey M. *Introductory Econometrics: A Modern Approach*, 6th ed., Chapters 13–14, supports the panel error decomposition, first differences, within FE, strict exogeneity, clustered standard errors, RE quasi-demeaning, Hausman comparisons, and correlated random effects.
- MIT OpenCourseWare, [14.382 Econometrics, Lecture 8](https://ocw.mit.edu/courses/14-382-econometrics-spring-2017/049489cf2ae5d050409ee2c5cbd5346f_MIT14_382S17_lec8.pdf), supports linear panel models under strict and weak exogeneity, FE/differencing when individual effects correlate with regressors, and the boundary created by feedback.
- StataCorp's [`xtreg` manual](https://www.stata.com/manuals/xtxtreg.pdf) supports the software definitions of FE, BE, RE, and CRE, within/between/overall decompositions, panel transformations, and cluster-robust inference options.
- Nickell, Stephen. “Biases in Dynamic Models with Fixed Effects.” *Econometrica* 49(6), 1981, 1417–1426, supports the short-$T$ bias result for fixed-effects models containing a lagged dependent variable.
- [[02_Economy/01_Econometrics/13_面板数据模型.md|Course note on panel-data models]] and Chapter 15 of the [[02_Economy/01_Econometrics/太白金星v计量.pdf|local econometrics course text]] (scan pages 169–176) support the course's pooled OLS–FE–RE structure. Core assumptions, formulas, and interpretations were rechecked against Wooldridge, MIT, and the Stata manual.
<!-- bilingual-en:end -->
