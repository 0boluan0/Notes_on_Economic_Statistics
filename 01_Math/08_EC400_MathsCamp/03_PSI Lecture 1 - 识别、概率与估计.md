---
aliases:
  - EC400 PSI Lecture 1 Foundations
---
i
# PSI Lecture 1：从识别到概率与估计

<!-- bilingual-en:start -->
*From identification to probability and estimation*
<!-- bilingual-en:end -->

[[Lecture 1 - Foundations.pdf|课程 slides]] · [[PSI Lecture 1 - Foundations - 手写笔记.pdf|手写原稿（9 页）]] · [[PSI Lecture 1 - Foundations - Claude课堂记录|课堂记录原文]] · [[01_Math/08_EC400_MathsCamp/00_课程总览|课程阅读路径]]

本笔记按 Lecture 1 的讲课顺序展开：先问数据能识别什么，再问怎样从有限样本估计，随后用概率、抽样分布和估计量性质检查这条推断链。行政信息见课程资料索引；正文覆盖 slides 8–74。手写中要求补全的分位数例子、抽样分布组图、一致性定义、MSE 推导和两个反例都放回对应位置。

<!-- bilingual-en:start -->
The note follows Lecture 1: what the data can identify, how a finite sample can estimate it, and how probability, sampling distributions, and estimator properties justify the inference. The main text covers slides 8–74; administrative information remains in the materials index. Worked quantiles, the sampling-distribution figures, consistency, the MSE derivation, and both counterexamples appear in their lecture positions.
<!-- bilingual-en:end -->

## 1. 先明确问题：描述、因果与预测（slides 8–10）

<!-- bilingual-en:start -->
*Description, causality, and prediction*
<!-- bilingual-en:end -->

回归式 $Y_i=\alpha+\beta X_i+\varepsilon_i$ 先是一种表示变量关系的方式。$i$ 是[[观察单位]]的编号；$Y_i$ 是结果变量；$X_i$ 是解释变量；$\alpha$ 是截距；$\beta$ 是斜率；$\varepsilon_i$ 是该直线没有解释的部分。“因果效应”不是 $\beta$ 这个字母自带的含义。[[预测与因果目标]]区分三种问题：描述当前数据中的关联、改变某个因素会怎样、未来观察会怎样。它们可能使用相似公式，却需要不同依据。

<!-- bilingual-en:start -->
In the regression equation, i indexes the unit, Y is the outcome, X the explanatory variable, alpha the intercept, beta the slope, and epsilon the part not explained by the line. The symbol beta does not itself imply causality. [[预测与因果目标|Prediction and causal targets]] distinguish description of current associations, consequences of interventions, and future outcomes. Similar formulas can serve different aims and require different justification.
<!-- bilingual-en:end -->

[[识别与估计|识别（identification）]]先问：假如完全知道可观测变量的总体分布，能否唯一确定想知道的量？[[估计对象三分|估计（estimation）]]再问：现在只有一个有限样本，采用什么计算规则逼近已明确的目标？例如，完全知道报名者与未报名者成绩的分布，仍未必知道同一批人“报名与不报名”的差；收集更多同类观测只会把原有组间差估得更准。

<!-- bilingual-en:start -->
[[识别与估计|Identification]] asks whether complete knowledge of the observable distribution uniquely determines the desired quantity. [[估计对象三分|Estimation]] asks which finite-sample rule approximates that target. Perfect knowledge of outcomes among enrollees and non-enrollees may still leave the effect on the same people unknown. More data can estimate the existing group contrast more precisely without identifying that effect.
<!-- bilingual-en:end -->

## 2. 随机试验的语言：单位、处理与潜在结果（slides 11–17）

<!-- bilingual-en:start -->
*Units, treatment, and potential outcomes*
<!-- bilingual-en:end -->

课程用印度 Balsakhi 补习项目作背景：学校—年级单元被随机分配是否获得辅导，结果用测验成绩衡量。这里先用简化的二元分配模型解释逻辑，后面的数字表是教学假想数据。[[总体与样本]]规定研究面向的人群和实际观察；随机抽样决定谁进入样本，随机分配决定样本中的谁进入某个处理状态。两者不能互换。

<!-- bilingual-en:start -->
The Balsakhi example assigns tutoring to school-grade cells and measures test scores. The following binary model explains the logic; the numerical tables are hypothetical teaching data. [[总体与样本|Population and sample]] distinguish the target population from the observed units. Random sampling selects the sample, while random assignment allocates treatment states.
<!-- bilingual-en:end -->

用 $D_i\in\{0,1\}$ 表示单位 $i$ 的分配：$D_i=1$ 是处理组，$D_i=0$ 是对照组。[[潜在结果反事实|潜在结果]] $Y_i(1)$ 表示在处理状态下会出现的结果，$Y_i(0)$ 表示在对照状态下会出现的结果；括号里的 0、1 是状态标记，不是乘法、幂或时间。观测结果 $Y_i$ 满足切换关系：
$$Y_i=D_iY_i(1)+(1-D_i)Y_i(0).$$
代入 $D_i=1$ 得 $1\times Y_i(1)+0\times Y_i(0)=Y_i(1)$；代入 $D_i=0$ 得 $0\times Y_i(1)+1\times Y_i(0)=Y_i(0)$。因此同一单位在同一处理比较中只能观察其中一个潜在结果；另一个是未观察的反事实，不是“没有定义”或“不存在”。

<!-- bilingual-en:start -->
Assignment D takes values zero and one. Potential outcomes index the result under each treatment state; parentheses label states rather than powers, products, or time. Substituting one and zero in the switching equation selects the relevant potential outcome. Only one is observed for the same unit in this treatment comparison; the other is an unobserved counterfactual, not an undefined quantity.
<!-- bilingual-en:end -->

个体效应为 $\tau_i=Y_i(1)-Y_i(0)$。例如同一个人的两个潜在成绩为 70 与 62，效应为 $70-62=8$。实际只观察到 70 时，不能靠这一行直接算出 8，因为 62 未观察。这是[[反事实不可同时观察]]的问题。模型还要明确处理版本、测量时点以及是否存在单位之间的干扰；“分配辅导”与“实际参加辅导”也必须分别定义。如果 $D$ 表示分配，那么对应的是分配效应。

<!-- bilingual-en:start -->
An individual effect subtracts the same unit’s untreated outcome from its treated outcome. If those potential scores were 70 and 62, the effect would be eight; observing only 70 does not reveal it. This is the [[反事实不可同时观察|missing-counterfactual problem]]. Treatment versions, timing, and interference must be specified. Assignment to tutoring and actual attendance define different interventions; an assignment indicator targets an assignment effect.
<!-- bilingual-en:end -->

## 3. 从个体效应到 ATE 与 ATET（slides 18–19）

<!-- bilingual-en:start -->
*From individual effects to ATE and ATET*
<!-- bilingual-en:end -->

[[期望]] $E[X]$ 是按分布概率加权的平均。有限总体若各单位权重相等，便是“所有单位的值相加，再除以单位数”。[[条件期望]] $E[X\mid D=1]$ 中的竖线读作“在已知 $D=1$ 的条件下”；它先限定到处理组，再使用该组内的概率权重平均。竖线不是除号。若六个单位中前三个接受处理，$E[\tau\mid D=1]$ 在这个等权有限例子里只平均前三人的效应。

<!-- bilingual-en:start -->
Expectation is a probability-weighted average. An equally weighted finite population uses the arithmetic average of all units. A conditional expectation first restricts attention to the indicated group, then averages using the conditional probabilities. The conditioning bar is not division. In a six-unit example with the first three treated, the conditional mean effect averages those three effects.
<!-- bilingual-en:end -->

[[平均处理效应|ATE]] 面向全体目标单位，[[已处理者平均处理效应|ATET（也写 ATT）]] 面向已经进入处理组的单位：
$$\mathrm{ATE}=E[Y(1)-Y(0)],\qquad \mathrm{ATET}=E[Y(1)-Y(0)\mid D=1].$$
只要期望存在，[[期望线性性]]允许把“差的期望”拆成“期望之差”：$E[Y(1)]-E[Y(0)]$。这里不需要假设两种潜在结果独立。ATET 仍是同一批处理组成员的两种潜在结果之间的比较，不能把第二项换成另一批对照组的观测均值。

<!-- bilingual-en:start -->
ATE averages across the target population; ATET or ATT averages among treated units. Linearity separates each expectation of a difference into a difference of expectations without requiring independence. ATET compares two treatment states for the same treated group, so its untreated term cannot simply be replaced by observed controls.
<!-- bilingual-en:end -->

| 单位 | $Y(1)$ | $Y(0)$ | $D$ | 观测 $Y$ | $\tau$ |
|---|---:|---:|---:|---:|---:|
| A | 70 | 62 | 1 | 70 | 8 |
| B | 78 | 68 | 1 | 78 | 10 |
| C | 74 | 65 | 1 | 74 | 9 |
| D | 60 | 55 | 0 | 55 | 5 |
| E | 66 | 58 | 0 | 58 | 8 |
| F | 58 | 51 | 0 | 51 | 7 |

<!-- bilingual-en:start -->
*Columns show unit, treated and untreated potential outcomes, assignment, observed outcome, and individual effect. Both potential outcomes are supplied only to make this hypothetical calculation possible.*
<!-- bilingual-en:end -->

逐项相加，个体效应总和为 $8+10+9+5+8+7=47$，所以 $\mathrm{ATE}=47/6\approx7.833$。处理组只有 A、B、C，故 $\mathrm{ATET}=(8+10+9)/3=27/3=9$。两者不同，是因为平均的人群不同；不是哪个公式算错。若处理组比例为 $p$，全概率的分组思路还给出 $\mathrm{ATE}=p\,\mathrm{ATET}+(1-p)E[\tau\mid D=0]$。本表有 $p=1/2$，因此 $\mathrm{ATE}=(1/2)\times9+(1/2)\times(20/3)=47/6$。

<!-- bilingual-en:start -->
The six effects sum to 47, giving ATE 47/6. The treated effects sum to 27, giving ATET nine. The difference reflects different populations being averaged. Weighting the treated and untreated mean effects by their population proportions recovers the ATE, as the displayed arithmetic verifies.
<!-- bilingual-en:end -->

## 4. 可观测的均值差为何还不是因果效应（slides 19–23）

<!-- bilingual-en:start -->
*Why the observed mean contrast needs identification*
<!-- bilingual-en:end -->

把总体可观测均值差记为 $\beta=E[Y\mid D=1]-E[Y\mid D=0]$。这与含截距、仅有二元解释变量的总体均值表示相容：$E[Y\mid D]=\alpha+\beta D$。代入 $D=0$ 得 $\alpha=E[Y\mid D=0]$；代入 $D=1$ 得 $\alpha+\beta=E[Y\mid D=1]$；两式相减便得到 $\beta$。这是[[指示变量系数]]的总体解释。写 $Y=\alpha+\beta D+\varepsilon$ 并令 $E[\varepsilon\mid D]=0$，并不能额外证明因果关系；这个二组均值表示已经能够把组间选择装进 $\beta$。

<!-- bilingual-en:start -->
Define beta as the observable population mean contrast. With a binary regressor, the conditional means can be written using an intercept and a slope: substitute D equal to zero and one, then subtract. A zero conditional-mean residual in this representation does not establish causality: group selection can already be absorbed into beta. See [[指示变量系数|indicator coefficients]].
<!-- bilingual-en:end -->

[[均值差的因果分解]]的关键是加减同一个量 $E[Y(0)\mid D=1]$。加上它又减去它，相当于加 0，因此不改变原式：
$$
\begin{aligned}
\beta
&=E[Y(1)\mid D=1]-E[Y(0)\mid D=0]\\
&=E[Y(1)\mid D=1]\underbrace{-E[Y(0)\mid D=1]+E[Y(0)\mid D=1]}_{=0}-E[Y(0)\mid D=0]\\
&=\underbrace{E[Y(1)\mid D=1]-E[Y(0)\mid D=1]}_{\mathrm{ATET}}\\
&\quad+\underbrace{E[Y(0)\mid D=1]-E[Y(0)\mid D=0]}_{\mathrm{SB}}.
\end{aligned}
$$
第一括号让同一组人接受两种状态，所以是因果效应；第二括号让两组人都处在未处理状态，所以是[[选择偏差]]：没有处理时原本就存在的平均差。

<!-- bilingual-en:start -->
Add and subtract the treated group’s untreated mean, which adds zero. Regrouping yields a same-group treatment contrast, ATET, plus a between-group untreated contrast, selection bias. The first changes treatment state while holding group membership fixed; the second changes group membership while holding the untreated state fixed.
<!-- bilingual-en:end -->

上表的观测均值差为 $(70+78+74)/3-(55+58+51)/3=222/3-164/3=58/3$。处理组未处理均值为 $(62+68+65)/3=195/3=65$；对照组未处理均值为 $164/3$；故 $\mathrm{SB}=195/3-164/3=31/3$。检查：$\mathrm{ATET}+\mathrm{SB}=9+31/3=27/3+31/3=58/3=\beta$。如果目标是 ATE，还要再减去它：
$$\beta-\mathrm{ATE}=\mathrm{SB}+(\mathrm{ATET}-\mathrm{ATE}).$$
因而只说“相关不等于因果是因为 SB”遗漏了目标人群差异。SB 为 0 只直接推出 $\beta=\mathrm{ATET}$；要等于 ATE，还需要 ATET 与 ATE 相等。

<!-- bilingual-en:start -->
The observed contrast is 58/3, while the untreated baseline difference is 31/3. Adding ATET nine verifies the decomposition. Relative to ATE, the discrepancy also includes the ATET–ATE gap. Eliminating selection bias alone identifies ATET, and identifying ATE requires addressing the target-population difference as well.
<!-- bilingual-en:end -->

[[混淆因子]]解释选择机制可能怎样出现：家长参与度既影响补习报名，也影响成绩，产生“报名 ← 家长参与度 → 成绩”这条非因果路径。项目若优先接收较弱学生，选择偏差也可为负；于是有效的补习项目可能对应负的观测均值差。判断符号之前，先明确比较组如何形成。

<!-- bilingual-en:start -->
A [[混淆因子|confounder]] can produce selection: parental involvement affects both enrolment and scores. If a programme instead targets weaker students, selection bias can be negative, and a beneficial intervention may have a negative observed mean contrast. The group-formation mechanism comes before interpreting the sign.
<!-- bilingual-en:end -->

## 5. 随机分配如何完成识别（slides 24–28）

<!-- bilingual-en:start -->
*How random assignment identifies the target*
<!-- bilingual-en:end -->

[[随机分配识别]]使用的核心独立性是 $D\perp\!\!\!\perp(Y(1),Y(0))$，读作“分配与两种潜在结果独立”。还需两组有正概率、处理定义一致和适当的无干扰条件。独立意味着知道分组不会改变潜在结果的分布，故对 $d=0,1$ 有 $E[Y(0)\mid D=d]=E[Y(0)]$，于是 SB $=E[Y(0)]-E[Y(0)]=0$。同理：
$$
\begin{aligned}
\mathrm{ATET}
&=E[Y(1)\mid D=1]-E[Y(0)\mid D=1]\\
&=E[Y(1)]-E[Y(0)]\\
&=E[Y(1)-Y(0)]
=\mathrm{ATE}.
\end{aligned}
$$
两步合起来才是 $\beta=\mathrm{ATET}+0=\mathrm{ATE}$。这里的两个“consistency”要区分：潜在结果框架的一致性是观察结果对应实际状态，后文估计量一致性是样本量增长时的概率极限。

<!-- bilingual-en:start -->
Random assignment makes assignment independent of both potential outcomes. With positive group probabilities, consistency of observed and potential outcomes, and suitable interference restrictions, conditional potential-outcome means equal their marginal means. This first eliminates selection bias and then equates ATET with ATE. Potential-outcome consistency is a treatment-state assumption; estimator consistency later refers to a probability limit.
<!-- bilingual-en:end -->

识别完成后，有限样本采用 $\hat\beta=\bar Y_1-\bar Y_0$。帽子表示由数据构造的估计；下标 1、0 表示组别。$n_1=\sum_i\mathbf1\{D_i=1\}$、$n_0=\sum_i\mathbf1\{D_i=0\}$ 是两组人数，来自[[事件指示变量]]，不是两组均值，且 $n=n_1+n_0$。带条件的求和 $\sum_{i:D_i=1}$ 意味着只把处理组的观察相加：
$$\hat\beta=\frac1{n_1}\sum_{i:D_i=1}Y_i-\frac1{n_0}\sum_{i:D_i=0}Y_i.$$
处理组 $64,72,68$，对照组 $55,61,60$ 时，$n_1=n_0=3$，所以 $\bar Y_1=204/3=68$，$\bar Y_0=176/3$，差为 $(204-176)/3=28/3\approx9.33$。9.33 是本次估计值，不是已证明的真实 ATE。

<!-- bilingual-en:start -->
After identification, the sample mean difference estimates the contrast. The hat indicates an estimate constructed from data; subscripts identify the groups. Group sizes count observations using indicator variables, rather than denoting group means. In the displayed sample, the two sums are 204 and 176 and both counts are three, yielding 28/3 or about 9.33. This is the realised estimate, not a proven true ATE.
<!-- bilingual-en:end -->

为什么 OLS 也给出同一个差？[[二元回归的样本均值差]]把残差平方和拆为 $Q=\sum_{D_i=0}(Y_i-\alpha)^2+\sum_{D_i=1}(Y_i-\alpha-\beta)^2$。令 $m_0=\alpha,m_1=\alpha+\beta$，两组分别选择自己的预测值。对任一组，$\frac{d}{dm}\sum(Y_i-m)^2=\sum2(Y_i-m)(-1)=2nm-2\sum Y_i$；令其为 0 得 $m=\sum Y_i/n=\bar Y$。二阶导数 $2n>0$，确认是最小值。于是 $\hat\alpha=\bar Y_0$，$\hat\alpha+\hat\beta=\bar Y_1$，相减得到 $\hat\beta=\bar Y_1-\bar Y_0$。这一样本代数结果不会替我们消除选择偏差。

<!-- bilingual-en:start -->
OLS splits into two sums of squares after parameterising the fitted group means separately. Differentiating each sum produces twice the group size times the fitted mean minus twice the observed sum. Setting this to zero gives the sample mean; the positive second derivative confirms minimisation. Subtracting the two fitted means yields the slope. This algebra does not remove selection bias.
<!-- bilingual-en:end -->

## 6. 识别的范围与其他比较设计（slides 29–30）

<!-- bilingual-en:start -->
*Identification beyond the randomised example*
<!-- bilingual-en:end -->

[[识别与估计]]也适用于预测和结构模型：即使知道历史分布，未来规律是否延续仍需依据；即使知道选择行为，偏好参数能否被唯一反推仍取决于模型和数据变化。因果研究中的不同设计，都要说明为什么某个比较能够代表缺失的反事实。

<!-- bilingual-en:start -->
Identification also matters for prediction and structural models. Historical distributions need a justified link to future conditions, and choice data must distinguish candidate preferences. Causal designs each need an argument explaining why their comparison reveals the missing counterfactual.
<!-- bilingual-en:end -->

[[协变量匹配]]比较处理前可观测特征相近的人，因果解释依赖条件交换性与重叠等条件；[[断点回归设计]]比较阈值两侧，在连续性条件下识别阈值处的局部效应；[[工具变量有效条件|工具变量]]用能推动处理、并满足相关性、外生性与相应排除限制的变化；[[双重差分法]]比较两组的前后变化，其反事实论证依赖[[平行趋势]]等条件。本讲是设计入口，不把这些名称当作已经完成的识别证明。

<!-- bilingual-en:start -->
[[协变量匹配|Matching]] uses similar observed pretreatment characteristics, with conditional exchangeability and overlap among its assumptions. [[断点回归设计|RD]] uses a cutoff and continuity to identify a local effect. [[工具变量有效条件|IV]] uses relevant external variation with exogeneity and appropriate exclusion restrictions. [[双重差分法|DID]] compares changes across groups and relies on assumptions such as [[平行趋势|parallel trends]]. These are introductions to designs, not completed identification arguments.
<!-- bilingual-en:end -->

## 7. 统计量把样本压缩成可计算的信息（slides 31–34）

<!-- bilingual-en:start -->
*Statistics as computable sample summaries*
<!-- bilingual-en:end -->

[[统计量]]是样本的可测函数：$T_n=g(X_1,\ldots,X_n)$。$g$ 是计算规则，$n$ 是观察个数，大写 $X_i$ 表示尚未实现的随机观察，小写 $x_i$ 表示已经看到的数值。可测性确保“统计量落入某个范围”是可赋概率的事件。计算规则可以包含已知常数，不能需要未知参数才能执行；比如 $\bar X$ 是统计量，$\bar X-\mu$ 在 $\mu$ 未知时不是可直接计算的统计量。统计量是总类，作为某个目标的估计规则时才具有 estimator 的角色。

<!-- bilingual-en:start -->
A statistic is a measurable sample function: g is the calculation rule, n the number of observations, uppercase X a random observation, and lowercase x its realised value. Measurability ensures that the statistic falling in a range is a probabilistic event. The rule can use known constants but cannot require an unknown parameter. Estimator is the role a statistic takes when used for a specified target.
<!-- bilingual-en:end -->

位置看[[样本均值向量|样本均值]]、中位数与分位数；离散程度看[[方差|方差和标准差]]及[[四分位距]]；两变量共同变化看[[协方差]]与[[相关系数]]；不对称与尾部看[[偏度]]、[[峰度]]和极端分位数。令 $\bar x=n^{-1}\sum_i x_i$，常用样本协方差为 $s_{xy}=(n-1)^{-1}\sum_i(x_i-\bar x)(y_i-\bar y)$；当 $s_x,s_y>0$，相关系数为 $r=s_{xy}/(s_xs_y)$。它消除计量单位，但零相关不一般意味着独立。

<!-- bilingual-en:start -->
Location summaries include means, medians, and quantiles; spread uses variance, standard deviation, and IQR. Covariance and correlation describe joint variation, while skewness, kurtosis, and extreme quantiles capture other distributional features. Sample covariance averages products of centred observations with divisor n minus one; dividing by the two positive sample standard deviations yields correlation. Zero correlation need not mean independence.
<!-- bilingual-en:end -->

均值的求和号展开为 $\bar x=(x_1+x_2+\cdots+x_n)/n$。样本 $2,4,6$ 有 $\bar x=12/3=4$；逐项减去均值得 $-2,0,2$；平方得 $4,0,4$；相加得 8。于是样本方差 $s^2=8/(3-1)=4$，样本标准差 $s=\sqrt4=2$。方差的单位是原单位的平方，开平方才回到原单位。这里算出的 $s^2$ 是一个样本统计量，与总体参数 $\sigma^2=\operatorname{Var}(X)$ 需要区分。

<!-- bilingual-en:start -->
For the sample 2, 4, 6, the mean is four. Centred deviations are minus two, zero, and two; their squared sum is eight. Dividing by n minus one yields variance four and standard deviation two. Variance has squared units, whereas standard deviation returns to the original units. The sample statistic is distinct from the population variance parameter.
<!-- bilingual-en:end -->

再给协方差与相关一个完整小例子：保持 $x=(2,4,6)$，取 $y=(1,3,2)$。$\bar y=(1+3+2)/3=2$，所以 y 的中心偏离为 $-1,1,0$。逐项乘以 x 的中心偏离 $-2,0,2$，得到 $2,0,0$；相加为 2；除以 $n-1=2$ 得 $s_{xy}=1$。y 的样本方差为 $(1+1+0)/2=1$，故 $s_y=1$；已有 $s_x=2$，于是 $r=s_{xy}/(s_xs_y)=1/(2\times1)=0.5$。

<!-- bilingual-en:start -->
For a complete covariance example, pair x = (2, 4, 6) with y = (1, 3, 2). The y mean is two and its centred deviations are minus one, one, and zero. Multiplying the paired deviations gives two, zero, zero. Their sum divided by n minus one yields sample covariance one. The y sample standard deviation is one and the x standard deviation two, giving correlation one half.
<!-- bilingual-en:end -->

### 7.1 为什么方差分母是 n−1

<!-- bilingual-en:start -->
*Why sample variance divides by n minus one*
<!-- bilingual-en:end -->

这是已有[[样本协方差无偏分母]]的一维情况，也直接连接[[01_Math/04_多元统计分析/03_样本几何与随机抽样Sample Geometry and Random Sampling#1.2.3. 样本协方差矩阵的期望|多元统计课程的同一推导]]。设 $X_i$ iid、均值 $\mu$、有限方差 $\sigma^2$，且 $n\ge2$。先注意 $X_i-\bar X=(X_i-\mu)- (\bar X-\mu)$。对每项平方，再求和：
$$
\begin{aligned}
\sum_i(X_i-\bar X)^2
&=\sum_i(X_i-\mu)^2-2(\bar X-\mu)\sum_i(X_i-\mu)+n(\bar X-\mu)^2\\
&=\sum_i(X_i-\mu)^2-2n(\bar X-\mu)^2+n(\bar X-\mu)^2\\
&=\sum_i(X_i-\mu)^2-n(\bar X-\mu)^2.
\end{aligned}
$$
第二行用了 $\sum_i(X_i-\mu)=\sum_iX_i-n\mu=n\bar X-n\mu=n(\bar X-\mu)$。

<!-- bilingual-en:start -->
The scalar calculation is the one-dimensional case of [[样本协方差无偏分母|the unbiased covariance divisor]], also derived in the multivariate course. Expand each centred square around the population mean, then sum. The sum of deviations from that mean is n times the sample mean’s deviation, which simplifies the cross term as shown.
<!-- bilingual-en:end -->

由[[样本均值无偏性]]，$E[\bar X]=\mu$；由[[样本均值协方差]]的一维 iid 特例，$\operatorname{Var}(\bar X)=\sigma^2/n$。因此上一式取期望得到 $n\sigma^2-n(\sigma^2/n)=n\sigma^2-\sigma^2=(n-1)\sigma^2$。再除以 $n-1$，便有 $E[s^2]=\sigma^2$。若除以 $n$，期望变为 $((n-1)/n)\sigma^2$。这个结论需要指定抽样条件；不是任意相关数据都能只凭分母自动无偏。

<!-- bilingual-en:start -->
The iid sample mean is unbiased and has variance sigma squared over n. Taking expectations gives n sigma squared minus sigma squared, or n minus one times sigma squared. Dividing by n minus one removes this bias; division by n retains the shrinkage factor. The conclusion depends on the sampling conditions and is not automatic for dependent data.
<!-- bilingual-en:end -->

### 7.2 排序、分位数与 IQR

<!-- bilingual-en:start -->
*Ordering, quantiles, and IQR*
<!-- bilingual-en:end -->

[[顺序统计量]]用带括号的下标记录排序位置：$X_{(1)}\le\cdots\le X_{(n)}$。样本原顺序为 $12,4,19,7,22,9,15,11$，排序后为 $4,7,9,11,12,15,19,22$。$X_1=12$ 是原来的第一个观察，$X_{(1)}=4$ 是最小值。本课采用[[样本分位数]]约定 $\hat q_p=X_{(\lceil np\rceil)}$，其中 $0<p\le1$，$\lceil a\rceil$ 表示不小于 $a$ 的最小整数：例如 $\lceil2.1\rceil=3$，$\lceil2\rceil=2$。它取的是排序后的观察值，不是秩本身。

<!-- bilingual-en:start -->
Parenthesised indices denote rank after sorting. In the displayed sample, the first recorded observation is 12 but the minimum is four. The lecture’s sample quantile uses rank ceil(np), where the ceiling rounds upward, retaining an integer unchanged. The quantile is the observed value at that rank, not the rank itself.
<!-- bilingual-en:end -->

这里 $n=8$：$p=0.25$ 给出 $np=2$、秩 2、值 7；$p=0.50$ 给出 $np=4$、秩 4、值 11；$p=0.75$ 给出 $np=6$、秩 6、值 15。因此第一、第三四分位数是 7 与 15，$\mathrm{IQR}=15-7=8$。若取 $p=0.30$，$np=2.4$、向上取整为 3、分位数是 9。偶数样本的中位数也常被定义为中间两值平均，本表会得到 $(11+12)/2=11.5$；那是另一种明确的约定。

<!-- bilingual-en:start -->
For n equal to eight, the quartile ranks are two and six, yielding seven and fifteen and an IQR of eight. The median under this convention is the fourth value, eleven. At p equal to 0.30, rank 2.4 rounds up to three, yielding nine. Averaging the two central values would give median 11.5 under another common convention.
<!-- bilingual-en:end -->

形状统计量也要说清口径。令 $m_k=n^{-1}\sum_i(x_i-\bar x)^k$；一种样本[[偏度]]为 $m_3/m_2^{3/2}$，一种[[峰度]]为 $m_4/m_2^2$（要求分母非零）。三次方保留方向，四次方放大两侧的大偏离。总体版本还要求对应阶数的矩存在；峰度与超额峰度相差 3，软件的有限样本调整也可能不同。它们提供各自的信息，不能代替整条分布。

<!-- bilingual-en:start -->
Specify conventions for shape summaries as well. Empirical central moments yield one version of sample skewness and kurtosis, provided the denominator is positive. Odd powers retain sign; fourth powers heavily weight extreme deviations on both sides. Population versions need the relevant finite moments. Kurtosis and excess kurtosis differ by three, and software adjustments can differ.
<!-- bilingual-en:end -->

仍用 $x=(2,4,6)$ 演示矩的每一步：$m_2=[(-2)^2+0^2+2^2]/3=8/3$；$m_3=[(-2)^3+0^3+2^3]/3=(-8+8)/3=0$，故样本偏度为 0。$m_4=[(-2)^4+0^4+2^4]/3=(16+16)/3=32/3$；峰度为 $(32/3)/(8/3)^2=(32/3)/(64/9)=(32/3)\times(9/64)=3/2$，超额峰度为 $3/2-3=-3/2$。这里全部 $m_k$ 都用分母 n，不能把前面 n−1 版本的样本方差未经调整直接代进来。

<!-- bilingual-en:start -->
For the same x sample, the second empirical central moment is 8/3, the third zero, and the fourth 32/3. These give skewness zero, kurtosis 3/2, and excess kurtosis minus 3/2. All empirical moments here use divisor n; substituting the earlier variance with divisor n minus one would change the convention.
<!-- bilingual-en:end -->

## 8. 概率空间：先分清三个层级（slides 35–37）

<!-- bilingual-en:start -->
*The three levels of a probability space*
<!-- bilingual-en:end -->

[[概率空间]]写作 $(\Omega,\mathcal F,P)$；slides 有时用 $\mathcal A$ 表示中间那一项，与这里的 $\mathcal F$ 是同一角色。一次试验的单个结果写为 $\omega$；所有可能结果组成 $\Omega$；事件 $A$ 是 $\Omega$ 的子集；$\mathcal F$ 则是“事件的集合”，所以 $\omega\in\Omega$、$A\subseteq\Omega$、$A\in\mathcal F$。$\in$ 读“是成员”，$\subseteq$ 读“是子集”。最后 $P$ 是把事件映射到 $[0,1]$ 内数值的函数。

<!-- bilingual-en:start -->
A probability space consists of outcomes, an event family, and a probability measure. The slides may write the event family as script A; this note uses script F. A single outcome is a member of the sample space, an event is a subset of that space, and the event is a member of the event family. Membership and subset inclusion are different relationships. The probability function maps events to numbers between zero and one.
<!-- bilingual-en:end -->

以一次掷硬币为例，$\Omega=\{H,T\}$，其中 $H$ 与 $T$ 分别代表正面、反面。$\{H\}$ 是“结果为正面”这个事件；$\{H,T\}=\Omega$ 是必然事件；$\varnothing$ 是没有任何结果的不可能事件。若允许区分正反面，事件族是 $\mathcal F=\{\varnothing,\{H\},\{T\},\{H,T\}\}$。外层花括号包含四个事件，内层花括号包含每个事件的结果。公平性用 $P(\{H\})=P(\{T\})=1/2$ 表示；公平性属于赋概率的模型假设，不来自符号 H、T。

<!-- bilingual-en:start -->
For one coin toss, the sample space has two outcomes. The heads event contains one outcome; the whole space is certain and the empty event impossible. The full event family contains four subsets, not four outcomes. Fairness assigns equal probabilities to the singleton events and is a model assumption, rather than a property of their names.
<!-- bilingual-en:end -->

### 8.1 σ-代数为什么要有封闭性

<!-- bilingual-en:start -->
*Why a sigma-algebra needs closure*
<!-- bilingual-en:end -->

[[σ-代数]]是符合三项条件的事件族。第一，$\Omega\in\mathcal F$，允许询问“某个可能结果发生了吗”。第二，若 $A\in\mathcal F$，则补集 $A^c=\Omega\setminus A\in\mathcal F$：能问 A 是否发生，也能问 A 是否不发生。第三，对 $A_1,A_2,\ldots\in\mathcal F$，可数并 $\bigcup_{j=1}^{\infty}A_j\in\mathcal F$：能把一列允许的事件用“至少一个发生”组合起来。$\cup$ 表示“或”，$\cap$ 表示“同时”，这里的“或”允许多个同时发生。

<!-- bilingual-en:start -->
A sigma-algebra contains the whole space, closes under complements, and closes under countable unions. These conditions let allowed questions remain allowed when negated or combined into “at least one occurs.” Union means inclusive or; intersection means both. Countable refers to a collection that can be listed in a sequence.
<!-- bilingual-en:end -->

为什么还允许“同时发生”？由 De Morgan 关系 $A\cap B=(A^c\cup B^c)^c$：先取两次补集，再取并，最后取补集，每一步都没有离开 $\mathcal F$，所以交集也在里面。同样，$\varnothing=\Omega^c\in\mathcal F$。有限并也是可数并的特例：把后续各项都取空集即可。σ 强调的是允许可数无穷组合，不能把它改成“任意不可数并都封闭”。

<!-- bilingual-en:start -->
Intersections are allowed because an intersection equals the complement of the union of complements. Every step stays within the event family. The empty event is the complement of the whole space. Finite unions are covered by filling the remaining sequence with empty events. Closure under countable unions is not a claim about every uncountable union.
<!-- bilingual-en:end -->

检验候选族 $\{\varnothing,\{H\},\Omega\}$：它有全集，但 $\{H\}$ 的补集 $\{T\}$ 不在里面，所以不是 σ-代数。另一方面，只有 $\{\varnothing,\Omega\}$ 也合法，只是它无法区分正面和反面。骰子只报告奇偶时，可用 $\{\varnothing,\{1,3,5\},\{2,4,6\},\Omega\}$；“奇数”是事件，“恰好 1”不属于这一级信息下的事件族。由此，σ-代数既处理赋概率的合法性，也能描述所掌握的信息精细程度，后来通过[[滤过定义]]接入随时间增长的信息。

<!-- bilingual-en:start -->
The proposed coin event family missing tails fails closure under complements. The family containing only the empty set and whole space is valid but cannot distinguish heads from tails. A die’s parity-only family distinguishes odd from even but not a singleton outcome. Sigma-algebras therefore also represent information resolution, leading to [[滤过定义|filtrations]] when information grows over time.
<!-- bilingual-en:end -->

有限样本空间通常可以把所有子集都当作事件。实数这样的连续空间则通常选合适的事件族，使区间、极限等常用操作与概率测度相容；Lecture 1 不需要构造不可测集合。这里的实际目标是能够分清“可能发生什么、允许问哪些事件、如何给事件赋概率”，而不是只记住三个符号。

<!-- bilingual-en:start -->
Finite spaces commonly use every subset as an event. Continuous spaces usually use an appropriate event family compatible with intervals, limiting operations, and probability measures. Lecture 1 does not require constructing nonmeasurable sets. The immediate aim is to distinguish possible outcomes, admissible events, and assigned probabilities.
<!-- bilingual-en:end -->

### 8.2 三条公理怎样变成可用的运算

<!-- bilingual-en:start -->
*From probability axioms to calculations*
<!-- bilingual-en:end -->

[[概率三公理]]是：每个事件 $P(A)\ge0$；全集 $P(\Omega)=1$；若事件两两互斥，即 $A_j\cap A_k=\varnothing$ 对 $j\ne k$ 成立，则 $P(\bigcup_jA_j)=\sum_jP(A_j)$。注意非负是 $\ge0$，允许概率为零；第三条允许可数无穷个事件，却要求两两互斥，不能只凭“不同事件”就相加。

<!-- bilingual-en:start -->
The axioms require nonnegative probabilities, probability one for the whole space, and additivity across pairwise disjoint events, including countably many. Probability zero is allowed. Distinct events need not be disjoint, so distinctness alone does not permit addition.
<!-- bilingual-en:end -->

由 $\Omega=\Omega\cup\varnothing$ 的互斥分解得 $1=1+P(\varnothing)$，所以空事件概率为 0。由 $\Omega=A\cup A^c$ 得 $1=P(A)+P(A^c)$，移项便是[[补事件法]] $P(A^c)=1-P(A)$。一般两个事件可拆成只在 A、只在 B、同时在两者中三块；直接加 $P(A)+P(B)$ 会把交集算两次，故[[互斥事件加法|一般加法公式]]为 $P(A\cup B)=P(A)+P(B)-P(A\cap B)$。

<!-- bilingual-en:start -->
Disjoint decompositions yield probability zero for the empty event and the complement rule. For overlapping events, adding their probabilities counts the intersection twice, so subtract it once. These are consequences of the axioms rather than separate assumptions.
<!-- bilingual-en:end -->

## 9. 概率解释与 Bayes 更新（slides 38–39）

<!-- bilingual-en:start -->
*Probability interpretations and Bayesian updating*
<!-- bilingual-en:end -->

[[频率派与贝叶斯推断]]中，频率解释把概率联系到重复试验的长期比例；贝叶斯方法用概率表达在模型和信息下对未知量的不确定性。课程主要研究前一种重复抽样评价。不能因此说贝叶斯方法的数据不随机，或贝叶斯估计量不能谈偏差与一致性；后验均值等规则同样能放回重复抽样机制下评价。

<!-- bilingual-en:start -->
Frequentist interpretation connects probability to long-run frequencies; Bayesian inference uses probability to represent uncertainty under a model and information. The course mainly studies repeated-sampling assessment. Bayesian models can also have random data, and posterior-based estimators can be assessed for frequentist bias or consistency.
<!-- bilingual-en:end -->

先从[[条件概率]]写起。若 $P(B)>0$，则 $P(A\mid B)=P(A\cap B)/P(B)$：只保留 B 发生的情形，把其中 A 也发生的部分除以 B 的总概率，重新归一化。同一个交集还满足 $P(A\cap B)=P(B\mid A)P(A)$。代回去，得到[[Bayes法则]]：
$$P(A\mid B)=\frac{P(B\mid A)P(A)}{P(B)}.$$
左右条件方向发生反转，因此必须保留分母。$P(B\mid A)$ 问“A 成立时，B 有多常见”；$P(A\mid B)$ 问“B 已发生，现在 A 有多可信”，不是同一个问题。

<!-- bilingual-en:start -->
Conditional probability restricts attention to event B and renormalises. Expressing the same intersection using the opposite conditioning direction yields Bayes’ rule. The probability of evidence under a hypothesis and the probability of the hypothesis after evidence answer different questions, so the denominator cannot be discarded.
<!-- bilingual-en:end -->

### 9.1 三个对象分别是什么

<!-- bilingual-en:start -->
*Prior, likelihood, and posterior*
<!-- bilingual-en:end -->

[[先验分布]] $\pi(\theta)$：纳入这次数据前，未知参数可能取哪些值、各有多大权重。[[似然函数]] $L(\theta;x)=f_\theta(x)$：固定已经看到的数据 $x$，逐个考察候选参数 $\theta$ 在模型下产生这份数据的概率或密度。[[后验分布]] $\pi(\theta\mid x)$：综合先验与这次数据之后，未知量的条件分布。似然的自变量是候选参数，但它并不因此就是参数的概率分布。

<!-- bilingual-en:start -->
The prior assigns weights to parameter values before incorporating the current data. The likelihood fixes the observed data and evaluates its probability or density under candidate parameters. The posterior is the conditional parameter distribution after combining prior and evidence. Having the parameter as the likelihood’s argument does not make it a parameter probability distribution.
<!-- bilingual-en:end -->

一个完整例子：两种袋子 A、B，A 中红球比例为 0.9，B 中为 0.1。先随机选一个袋子，先验 $P(A)=P(B)=0.5$。从选中的同一袋子有放回、独立抽三次，观察到数据 $x=(红,红,红)$。给定 A，三次独立意味着概率相乘：$L(A;x)=0.9\times0.9\times0.9=0.729$；给定 B，同理 $L(B;x)=0.1^3=0.001$。0.729 的意思是“A 为真时，这串红球的概率”，不是“看到红球后，A 为真的概率”。

<!-- bilingual-en:start -->
Choose between two bags with equal prior probability. Bag A has red proportion 0.9 and B proportion 0.1. Three independent draws with replacement from the chosen bag are all red. Multiplication gives likelihoods 0.729 and 0.001. The first number describes this evidence under bag A, not the probability of A after the evidence.
<!-- bilingual-en:end -->

先乘先验，得到联合权重：$P(A,x)=0.5\times0.729=0.3645$，$P(B,x)=0.5\times0.001=0.0005$。A、B 是完备互斥可能性，用[[全概率公式]]得到 $P(x)=0.3645+0.0005=0.365$。最后分别除以同一个总量：
$$P(A\mid x)=\frac{0.3645}{0.365}=\frac{729}{730}\approx0.998630,$$
$$P(B\mid x)=\frac{0.0005}{0.365}=\frac1{730}\approx0.001370.$$
检查两者相加为 1。分母不是一种额外证据，而是所有候选情形产生当前数据的总概率。

<!-- bilingual-en:start -->
Multiply each likelihood by its prior to obtain joint weights 0.3645 and 0.0005. Summing over the exhaustive, disjoint hypotheses gives evidence probability 0.365. Dividing both weights by it yields posterior probabilities 729/730 and 1/730, which sum to one. The denominator aggregates the same evidence across hypotheses.
<!-- bilingual-en:end -->

保持数据和似然不变，改用先验 $P(A)=0.01$、$P(B)=0.99$：A 的权重为 $0.01\times0.729=0.00729$；B 的权重为 $0.99\times0.001=0.00099$；总和 $0.00828$；A 的后验为 $0.00729/0.00828=81/92\approx0.880435$。同样的 0.729 似然，两种先验得到不同后验，这正是二者不应混淆的直接检验。若把两个似然直接除以它们的和，等价于在这个有限模型里使用相等先验，不能省略这个前提。

<!-- bilingual-en:start -->
Keeping the evidence fixed but changing the priors to 0.01 and 0.99 yields joint weights 0.00729 and 0.00099. The posterior for A becomes 81/92, about 0.880435. Thus the same likelihood produces different posteriors. Normalising likelihoods alone corresponds to equal priors in this finite model and requires that assumption.
<!-- bilingual-en:end -->

### 9.2 一般公式与连续数据的区别

<!-- bilingual-en:start -->
*General notation and continuous data*
<!-- bilingual-en:end -->

对离散候选参数 $\theta_1,\ldots,\theta_k$，后验为
$$\pi(\theta_j\mid x)=\frac{L(\theta_j;x)\pi(\theta_j)}{\sum_{\ell=1}^kL(\theta_\ell;x)\pi(\theta_\ell)}.$$
$j$ 表示现在关心的候选，$\ell$ 是分母遍历所有候选的哑下标。若参数连续，则分母写成 $m(x)=\int_\Theta L(t;x)\pi(t)\,dt$，$t$ 是积分时变化的参数；$x$ 始终固定。$\propto$ 表示在固定数据后，只差一个与参数无关的正归一化常数，即 posterior $\propto$ likelihood $\times$ prior。

<!-- bilingual-en:start -->
For discrete candidates, the numerator selects one hypothesis while the denominator sums over all candidates using a separate dummy index. For continuous parameters, integration replaces summation. The data remain fixed throughout. Proportionality means that the missing normalising factor does not depend on the parameter.
<!-- bilingual-en:end -->

[[概率密度函数|连续密度]]的值不能直接读成“恰好这个观测值的概率”；连续变量的单点概率通常是 0，但密度可以为正甚至大于 1。Bayes 公式用相容的概率或密度建立联合分布，再归一化。课堂中“感冒直觉由 10% 更新到 60%”可以说明信念变化，但没有条件概率与替代解释就无法核算这两个数。一个纯教学的检测例子则可以核算：先验 0.01，真阳性率 0.9，假阳性率 0.1，阳性后的概率为 $0.009/(0.009+0.099)=1/12\approx0.08333$，见[[灵敏度与后验概率]]。这些数字是模型设定。

<!-- bilingual-en:start -->
A continuous density value is not the probability of observing exactly that value; point probabilities are usually zero, while densities can be positive or exceed one. Bayesian calculations use compatible masses or densities and normalisation. A verbal change from 10% to 60% illustrates belief updating but cannot be verified without a model. A hypothetical detection model with prior 0.01, sensitivity 0.9, and false-positive rate 0.1 instead gives posterior 1/12, illustrating [[灵敏度与后验概率|sensitivity versus posterior probability]].
<!-- bilingual-en:end -->

## 10. 估计量与估计值：公式为什么具有分布（slides 40–48）

<!-- bilingual-en:start -->
*Why an estimation rule has a distribution*
<!-- bilingual-en:end -->

[[估计对象三分]]区分目标 estimand、规则 estimator、数值 estimate。目标可以是总体均值差 $\beta$；规则是 $\hat\beta=\bar Y_1-\bar Y_0$；某次数据算出的数值是 $28/3$。数据实现前，规则的输入会随样本变化，所以输出是[[随机变量]]；数据实现后，输出就是一个固定数。课程可能用同一符号 $\hat\beta$ 表示两者，阅读时要判断是在讨论规则还是其一次实现。

<!-- bilingual-en:start -->
The estimand is the target, the estimator is the rule, and the estimate is its realised output. Before observations are realised, varying random inputs induce a random output; afterwards one obtains a fixed number. Courses may use the same hat notation for both, so determine which role the statement concerns.
<!-- bilingual-en:end -->

[[抽样分布]]回答：在固定的[[数据生成过程|DGP]]和抽样或分配机制下，如果重新产生数据、再使用同一规则，估计量有哪些可能值，各以多大概率出现？它不是原始观察值的分布，也不是把同一份数据反复按计算器得到的输出。为使问题确定，需要固定真实参数、每次样本量、数据依赖关系以及统计规则。

<!-- bilingual-en:start -->
A sampling distribution describes the possible outputs and their probabilities when the same rule is applied to newly generated data under a fixed DGP and sampling or assignment mechanism. It differs from the raw observations’ distribution and from recalculating unchanged data. Parameters, sample size, dependence, and the rule must be specified.
<!-- bilingual-en:end -->

### 10.1 完整枚举一个随机分配分布

<!-- bilingual-en:start -->
*Enumerating a randomisation distribution*
<!-- bilingual-en:end -->

取固定的四人有限总体，潜在结果为 A $(80,70)$、B $(76,64)$、C $(60,52)$、D $(66,54)$，括号依次是 $Y(1),Y(0)$。个体效应为 $10,12,8,12$，目标有限总体 ATE 为 $42/4=10.5$。随机等概率选两人处理，共有六种分配。每次都计算两组观测均值差；数据随机性来自重新分配，而不是把固定潜在结果改写。

<!-- bilingual-en:start -->
Consider four fixed units with the displayed treated and untreated potential outcomes. Their effects sum to 42, giving finite-population ATE 10.5. Assign two of four to treatment uniformly, producing six possible allocations. Randomness comes from assignment, while potential outcomes remain fixed.
<!-- bilingual-en:end -->

| 处理成员 | 处理组观测均值 | 对照组观测均值 | $\hat\beta(D)$ | 该次分配下的未处理基线差 |
|---|---:|---:|---:|---:|
| AB | $(80+76)/2=78$ | $(52+54)/2=53$ | 25 | 14 |
| AC | $(80+60)/2=70$ | $(64+54)/2=59$ | 11 | 2 |
| AD | $(80+66)/2=73$ | $(64+52)/2=58$ | 15 | 4 |
| BC | $(76+60)/2=68$ | $(70+54)/2=62$ | 6 | −4 |
| BD | $(76+66)/2=71$ | $(70+52)/2=61$ | 10 | −2 |
| CD | $(60+66)/2=63$ | $(70+64)/2=67$ | −4 | −14 |

<!-- bilingual-en:start -->
*Each row has probability 1/6. Columns identify the treated pair, observed treated and control means, realised estimator, and that allocation’s difference in untreated baseline means.*
<!-- bilingual-en:end -->

例如 AB 的未处理基线差是 $(70+64)/2-(52+54)/2=67-53=14$。这是该次分配的有限总体基线不平衡，不能直接记作总体选择偏差参数 SB。六种分配平均后，基线差为 $(14+2+4-4-2-14)/6=0$；估计量的分配期望为 $(25+11+15+6+10-4)/6=63/6=10.5$。随机分配保证适当的期望关系，不保证每一次分组精确平衡。这个例子同时连接[[随机分配识别]]、[[抽样分布]]和[[估计量无偏性]]，但没有给出样本量增长的序列，因此不能单靠它判断一致性。

<!-- bilingual-en:start -->
For AB, the baseline difference is 14. This is realised finite-population imbalance, not the population selection-bias parameter. Averaging over all six assignments eliminates the imbalance and gives estimator expectation 10.5. Randomisation supplies expectation relationships, not exact balance in every allocation. This example connects identification, sampling distributions, and unbiasedness, but does not specify a growing-sample sequence for assessing consistency.
<!-- bilingual-en:end -->

### 10.2 R 和 n：两条不同的轴

<!-- bilingual-en:start -->
*R and n are different dimensions*
<!-- bilingual-en:end -->

[[样本量与模拟重复次数]]要按两层读。外层编号 $r=1,\ldots,R$ 是第几次完整重复；内层编号 $i=1,\ldots,n$ 是这一次样本的第几个观察。可以写成 $X_i^{(r)}$：上标括号是重复编号，不是幂。每一行的 $n$ 个观察通过同一个函数产生一个估计值 $T_n^{(r)}$：
$$
\begin{array}{cccccc}
r=1:&X_1^{(1)},&X_2^{(1)},&\cdots,&X_n^{(1)}&\longrightarrow T_n^{(1)}\\
r=2:&X_1^{(2)},&X_2^{(2)},&\cdots,&X_n^{(2)}&\longrightarrow T_n^{(2)}\\
\vdots&&&&&\vdots\\
r=R:&X_1^{(R)},&X_2^{(R)},&\cdots,&X_n^{(R)}&\longrightarrow T_n^{(R)}.
\end{array}
$$
画抽样分布直方图时，用右边的 $R$ 个估计值；画原始数据直方图时，用某一行的 $n$ 个观察。两张图的横轴对象不同。

<!-- bilingual-en:start -->
Read the simulation as two indices: r labels a complete repetition, and i labels observations within that sample. Parenthesised superscripts identify repetitions rather than powers. Each row produces one estimate. The sampling-distribution histogram uses the R outputs on the right; a raw-data histogram uses the n observations in one row. Their horizontal axes refer to different objects.
<!-- bilingual-en:end -->

固定 $n=10$，从 $R=100$ 增到 $R=10000$：每个估计量仍只使用十个观察，但更容易看清其抽样分布的形状。固定足够大的 $R$，把 $n$ 从 10 增到 100：每次估计使用更多观察，若规则能够利用新增信息，抽样分布可能缩窄。对 $X_i$ iid、方差为 $\sigma^2$ 的样本均值，[[样本均值协方差|方差公式]]逐步为
$$\operatorname{Var}(\bar X_n)=\operatorname{Var}\!\left(\frac1n\sum_iX_i\right)=\frac1{n^2}\operatorname{Var}\!\left(\sum_iX_i\right)=\frac1{n^2}\sum_i\sigma^2=\frac{n\sigma^2}{n^2}=\frac{\sigma^2}{n}.$$
提出常数时方差乘常数的平方；独立性使协方差交叉项为零；求和中有 $n$ 个 $\sigma^2$。标准差是 $\sigma/\sqrt n$，所以 $n$ 增为四倍，标准差减为一半。公式中没有 $R$。

<!-- bilingual-en:start -->
At fixed n equal to ten, more repetitions clarify the same sampling distribution. Increasing n instead changes the data per estimate and may narrow its distribution. For the iid mean, extracting the constant contributes a squared factor, independence removes cross-covariances, and n equal variance terms remain. Its variance is sigma squared over n and standard deviation sigma over the square root of n. Quadrupling n halves this standard deviation. R does not enter the formula.
<!-- bilingual-en:end -->

“做更多重复”若只是模拟，不会让实际拿到的那次十人样本变成一万人样本。如果另把 $R$ 个模拟估计值求平均，这已定义了一个新的[[Monte Carlo估计]]，它的 Monte Carlo 标准误会随重复数变化；不能把新统计量的误差与原来单次估计量的方差混在一起。也不是任意估计规则都随着 $n$ 增加而改善：始终只使用 $X_1$ 就是后文的反例。

<!-- bilingual-en:start -->
Simulation repetitions do not enlarge the actual observed sample. Averaging simulated estimates defines a new Monte Carlo statistic with its own simulation error. Do not confuse its precision with the original estimator’s sampling variance. Nor does every rule improve with n: a rule using only the first observation supplies a counterexample.
<!-- bilingual-en:end -->

这六页固定每次样本量，依次增加重复数：$R=1,20,100,500,5000$，最后画出很大 R 下的平滑示意。随着 R 增加，直方图箱宽从 1 缩到 0.5、0.25、0.125。横轴是估计值 $\hat\beta$；纵轴 density 是密度，柱子的“高度 × 宽度”才代表该箱内的频率，所有柱面积合计为 1。此处选择连续抽样分布演示；离散统计量应使用概率质量而非强行画平滑 PDF。

<!-- bilingual-en:start -->
The six slides hold sample size fixed and increase repetitions through 1, 20, 100, 500, and 5000, ending with a smooth large-R illustration. Bin widths decrease from one to 0.5, 0.25, and 0.125. The horizontal axis contains estimates; density height times bin width gives relative frequency, and total area is one. This example uses a continuous sampling distribution; discrete statistics instead have probability masses.
<!-- bilingual-en:end -->

### Slides 42–47：原始抽样分布组图

<!-- bilingual-en:start -->
*Original sequence of sampling-distribution figures*
<!-- bilingual-en:end -->

![[Lecture 1 - Foundations.pdf#page=42]]

![[Lecture 1 - Foundations.pdf#page=43]]

![[Lecture 1 - Foundations.pdf#page=44]]

![[Lecture 1 - Foundations.pdf#page=45]]

![[Lecture 1 - Foundations.pdf#page=46]]

![[Lecture 1 - Foundations.pdf#page=47]]

读这组图时，每看到一个估计点都问它来自哪一整份样本；看到估计值的堆积，再问是重复次数增加让同一分布更清楚，还是每次样本量改变让分布本身改变。图展示一个分布的直觉，而概率模型定义这个分布，即使没有真的进行无限次试验，它也已经有明确含义。

<!-- bilingual-en:start -->
For each estimate in the figure sequence, identify the complete sample producing it. Then distinguish more repetitions revealing the same distribution from a change in sample size changing the distribution itself. Simulation illustrates a distribution already defined by the probability model; infinitely many actual experiments are unnecessary.
<!-- bilingual-en:end -->

## 11. 无偏、有效、一致分别检查什么（slides 49–65）

<!-- bilingual-en:start -->
*What unbiasedness, efficiency, and consistency assess*
<!-- bilingual-en:end -->

[[估计量偏差]]为 $\operatorname{Bias}_\theta(T_n)=E_\theta[T_n]-\theta$。下标 $\theta$ 表示在真实参数固定为该值的模型下取期望。[[估计量无偏性]]要求对模型内每个参数值，$E_\theta[T_n]=\theta$，而不是只在某次数据中命中目标。若目标是 $g(\theta)$，右侧就要换成它。期望是对估计量的抽样分布平均，不能把它换成对一份样本中原始数据的平均而不说明对象。

<!-- bilingual-en:start -->
Bias is the estimator’s sampling expectation minus its target. The parameter subscript identifies the model under which expectations are taken. Unbiasedness requires equality at every parameter value, not a lucky estimate in one sample. If the target is a parameter function, use that function. The expectation averages the sampling distribution of the estimator.
<!-- bilingual-en:end -->

设真实目标为 20，已给定完整抽样分布：$T$ 等概率取 $14,18,22,26$。则 $E[T]=(14+18+22+26)/4=80/4=20$，偏差为 0。中心偏离是 $-6,-2,2,6$；平方后是 $36,4,4,36$；方差 $=(36+4+4+36)/4=80/4=20$。这里除以 4 是对已知四点概率分布取期望，不是在估计未知总体方差，因此不使用 $4-1$。若只知道本次估计值为 14，则没有足够信息判断整个规则是否无偏。

<!-- bilingual-en:start -->
The specified four-point sampling distribution has expectation 20 and variance 20. Dividing by four averages its known equally weighted outcomes; it is not estimating a variance from a four-observation sample and does not use n minus one. A single estimate of 14 would not be enough to assess the rule’s unbiasedness.
<!-- bilingual-en:end -->

[[估计量有效性]]比较同一目标、同一样本量和同一模型下某个候选类别内的方差。若另一规则 $S$ 等概率取 $18,19,21,22$，则 $E[S]=80/4=20$，方差 $=(4+1+1+4)/4=10/4=2.5$；在给定目标 20 的情形下，两者偏差都为零，S 比 T 更精确。未给出其他参数值下的分布时，这个表只验证当前目标处的偏差；也不能由此声称 S 在所有可能估计量中最优。常数 7 的方差虽然为 0，但当目标是 20 时偏差为 $7-20=-13$，说明只压低方差不够。

<!-- bilingual-en:start -->
Another equally weighted rule taking 18, 19, 21, and 22 is unbiased at this target with variance 2.5. At target 20 both rules have zero bias, and S is more precise. Without distributions at other parameter values, this table verifies bias only at the stated target; it does not establish optimality among every possible estimator. A constant seven has zero variance but bias minus thirteen for target 20, so variance alone cannot define a universally good rule.
<!-- bilingual-en:end -->

[[经典 Gauss–Markov 定理]]中的 BLUE 展开为 Best Linear Unbiased Estimator：best 是该类别内方差最小，linear 是对观测结果的线性规则，unbiased 是无偏。经典条件包括设计矩阵满列秩、误差零条件均值以及条件协方差为常数乘单位阵等；[[BLUE正态性边界|不需要额外正态性才能得到 BLUE 结论]]。[[Cramér–Rao下界]]则在正则参数模型中给无偏估计量方差下界。二者各有适用范围，都不能读成“某个估计量无条件最好”。

<!-- bilingual-en:start -->
BLUE means best linear unbiased estimator, where best refers to variance within that class. Classical Gauss–Markov assumptions include full rank, zero conditional-mean errors, and a conditional covariance proportional to the identity. Normality is not required for the BLUE conclusion. Cramér–Rao instead supplies an unbiased variance bound in regular parametric models. Both conclusions have defined scopes.
<!-- bilingual-en:end -->

### 11.1 一致性：逐个读懂符号

<!-- bilingual-en:start -->
*Reading the consistency definition symbol by symbol*
<!-- bilingual-en:end -->

[[估计量一致性]]必须讨论随样本量变化的一列规则 $T_1,T_2,\ldots$：
$$\forall\varepsilon>0,\qquad P_\theta(|T_n-\theta|>\varepsilon)\longrightarrow0\quad(n\longrightarrow\infty).$$
$\forall$ 是“任意”；$\varepsilon$ 是预先固定的正容许误差；$T_n-\theta$ 是有正负号的误差；绝对值 $|\cdot|$ 把它转为距离；事件 $\{|T_n-\theta|>\varepsilon\}$ 表示估得太远；外面的 $P_\theta$ 计算这件事的概率；箭头要求该概率随 $n$ 增长趋零。这就是[[依概率收敛]]，也记作 $T_n\xrightarrow{p}\theta$。

<!-- bilingual-en:start -->
Consistency concerns a sequence of rules indexed by sample size. For every fixed positive tolerance, take the absolute estimation error, form the event that it exceeds the tolerance, and compute that event’s probability under the parameter value. That probability must tend to zero as n grows. This is convergence in probability.
<!-- bilingual-en:end -->

例如目标是 20，$\varepsilon=0.5$ 时，误差合格区间是 $[19.5,20.5]$；一致性要求落在区间外的概率趋零。换成 $\varepsilon=0.01$，区间变成 $[19.99,20.01]$，也必须有同样的极限性质。每个容许误差先固定，再让 $n$ 增长；不能只证明误差小于一个很宽范围，也不能把 $\varepsilon$ 偷换成随 $n$ 任意缩小的数列。等价说法是：对任意 $\varepsilon>0$ 和 $\delta>0$，存在 $N$，使所有 $n\ge N$ 的越界概率小于 $\delta$。

<!-- bilingual-en:start -->
At target 20, tolerance 0.5 allows the interval 19.5 to 20.5; tolerance 0.01 permits only 19.99 to 20.01. Each fixed tolerance must work as n grows. One wide interval is insufficient, and the definition does not freely replace the tolerance with an n-dependent sequence. Equivalently, any positive error tolerance and probability threshold are eventually satisfied for all larger sample sizes.
<!-- bilingual-en:end -->

一致性不是说一条实际路径每增加一个观察就必定更接近真值，也不是说某个有限样本量之后绝无大误差。单独给出固定 n 的一条抽样分布，通常不足以判定一致性。方差越来越小也不单独保证一致：分布可能集中到错误中心。另一方面，有限样本有偏并不排除一致，下一节的反例会逐项验证。

<!-- bilingual-en:start -->
Consistency does not require every realised step to move closer to the truth or eliminate all errors after a finite sample size. One fixed-n distribution usually cannot establish it. A shrinking variance may concentrate around the wrong centre, while finite-sample bias can coexist with consistency.
<!-- bilingual-en:end -->

![[Lecture 1 - Foundations.pdf#page=62]]

## 12. MSE：完整展开偏差—方差分解（slide 66）

<!-- bilingual-en:start -->
*Deriving the bias–variance decomposition of MSE*
<!-- bilingual-en:end -->

[[均方误差]]定义为 $\operatorname{MSE}_\theta(T)=E_\theta[(T-\theta)^2]$：先算误差，再平方，再对重复数据取期望。括号顺序重要，$E[(T-\theta)^2]$ 一般不等于 $(E[T]-\theta)^2$。设 $T$ 的二阶矩有限，目标 $\theta$ 固定。以下省略期望下标以便阅读，并令 $m=E[T]$；$m$ 是抽样分布的中心，是固定数，不是另一份随机估计。

<!-- bilingual-en:start -->
MSE first computes the estimation error, squares it, then averages over repeated data. Squaring before expectation differs from squaring the mean error. Assume a finite second moment and a fixed target, and write m for the estimator’s expectation. This centre is a fixed number.
<!-- bilingual-en:end -->

第一步，在误差中加上又减去 $m$：$T-\theta=(T-m)+(m-\theta)$。前半部分 $T-m$ 是围绕自身中心的随机波动；后半部分 $m-\theta$ 是中心偏离目标的偏差。第二步使用 $(a+b)^2=a^2+2ab+b^2$，逐项展开：
$$
\begin{aligned}
(T-\theta)^2
&=[(T-m)+(m-\theta)]^2\\
&=(T-m)^2+2(T-m)(m-\theta)+(m-\theta)^2.
\end{aligned}
$$
第三步对三项分别取期望，使用[[期望线性性]]；因为 $m-\theta$ 为常数，可以提出：
$$E[(T-\theta)^2]=E[(T-m)^2]+2(m-\theta)E[T-m]+(m-\theta)^2.$$
最后一项无需再写期望，是因为常数的期望等于自身。

<!-- bilingual-en:start -->
Add and subtract the estimator’s centre. One part is random fluctuation around that centre; the other is bias relative to the target. Expand the square into two squared terms and a cross term. Linearity distributes expectation across them, and fixed constants can be taken outside. The expectation of the final constant is itself.
<!-- bilingual-en:end -->

第四步，$E[T-m]=E[T]-m=m-m=0$，所以交叉项 $2(m-\theta)\times0=0$。第一项按照[[方差]]定义就是 $\operatorname{Var}(T)$，最后一项是偏差的平方。因此[[MSE偏差方差分解]]为：
$$\boxed{\operatorname{MSE}_\theta(T)=\operatorname{Var}_\theta(T)+\operatorname{Bias}_\theta(T)^2.}$$
这里没有用“偏差与波动独立”这样的假设；交叉项消失只因中心化的期望为 0。偏差项一定带平方。若无偏，MSE 等于方差；若有偏，仍可能因方差较低而取得更小 MSE。

<!-- bilingual-en:start -->
The centred estimator has expectation zero, so the cross term vanishes. The first term is variance and the last squared bias. No independence assumption between bias and fluctuation is needed. An unbiased estimator has MSE equal to variance, but a biased estimator may achieve lower MSE by reducing variance.
<!-- bilingual-en:end -->

| 规则 | 偏差 $B$ | 方差 $V$ | $B^2+V$ |
|---|---:|---:|---:|
| A | 0 | 16 | $0^2+16=16$ |
| B | 2 | 4 | $2^2+4=8$ |
| C | 3 | 1 | $3^2+1=10$ |

<!-- bilingual-en:start -->
*For the same target and sample size, rule B has the smallest MSE in this comparison, even though A is unbiased and C has the lowest variance.*
<!-- bilingual-en:end -->

本表中 B 的 MSE 最小：它既不是唯一无偏的 A，也不是方差最小的 C。选择指标前要明确问题：无偏性关心中心，有效性关心指定类别内的方差，MSE 关心平方损失的平均。一致性则研究 n 增长的极限，并不直接给有限样本规则排出优劣。

<!-- bilingual-en:start -->
Rule B minimises MSE among these three. Unbiasedness assesses the centre, efficiency compares variance within a specified class, and MSE averages squared loss. Consistency concerns the growing-sample limit and does not directly rank finite-sample performance.
<!-- bilingual-en:end -->

## 13. 有限样本与大样本：三个反例逐步检查（slides 67–69）

<!-- bilingual-en:start -->
*Checking finite-sample and asymptotic properties*
<!-- bilingual-en:end -->

统一模型为 $X_1,\ldots,X_n\overset{iid}{\sim}N(\theta,1)$：iid 是[[独立同分布]]；$N(\theta,1)$ 表示均值 $\theta$、方差 1 的正态分布。独立表示不同观察的随机性不相互依赖，同分布表示每个观察服从同一个分布，不表示每个观测值相等。目标始终为 $\theta$；比较的只是估计规则。

<!-- bilingual-en:start -->
Use iid normal observations with mean theta and variance one throughout. Independence and identical distributions do not mean identical realised values. Keep the target fixed and compare estimation rules.
<!-- bilingual-en:end -->

### 13.1 只用第一个观察：无偏但不一致

<!-- bilingual-en:start -->
*First observation: unbiased but inconsistent*
<!-- bilingual-en:end -->

取 $T_n=X_1$，不管拿到多少数据都只看第一项。期望为 $E[T_n]=E[X_1]=\theta$，所以每个 n 都无偏。方差为 $\operatorname{Var}(T_n)=1$，分布始终是 $N(\theta,1)$。为了真正检查[[无偏不推出一致]]，取固定 $\varepsilon=1$；令 $Z=X_1-\theta\sim N(0,1)$，便有
$$P(|T_n-\theta|>1)=P(|Z|>1)=2[1-\Phi(1)]\approx0.3173.$$
$\Phi(z)=P(Z\le z)$ 是标准正态[[累积分布函数|CDF]]。这个概率对所有 n 都一样，不能趋零；所以不一致。MSE $=0^2+1=1$ 也不改善。

<!-- bilingual-en:start -->
Using only the first observation gives expectation theta and variance one for every n. At fixed tolerance one, the error probability remains about 0.3173, using the standard normal CDF. This explicitly violates consistency. MSE stays one because the rule ignores new observations.
<!-- bilingual-en:end -->

![[Lecture 1 - Foundations.pdf#page=68]]

### 13.2 均值加 1／n：有偏但一致

<!-- bilingual-en:start -->
*Sample mean plus 1/n: biased but consistent*
<!-- bilingual-en:end -->

取 $S_n=\bar X_n+1/n$。先算期望：$E[S_n]=E[\bar X_n]+E[1/n]=\theta+1/n$，偏差就是 $1/n$。它对任何有限 n 都不为零。再算方差：加常数不会改变方差，所以 $\operatorname{Var}(S_n)=\operatorname{Var}(\bar X_n)=1/n$。正态的独立线性组合仍为正态，得到 $S_n\sim N(\theta+1/n,1/n)$。这条分布的中心逐渐移到真值，宽度也逐渐缩小。

<!-- bilingual-en:start -->
Adding 1/n to the mean gives expectation theta plus 1/n and nonzero finite-sample bias. Adding a constant leaves variance unchanged at 1/n. The normal linear-combination property gives the displayed normal distribution: its centre approaches the target and its spread shrinks.
<!-- bilingual-en:end -->

还要把直觉变成[[一致不推出无偏|一致性的证明]]。MSE 分解给出 $E[(S_n-\theta)^2]=1/n+1/n^2$。对非负随机变量 $(S_n-\theta)^2$ 使用[[Markov不等式]]：
$$P(|S_n-\theta|>\varepsilon)=P((S_n-\theta)^2>\varepsilon^2)\le\frac{1/n+1/n^2}{\varepsilon^2}\longrightarrow0.$$
等号因 $\varepsilon>0$，平方不改变绝对误差超过阈值的事件；不等号把尾部概率上界变成二阶矩；极限因固定分母 $\varepsilon^2>0$ 而分子趋零。这逐项满足一致性定义。

<!-- bilingual-en:start -->
The MSE decomposition gives squared error expectation 1/n plus 1/n squared. Apply Markov’s inequality to the nonnegative squared error. The fixed positive squared tolerance divides a numerator tending to zero, so the error probability tends to zero for every tolerance. This proves consistency rather than relying only on a picture.
<!-- bilingual-en:end -->

![[Lecture 1 - Foundations.pdf#page=69]]

### 13.3 均值加 5：方差趋零仍会集中到错误位置

<!-- bilingual-en:start -->
*Sample mean plus five: shrinking around the wrong value*
<!-- bilingual-en:end -->

取 $U_n=\bar X_n+5$。期望 $=\theta+5$，偏差 $=5$，方差 $=1/n$，MSE $=25+1/n\to25$。由[[固定偏移破坏一致性]]，其概率极限是 $\theta+5$，不是 $\theta$。例如当 $|\bar X_n-\theta|<1$ 时，$U_n-\theta$ 介于 4 与 6，因此 $|U_n-\theta|>2$；前一个事件概率趋 1，故后一个事件概率也趋 1。方差缩小只是“集中”，还要检查“集中到哪”。

<!-- bilingual-en:start -->
Adding five gives fixed bias five and shrinking variance 1/n, so MSE tends to 25. When the sample mean is within one of the target, the shifted estimate is between four and six above it. That event becomes overwhelmingly likely, making an error over two overwhelmingly likely too. Shrinking variance only describes concentration; its centre still matters.
<!-- bilingual-en:end -->

同样可以检查课堂的另外两个规则：$V_n=(X_1+X_2)/2$（$n\ge2$）的期望为 $\theta$、方差为 $(1/4)(1+1)=1/2$，新增数据未被使用，因此无偏但不一致。$W_n=\frac{n}{n+1}\bar X_n$ 的偏差为 $(\frac{n}{n+1}-1)\theta=-\theta/(n+1)$，方差为 $(\frac{n}{n+1})^2(1/n)=n/(n+1)^2$，MSE $=(\theta^2+n)/(n+1)^2\to0$，因此一致。它在 $\theta=0$ 处偏差恰为零，但在整个未知均值模型中不是无偏估计量。

<!-- bilingual-en:start -->
Averaging only the first two observations is unbiased with variance one half and ignores later data, so it is inconsistent. Multiplying the full sample mean by n/(n+1) instead produces the displayed bias, variance, and vanishing MSE, establishing consistency. Its bias happens to vanish at theta equal to zero, without making it unbiased over the whole model.
<!-- bilingual-en:end -->

无偏与一致之间没有双向蕴含；这里已经分别给出两个反例。有效性需要指定比较类别，不能把三种性质笼统说成“互相完全独立”。偏差趋零且方差趋零是 MSE 趋零、进而一致的一条充分路径，反向推理一般还需矩条件。大样本一致性也不会自动给出“n 等于某个数就足够”的承诺；有限样本仍需研究实际偏差、方差和尾部风险。

<!-- bilingual-en:start -->
Unbiasedness and consistency imply neither one another, as the two counterexamples establish. Efficiency needs a comparison class, so it is misleading to call all three properties completely independent. Vanishing bias and variance provide a sufficient route through vanishing MSE to consistency; reversing it generally needs additional moment conditions. Consistency alone does not guarantee an adequate particular finite sample size.
<!-- bilingual-en:end -->

## 14. 常见方法的条件提示与两条推断连接（slides 70–74）

<!-- bilingual-en:start -->
*Method conditions and the two inferential links*
<!-- bilingual-en:end -->

Lecture 1 的方法表是课程路标。[[普通最小二乘|OLS]]在经典条件下可无偏，[[OLS一致性条件|一致性]]还依赖外生性、识别所需变化及抽样极限条件；[[工具变量有效条件|IV／2SLS]]可能有限样本有偏，却在合适的有效工具与渐近条件下一致；[[极大似然估计|MLE]]使数据似然最大，其一致性还需识别与合适的正则条件；[[GMM矩条件|GMM]]用总体矩限制估计参数，还需矩识别和相应大样本条件。只有“likelihood 正确”或“moments 有效”的短句，不能替代完整定理。

<!-- bilingual-en:start -->
The estimator table is a set of signposts. OLS unbiasedness and consistency require their appropriate model and sampling assumptions. IV can be biased in finite samples yet consistent under suitable instruments and asymptotics. MLE maximises likelihood but also needs identification and regularity for consistency. GMM needs identifying moment restrictions and suitable limiting conditions. The lecture’s short condition labels are not complete theorems.
<!-- bilingual-en:end -->

[[识别与估计]]最终接成两条不同的连接。第一条是统计连接：在适当抽样条件下，$\hat\beta=\bar Y_1-\bar Y_0\xrightarrow{p}E[Y\mid D=1]-E[Y\mid D=0]=\beta$。第二条是识别连接：在相应因果条件下，$\beta=\mathrm{ATE}$。两条一起成立，才能说这个规则随着样本增长估到 ATE。也可以先从规则出发，求它的概率极限，再问该极限是不是研究目标；这是同一论证的另一种阅读顺序。

<!-- bilingual-en:start -->
The statistical link sends the sample mean difference to the observable population contrast under suitable sampling assumptions. The identification link equates that contrast with ATE under causal assumptions. Both are needed to estimate ATE consistently. One may equivalently start with the rule, derive its probability limit, then ask whether that limit is the desired target.
<!-- bilingual-en:end -->

后续 [[Lecture 2 - Statistics I.pdf|Statistics I]] 会继续展开期望、方差和抽样工具；本讲的 [[MSE偏差方差分解]]、[[样本均值协方差]]与[[样本协方差无偏分母]]仍是同一组共享知识。概率空间也连接 [[滤过定义|随机过程的信息结构]]，因果比较则连接 [[双重差分法|DID 的反事实构造]]。课程保持自己的讲解顺序，具体概念、条件和反例通过这些原子相通；需要另一门课的完整推导时，可以直接打开相应课程段落。

<!-- bilingual-en:start -->
Statistics I develops expectation, variance, and sampling tools further, while reusing the same MSE, sample-mean covariance, and unbiased covariance-divisor atoms. Probability spaces connect to information in stochastic processes; causal comparisons connect to DID counterfactuals. Each course retains its narrative order, with shared concepts, conditions, and counterexamples linking the routes. Direct course passages remain useful for complete alternative derivations.
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
*Sources and verification*
<!-- bilingual-en:end -->

- [[Lecture 1 - Foundations.pdf|Tom Glinnan, EC400 PSI Lecture 1 — Foundations]]：课程范围、顺序、记号及 slides 42–47、62、68–69 原图。
- [[PSI Lecture 1 - Foundations - 手写笔记.pdf|手写原稿]]与[[PSI Lecture 1 - Foundations - Claude课堂记录|课堂记录]]：保留课程中的例题与补充问题；原始文件未改写。
- [Durrett, Probability: Theory and Examples, §1.1](https://math.duke.edu/~rtd/PTE/PTE5_011119.pdf)：概率空间、σ-代数、概率测度的定义与闭包条件。
- [CMU 36-705 Lecture 24, pp.1–3](https://stat.cmu.edu/~larry/=stat705/Lecture24.pdf)：似然、先验、后验与频率派评价的关系。
- [Fithian, Statistical decision theory](https://www.stat.berkeley.edu/~wfithian/courses/stat210a/models.html)：统计目标、估计量、无偏性和平方损失风险。
- [Balakrishnan, CMU 36-705 Lecture 18](https://stat.cmu.edu/~siva/teaching/705/lec18.pdf)：MLE 一致性不能只靠“似然正确”。
- [NIST, Measures of Skewness and Kurtosis](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)：偏度、峰度与不同样本约定。

<!-- bilingual-en:start -->
*The lecture establishes scope, order, and notation. The retained originals supply classroom examples and questions. Durrett supports probability-space definitions; CMU Lecture 24 supports Bayesian objects and their frequentist assessment; Fithian supports estimation terminology and risk; Balakrishnan supports the qualification on MLE consistency; NIST supports shape-statistic conventions. Numerical examples and algebraic derivations have also been recalculated.*
<!-- bilingual-en:end -->
