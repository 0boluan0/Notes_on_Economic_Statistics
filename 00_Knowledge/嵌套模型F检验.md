---
aliases:
  - Classical nested-model F test
  - 受限与不受限回归 F 检验
student_os: knowledge-atom
atom_id: ECON-OLS-020
atom_set: regression-inference
atom_type: procedure
status: source-checked
part_of:
  - "[[回归推断.canvas|回归推断]]"
  - "[[回归模型比较与选择.canvas|回归模型比较与选择]]"
requires:
  - "[[线性组合推断]]"
  - "[[OLS经典协方差估计]]"
  - "[[BLUE正态性边界]]"
related:
  - "[[模型比较可比性]]"
  - "[[模型选择目标]]"
  - "[[回归t检验]]"
leads_to:
  - "[[回归联合显著边界]]"
  - "[[Chow结构稳定性检验]]"
---

<!-- bilingual-en:start -->
# Nested-model F test
<!-- bilingual-en:end -->

*嵌套模型 F 检验*

<!-- bilingual-en:start -->
A nested-model F test asks whether a **specified set of restrictions on regression coefficients** costs more fit than random noise would plausibly explain. “Nested” means that every permitted fit of the restricted model is also permitted by the larger, unrestricted model. A horizontal line is nested inside the family of all straight lines: set the slope to zero, while still allowing the intercept to vary.

Both models must describe the same observed response on the same rows. Fitting more freely can always reduce, or leave unchanged, the sum of squared residuals. The test therefore does not ask merely which fit is smaller; it asks how large the improvement is relative to the noise estimate and the number of restrictions.
<!-- bilingual-en:end -->

嵌套模型 F 检验问的是：对回归系数施加**一组明确限制**后，损失的拟合是否大到难以仅用随机噪声解释。“嵌套”表示受限模型允许的每一种拟合，也都属于更大的无约束模型。例如水平线包含在所有直线的集合中：把斜率设为零，但仍允许截距改变。

两个模型必须解释同一批观测行上的同一个响应。放宽拟合自由度，总能减小或保持残差平方和。因此检验不是简单问哪个拟合更小，而是把改善量与噪声估计、限制条数放在一起衡量。

> [!example] A fitted line versus a horizontal line · 直线与水平线的比较
> <!-- bilingual-en:start -->
> For $x=(0,1,2,3,4)$ and $y=(0,0,1,2,2)$, [[普通最小二乘|ordinary least squares (OLS)]] minimizes the sum of squared residuals. Its unrestricted fit is $\hat y=-0.2+0.6x$, with residuals $(0.2,-0.4,0,0.4,-0.2)$ and $RSS_U=0.4$. Imposing $H_0:\beta_1=0$ leaves only an intercept; its best value is $\bar y=1$, giving residuals $(-1,-1,0,1,1)$ and $RSS_R=4$.
>
> One restriction has increased RSS by $4-0.4=3.6$. The unrestricted model has $k=2$ coefficients and $n-k=3$ residual degrees of freedom, so its noise-variance estimate is $0.4/3$. The statistic is
> $$F=\frac{(4-0.4)/1}{0.4/(5-2)}=27.$$
> The calculation measures a large improvement relative to the estimated noise. Calling it significant still requires the sampling assumptions and reference law below.
> <!-- bilingual-en:end -->
>
> 对 $x=(0,1,2,3,4)$、$y=(0,0,1,2,2)$，[[普通最小二乘|普通最小二乘（OLS）]]最小化残差平方和，其无约束拟合为 $\hat y=-0.2+0.6x$，残差 $(0.2,-0.4,0,0.4,-0.2)$，$RSS_U=0.4$。施加 $H_0:\beta_1=0$ 后只剩截距，最佳值为 $\bar y=1$，残差 $(-1,-1,0,1,1)$，$RSS_R=4$。
>
> 一条限制让 RSS 增加 $3.6$。无约束模型有 $k=2$ 个系数、$n-k=3$ 个残差自由度，噪声方差估计为 $0.4/3$，因此 $F=27$。这个计算说明改善量相对估计噪声很大；要称为显著，还需要下面的抽样条件与参考分布。

<!-- bilingual-en:start -->
## Write the restrictions before comparing fits
<!-- bilingual-en:end -->

*先写限制，再比较拟合*

<!-- bilingual-en:start -->
Let $y=X\beta+u$ be the unrestricted model, where $y\in\mathbb R^n$ contains the responses, $X$ is an $n\times k$ design matrix, $\beta\in\mathbb R^k$ contains all coefficients, and $u$ is the unobserved error. Take $k\ge1$, and count any intercept in $k$. Assume $X$ has [[满列秩与OLS唯一性|full column rank]], meaning its columns are linearly independent, and $n>k$ so there is residual information for estimating noise variance.

State the [[零假设与备择假设|null hypothesis]] as $H_0:R\beta=r$. The fixed $q\times k$ matrix $R$ has one row of coefficient weights for each restriction, and $r\in\mathbb R^q$ contains their proposed values. Require $1\le q\le k$ and $q$ [[线性无关|linearly independent]] rows. The alternative is that **at least one** restriction fails, not that every restriction fails. Duplicate, consistent restrictions do not add degrees of freedom; contradictory restrictions describe an empty null rather than a useful F test.
<!-- bilingual-en:end -->

设无约束模型为 $y=X\beta+u$：$y\in\mathbb R^n$ 包含响应，$X$ 是 $n\times k$ 设计矩阵，$\beta\in\mathbb R^k$ 包含全部系数，$u$ 是不可观测误差。取 $k\ge1$，若有截距也计入 $k$。要求 $X$ [[满列秩与OLS唯一性|满列秩]]，即各列线性无关，并且 $n>k$，留下估计噪声方差的残差信息。

把[[零假设与备择假设|零假设]]写成 $H_0:R\beta=r$。固定的 $q\times k$ 矩阵 $R$ 每行给出一条系数权重，$r\in\mathbb R^q$ 列出所要求的数值。要求 $1\le q\le k$，且 $q$ 行[[线性无关]]。备择是**至少一条**限制不成立，不是所有限制都不成立。相容的重复限制不会增加自由度；互相矛盾的限制则描述空的零假设，不是一个有意义的 F 检验。

<!-- bilingual-en:start -->
Restrictions need not set coefficients to zero. For $\beta=(\beta_0,\beta_1,\beta_2)^\top$, $R=(0,1,1)$ and $r=1$ impose the one restriction $\beta_1+\beta_2=1$. In the course's log production model $\log Y=\beta_0+\beta_1\log K+\beta_2\log L+u$, this is constant returns to scale. Substituting $\beta_2=1-\beta_1$ fits the restricted model with two free coefficients. Even if an algebraic rearrangement is used for computation, compare squared residuals for the same original responses; the restriction has not licensed a different outcome or sample.
<!-- bilingual-en:end -->

限制不一定把系数设为零。对 $\beta=(\beta_0,\beta_1,\beta_2)^\top$，$R=(0,1,1)$、$r=1$ 施加的是一条限制 $\beta_1+\beta_2=1$。课程的对数生产模型 $\log Y=\beta_0+\beta_1\log K+\beta_2\log L+u$ 中，这表示规模报酬不变。代入 $\beta_2=1-\beta_1$ 后，受限模型只剩两个自由系数。即使计算时做了代数移项，比较的仍应是同一原始响应的残差平方；限制并没有允许更换结果变量或样本。

<!-- bilingual-en:start -->
## What the numerator and denominator measure
<!-- bilingual-en:end -->

*分子与分母各自衡量什么*

<!-- bilingual-en:start -->
Fit ordinary, unweighted least squares both without and with the restriction. Write $RSS_U=\sum_i(y_i-\hat y_{U,i})^2$ and $RSS_R=\sum_i(y_i-\hat y_{R,i})^2$. The unrestricted model has $k$ free coefficients; the restricted affine family has $k-q$. Thus the residual degrees of freedom increase from $n-k$ to $n-k+q$ when the restriction is imposed. For $RSS_U>0$, define
<!-- bilingual-en:end -->

分别做不加限制与加限制的普通无权重最小二乘拟合。记 $RSS_U=\sum_i(y_i-\hat y_{U,i})^2$、$RSS_R=\sum_i(y_i-\hat y_{R,i})^2$。无约束模型有 $k$ 个自由系数，受限的仿射族有 $k-q$ 个；因此施加限制后，残差自由度由 $n-k$ 增至 $n-k+q$。当 $RSS_U>0$ 时定义

$$
F=\frac{(RSS_R-RSS_U)/q}{RSS_U/(n-k)}.
$$

<!-- bilingual-en:start -->
The numerator is the additional squared residual per independent restriction. The denominator is the [[OLS经典协方差估计|unrestricted noise-variance estimate]] $s_U^2$; it is **not** $RSS_R/(n-k+q)$. Under the null model below, both quantities have expectation $\sigma^2$, so a large ratio indicates unusually costly restrictions. This does not say that the ratio itself has expectation one.

Nested least squares ensures $RSS_R\ge RSS_U$. A substantial negative difference signals that the fits, rows, constraints or objectives are not the stated comparison; tiny negative numerical roundoff should be distinguished from a substantive mismatch. If $RSS_U=0$, the displayed ratio does not define an ordinary F statistic. Under the nondegenerate Gaussian conditions below and $n>k$, that event has probability zero.
<!-- bilingual-en:end -->

分子是每条独立限制增加的残差平方，分母是[[OLS经典协方差估计|无约束模型的噪声方差估计]] $s_U^2$，**不是** $RSS_R/(n-k+q)$。下面的零假设模型中，两者的期望都是 $\sigma^2$，很大的比值表示限制造成的损失异常大；这不表示比值自身的期望等于一。

嵌套最小二乘保证 $RSS_R\ge RSS_U$。若差值明显为负，应核对拟合、观测行、限制或优化准则是否真是声明的比较；微小的数值舍入误差与实质错配须分开。若 $RSS_U=0$，上述比值不定义普通 F 统计量；在下面的非退化 Gaussian 条件及 $n>k$ 下，该事件概率为零。

<!-- bilingual-en:start -->
## Why the classical reference distribution is F
<!-- bilingual-en:end -->

*经典 F 参考分布从哪里来*

<!-- bilingual-en:start -->
For fixed $X$, or conditional on $X$, a sufficient exact model is $u\mid X\sim N_n(0,\sigma^2I_n)$ with $0<\sigma^2<\infty$: a correct conditional mean and [[联合Gaussian|joint Gaussian errors]] with common variance and zero cross-covariances. Here $I_n$ is the identity matrix. Under $H_0$, the two quantities
<!-- bilingual-en:end -->

对固定的 $X$，或条件于 $X$，一套充分的精确模型是 $u\mid X\sim N_n(0,\sigma^2I_n)$，其中 $0<\sigma^2<\infty$：条件均值正确，误差[[联合Gaussian]]、共同方差且交叉协方差为零。其中 $I_n$ 是单位矩阵。零假设下，以下两个量

$$
\frac{RSS_R-RSS_U}{\sigma^2}\sim\chi_q^2,
\qquad
\frac{RSS_U}{\sigma^2}\sim\chi_{n-k}^2
$$

<!-- bilingual-en:start -->
are independent conditional on $X$. The first measures Gaussian noise along the $q$ fitted directions removed by the restrictions; the second measures it along the $n-k$ directions left outside the full fitted space. These directions are [[正交|perpendicular]], meaning their dot products are zero, and the joint Gaussian model makes their components independent. Their squared sums are [[卡方分布|chi-square variables]]. Dividing each by its own degrees of freedom gives the defining ratio of the [[F分布|F distribution]], hence
<!-- bilingual-en:end -->

在给定 $X$ 后相互独立。前者测量被限制移除的 $q$ 个拟合方向上的 Gaussian 噪声，后者测量完整拟合空间之外 $n-k$ 个方向上的噪声。两组方向[[正交|互相垂直]]，即点积为零，联合 Gaussian 模型使对应分量独立；平方和就是[[卡方分布|卡方变量]]。各自除以自由度后取比，恰是[[F分布]]的定义，因此

$$
F\mid X,H_0\sim F_{q,n-k}.
$$

<!-- bilingual-en:start -->
The order of the two degrees of freedom matters. Use the upper tail: reject a prespecified level-$\alpha$ test when $F_{\rm obs}>F_{q,n-k;1-\alpha}$, with the subscript after the semicolon denoting a lower-tail quantile. Equivalently report the [[p值条件解释|p-value]] $P(F_{q,n-k}\ge F_{\rm obs})$. In the five-point example, $F=27$ against $F_{1,3}$ gives $p\approx0.0138$. The evidence is against a zero slope within this model; it does not certify the unrestricted model or a causal explanation.

This normality assumption is a sufficient route to the displayed exact law. [[经典 Gauss–Markov 定理|BLUE under Gauss–Markov]] is a variance-ordering statement and does not, by itself, imply it. For a general error covariance, a [[Wald联合检验|joint Wald test]] using a justified covariance estimate and calibration may be appropriate. An unchanged ordinary RSS ratio does not become heteroskedasticity-, HAC-, or cluster-robust merely by changing its label.
<!-- bilingual-en:end -->

两个自由度的顺序不能颠倒。使用上尾：预先指定水平 $\alpha$ 时，若 $F_{\rm obs}>F_{q,n-k;1-\alpha}$ 就拒绝，分号后的下标表示下尾分位数。也可以报告[[p值条件解释|p 值]] $P(F_{q,n-k}\ge F_{\rm obs})$。五点例子以 $F_{1,3}$ 为参考，$F=27$ 得到 $p\approx0.0138$。这是该模型中反对零斜率的证据，不是对无约束模型或因果解释的认证。

上述正态性条件是得到精确分布的一条充分路径。[[经典 Gauss–Markov 定理|Gauss–Markov 下的 BLUE]]是方差排序结论，本身不推出这个分布。一般误差协方差下，可考虑具有相应估计与校准依据的[[Wald联合检验]]；普通 RSS 比值不可能只改标签就自动变成异方差、HAC 或聚类稳健检验。

> [!note]- Connecting one restriction to a t test · 一条限制为何对应 t 的平方
> <!-- bilingual-en:start -->
> Let $G=X^\top X$ and $d=R\hat\beta-r$, where $\top$ denotes [[矩阵转置|transpose]]. Full rank gives the least-squares identity
> $$RSS_R-RSS_U=d^\top[RG^{-1}R^\top]^{-1}d.$$
> It can be checked by writing the restricted coefficient estimate as $\tilde\beta=\hat\beta-G^{-1}R^\top[RG^{-1}R^\top]^{-1}d$: it satisfies the restriction, and the OLS residual is perpendicular to every column of $X$, so the cross term in the squared residual vanishes. When $q=1$, write $R=a^\top$ and $r=c$. With the same classical variance estimate,
> $$F=\frac{(a^\top\hat\beta-c)^2}{s_U^2a^\top G^{-1}a}=t^2.$$
> For the worked data, $t=3\sqrt3$ and $F=27$. This algebraic equality does not itself require Gaussian errors. Under the stated Gaussian null, the upper-tail F test and the **two-sided** [[回归t检验|t test]] have the same p-value; a one-sided t test also uses the sign, which the square removes. For more than one restriction there is generally no single coefficient t test that answers the joint question.
> <!-- bilingual-en:end -->
>
> 记 $G=X^\top X$、$d=R\hat\beta-r$，$\top$ 表示[[矩阵转置|转置]]。满秩下有上面的残差平方差恒等式。把受限估计写成 $\tilde\beta=\hat\beta-G^{-1}R^\top[RG^{-1}R^\top]^{-1}d$，可检查它满足限制；OLS 残差又与 $X$ 的每一列垂直，所以展开残差平方时交叉项消失。只有一条限制时，写 $R=a^\top$、$r=c$，用同一个经典方差估计便得到 $F=t^2$。例子里 $t=3\sqrt3$，所以 $F=27$。
>
> 这个代数等式本身不要求 Gaussian 误差；在声明的 Gaussian 零假设下，F 上尾检验与**双侧**[[回归t检验|t 检验]]有相同 p 值。单侧 t 检验还利用符号，平方会把它去掉。多条限制时，通常不能用某个单独系数的 t 检验回答联合问题。

<!-- bilingual-en:start -->
## Keep the comparison and the question fixed
<!-- bilingual-en:end -->

*保持比较对象与检验问题一致*

<!-- bilingual-en:start -->
Nesting is a relation between the permitted fits, not a comparison of their names or parameter counts. Two models with different functional forms need not be nested. Different observation rows, outcome scales, weights or fitting objectives also invalidate direct substitution into this ordinary RSS formula. Known-weight or transformed models may have their own properly derived tests, but their assumptions and residual metric must be stated.

A prespecified F test evaluates its stated restrictions. Searching through many models and then treating the chosen test as if it were specified in advance omits [[选择后推断|selection uncertainty]]. Whether the goal is an explanatory restriction, prediction or a search procedure belongs to [[模型选择目标|the model-selection objective]]; a significant F value does not automatically choose the true model.
<!-- bilingual-en:end -->

嵌套是允许拟合集合之间的关系，不是比较模型名字或参数个数。函数形式不同的两个模型未必嵌套。观测行、响应尺度、权重或拟合目标不同，也不能直接代入这个普通 RSS 公式。已知权重或变换模型可能有另外推导的有效检验，但须说明条件与残差度量。

预先指定的 F 检验评价的是那组限制。搜索很多模型后，再把选中的检验当成事先给定，会漏掉[[选择后推断|选择带来的不确定性]]。目标究竟是解释性的限制、预测，还是模型搜索，属于[[模型选择目标]]；显著的 F 值不会自动选出真模型。

> [!question]- Five hundred rows versus four hundred seventy · 两个样本量不同的拟合
> <!-- bilingual-en:start -->
> A restricted wage regression uses 500 rows; the unrestricted fit uses 470 because an added variable has missing values. Can their RSS values enter the formula? No: the difference mixes a sample change with the effect of restrictions. Refit both on the same 470 rows, then check nesting and the sampling conditions. This makes the RSS comparison coherent; it does not by itself justify the missing-data mechanism.
> <!-- bilingual-en:end -->
>
> 受限工资回归用 500 行；新增变量有缺失，无约束回归只用 470 行。能否直接比较 RSS？不能，差值混入了样本变化。应先在同一 470 行上重新拟合两者，再检查嵌套和抽样条件。这样才有一致的 RSS 比较，但并没有因此证明缺失机制合适。

## 来源与核验

<!-- bilingual-en:start -->
- [Penn State STAT 501, Lesson 6, §§6.1–6.2 and the partial-F example](https://online.stat.psu.edu/stat501/Lesson06#the-general-linear-f-test): the independent zero-mean normal-error setup, full/reduced fits, RSS difference, degrees of freedom, upper-tail calibration and the one-slope $t^2$ relation.
- [Hansen, *Econometrics* (2010), §4.13, PDF pp.68–70; §§6.3–6.4, pp.98–100](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf#page=99): Gaussian projection independence, classical versus general covariance estimates, and the RSS/Wald identity. The general $R\beta=r$ formula above is checked by substitution and orthogonal residual decomposition; the five-point numerical example is independently recomputed.
- [[02_Economy/01_Econometrics/03_多元线性回归.md#3.2. 多参数检验：F 检验|Local course F tests]] and [[02_Economy/01_Econometrics/03_多元线性回归.md#3.2.4. 具体步骤示例|the constant-returns restriction]] preserve the course context and the nonzero right-hand side. Its $n-k-1$ convention counts only slopes in $k$; this card counts all coefficients, giving $n-k$.
- [MIT 14.310x, Lecture 17, PDF pp.53–56](https://ocw.mit.edu/courses/14-310x-data-analysis-for-social-scientists-spring-2023/mit14_310x_s23_week08_lec17.pdf#page=53): supplementary support for the normal regression model and estimated residual variance; the F formula is supported by Penn State and Hansen rather than attributed to these slides.
<!-- bilingual-en:end -->

- Penn State 第6课：核对独立、零均值正态误差设定，完整/受限拟合、RSS 差、自由度、上尾校准及单斜率 $t^2$ 关系。
- Hansen 对应章节：核对 Gaussian 投影独立、经典与一般协方差估计及 RSS/Wald 恒等式。一般 $R\beta=r$ 写法在本卡通过代入和正交残差分解核验；五点例子独立重算。
- 本地 F 检验与规模报酬例子保留课程语境及非零限制右端。原课程 $n-k-1$ 把 $k$ 记为斜率数，本卡把全部系数记入 $k$，故写 $n-k$。
- MIT 第17讲第53–56页补充支持正态回归模型和残差方差估计；F 公式依据 Penn State 与 Hansen，不归给这些幻灯片。
