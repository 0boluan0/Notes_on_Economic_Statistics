---
aliases:
  - MANOVA reference-distribution calibration
  - MANOVA approximate F calibration
  - MANOVA参考分布校准
student_os: knowledge-atom
atom_id: STAT-MAN-004
atom_set: manova
atom_type: inference-boundary
status: source-checked
part_of:
  - "[[MANOVA 多元方差分析.canvas|MANOVA 多元方差分析]]"
requires:
  - "[[MANOVA四种统计量]]"
  - "[[MANOVA经典假设]]"
implies:
  - "[[多响应联合显著边界]]"
related:
  - "[[Wishart抽样假设]]"
  - "[[Pillai稳健性边界]]"
---

<!-- bilingual-en:start -->
# From a MANOVA statistic to a p-value
<!-- bilingual-en:end -->
MANOVA 统计量如何变成 p 值

<!-- bilingual-en:start -->
Suppose two analyses both report Wilks' $\Lambda=0.5$. That number describes a balance between fitted group differences and remaining variation. It does not tell us how surprising the result is: the same number can be ordinary with little data and unusual with much more data. **Calibration** supplies the reference distribution needed to make that comparison.
<!-- bilingual-en:end -->
假设两次分析都报告 Wilks 的 $\Lambda=0.5$。这个数描述拟合出的组间差异与剩余变动之间的关系，还没有说明结果有多罕见：数据少时可能很普通，数据多时可能很不寻常。**校准**就是为这种比较提供参考分布。

<!-- bilingual-en:start -->
[[MANOVA|MANOVA]] tests a joint question about several response means. Its [[MANOVA SSCP分解|hypothesis and error SSCP matrices]] $H$ and $E$ collect the relevant fitted differences and residual variation. The [[MANOVA四种统计量|four classical criteria]] summarize these matrices differently. Wilks' $\Lambda$ counts smaller values as stronger evidence against the null; Pillai's $V$, Hotelling–Lawley's $U$, and Roy's largest root $\Theta$ count larger values.
<!-- bilingual-en:end -->
[[MANOVA]]对多个响应均值提出联合检验问题。[[MANOVA SSCP分解|假设与误差 SSCP 矩阵]] $H,E$ 分别汇总相关的拟合差异和残差变动；[[MANOVA四种统计量|四种经典统计量]]用不同方式概括它们。Wilks 的 $\Lambda$ 越小越不利于零假设，Pillai 的 $V$、Hotelling–Lawley 的 $U$ 和 Roy 最大根 $\Theta$ 则是越大越不利。

<!-- bilingual-en:start -->
## What the reference distribution must know
<!-- bilingual-en:end -->
### 参考分布需要哪些信息

<!-- bilingual-en:start -->
Under a specified null hypothesis $H_0$ and sampling model, a [[p值条件解释|p-value]] is the probability of a statistic at least as extreme as the observed one. For Wilks this is a lower-tail probability; for the other three it is an upper-tail probability. The computation assumes $H_0$ for its reference calculation; it does not assign a probability that $H_0$ is true.
<!-- bilingual-en:end -->
在指定零假设 $H_0$ 与抽样模型后，[[p值条件解释|p 值]]是得到至少同样极端统计量的概率。Wilks 使用下尾，另外三者使用上尾。计算时把 $H_0$ 当作参考前提，并没有给出“$H_0$ 为真的概率”。

<!-- bilingual-en:start -->
For the usual full-rank multivariate linear model, distinguish three dimensions. Here $p$ is the number of linearly independent response coordinates being tested, $q$ is the number of linearly independent, estimable coefficient restrictions per response, and $\nu$ is the residual degrees of freedom. In one-way MANOVA with $g$ nonempty groups and total sample size $N$, testing equality of all group means gives $q=g-1$ and $\nu=N-g$. A single planned group contrast instead has $q=1$, even with more than two groups.
<!-- bilingual-en:end -->
在通常的满秩多元线性模型中，要区分三个维度：$p$ 是接受检验的线性独立响应坐标数；$q$ 是对每个响应施加的线性独立、可估系数约束数；$\nu$ 是残差自由度。单因素 MANOVA 有 $g$ 个非空组、总样本量 $N$，检验所有组均值相等时，$q=g-1$、$\nu=N-g$。若只检验一个预设组间对比，即使组数超过二，仍是 $q=1$。

<!-- bilingual-en:start -->
More generally, with a fixed design matrix $X$, $\nu=N-\operatorname{rank}(X)$: fitting the model uses up $\operatorname{rank}(X)$ independent directions across observations, leaving $\nu$ residual directions. The hypothesis dimension $q$ comes from the specified estimable restriction, not the observed rank of $H$. For example, $H$ is a $p\times p$ matrix and cannot have rank greater than $p$, but $g-1$ can exceed $p$. Duplicating a response or a constraint does not create an additional independent dimension.
<!-- bilingual-en:end -->
更一般地，对固定设计矩阵 $X$，$\nu=N-\operatorname{rank}(X)$：拟合模型占用 $\operatorname{rank}(X)$ 个线性独立的观测方向，留下 $\nu$ 个残差方向。假设维度 $q$ 来自预设的可估约束，不能用观测到的 $H$ 的秩替代。例如，$H$ 是 $p\times p$ 矩阵，秩不可能超过 $p$，但 $g-1$ 可以超过 $p$。重复一个响应或一个约束，不会增加独立维度。

> [!important] Dimensions belong to a model / 维度依赖模型
> <!-- bilingual-en:start -->
> The classical reference law also uses independent Gaussian errors with a common positive-definite response covariance, estimable restrictions, and enough residual information. Under that model, $\nu\ge p$ makes $E$ invertible almost surely; for fixed numerical data, check its actual rank too. See [[MANOVA经典假设|the classical MANOVA assumptions]]. A successful software calculation does not establish these conditions.
> <!-- bilingual-en:end -->
>
> 经典参考分布还依赖独立 Gaussian 误差、共同正定的响应协方差、可估约束和足够的残差信息。在该模型下，$\nu\ge p$ 保证 $E$ 几乎必然可逆；对固定数值数据仍须核对实际秩。具体见[[MANOVA经典假设]]。软件成功算出结果不等于这些条件已经成立。

<!-- bilingual-en:start -->
## An exact special case shows why degrees of freedom matter
<!-- bilingual-en:end -->
### 用一个精确特例看自由度的作用

<!-- bilingual-en:start -->
Compare two independent Gaussian groups with common positive-definite covariance and $p$ responses. Write $\nu=n_1+n_2-2\ge p$, with each group size at least two. There is one group contrast, $q=1$. The [[pooled Hotelling T²|pooled Hotelling test]] implies the exact null transformation
<!-- bilingual-en:end -->
比较两个独立 Gaussian 组，设共同协方差正定、每个观测含 $p$ 个响应，两组样本量均至少为二，$\nu=n_1+n_2-2\ge p$。这里只有一个组间对比，$q=1$。由[[pooled Hotelling T²]]可得精确零假设转换

$$
F_{\rm obs}=\frac{\nu-p+1}{p}\frac{1-\Lambda}{\Lambda},
\qquad F\sim F_{p,\nu-p+1}\quad\text{under }H_0.
$$

<!-- bilingual-en:start -->
To see the connection, $E=\nu S_p$ and $H=dd^T/h$, where $d$ is the difference of sample means and $h=1/n_1+1/n_2$. The only possibly nonzero root of $E^{-1}H$ is $T^2/\nu$, so $\Lambda=1/(1+T^2/\nu)$. Substitution gives the displayed F transformation. The [[F分布|F distribution]] has ordered numerator and denominator degrees of freedom.
<!-- bilingual-en:end -->
这里 $E=\nu S_p$，$H=dd^T/h$，$d$ 是样本均值差，$h=1/n_1+1/n_2$。$E^{-1}H$ 唯一可能非零的根为 $T^2/\nu$，因此 $\Lambda=1/(1+T^2/\nu)$，代入即得上述 F 转换。[[F分布]]的分子、分母自由度有固定顺序。

> [!example] The same Lambda, different evidence / 相同 Lambda，不同证据
> <!-- bilingual-en:start -->
> Let $p=2$ and observe $\Lambda=0.5$ in each analysis. With $\nu=9$, the result is $F_{\rm obs}=4$ on $(2,8)$ degrees of freedom, with upper-tail probability $0.0625$. With $\nu=19$, it is $F_{\rm obs}=9$ on $(2,18)$ degrees of freedom, with probability $0.001953125$. These are illustrative observations under the stated model, not a claim that Lambda stays fixed when data are added. For these checks, $P(F_{2,b}\ge x)=(b/(b+2x))^{b/2}$.
> <!-- bilingual-en:end -->
>
> 两次都取 $p=2$、观测 $\Lambda=0.5$。$\nu=9$ 时，$F_{\rm obs}=4$、自由度 $(2,8)$，上尾概率为 $0.0625$；$\nu=19$ 时，$F_{\rm obs}=9$、自由度 $(2,18)$，上尾概率为 $0.001953125$。这是指定模型下的示意观测，不是说增加数据后 Lambda 必然不变。数值可用 $P(F_{2,b}\ge x)=(b/(b+2x))^{b/2}$ 核对。

<!-- bilingual-en:start -->
## What an F column in software means
<!-- bilingual-en:end -->
### 软件里的 F 列表示什么

<!-- bilingual-en:start -->
For general MANOVA hypotheses, software commonly transforms a criterion and uses an **approximate** F reference. R's `summary.manova` documents such output: the four transformed values coincide for a one-degree-of-freedom term; otherwise its Roy F value is an upper bound. An output label alone therefore cannot establish an exact finite-sample F law.
<!-- bilingual-en:end -->
对一般 MANOVA 假设，软件常把统计量转换后使用**近似** F 参考分布。R 的 `summary.manova` 明确说明这种输出：被检验项只有一个自由度时，四个转换后的 F 值相同；其他情形下，它的 Roy F 值是一个上界。因此，仅凭输出列名不能断定存在精确有限样本 F 分布。

<!-- bilingual-en:start -->
SAS/STAT 12.3 documents a separate distinction: the default uses F approximations, whereas `MSTAT=EXACT` computes exact-model tail probabilities for Wilks, Hotelling–Lawley and Roy, and uses an improved F approximation for Pillai. An exact tail calculation need not mean a statistic has an exact F distribution. Neither numerical precision nor an “exact” option repairs a false sampling model.
<!-- bilingual-en:end -->
SAS/STAT 12.3 的文档还区分：默认使用 F 近似；`MSTAT=EXACT` 对 Wilks、Hotelling–Lawley、Roy 计算相应精确模型的尾概率，对 Pillai 则使用改进的 F 近似。精确尾概率计算并不意味着统计量本身服从精确 F 分布。数值精度或名为“exact”的选项，都不能修复不成立的抽样模型。

<!-- bilingual-en:start -->
A useful report identifies the tested effect, criterion and observed value, response and hypothesis dimensions, residual degrees of freedom, calibration method, reported F degrees when applicable, and p-value. Choose the criterion for a stated reason before inspecting competing p-values. Selecting the smallest afterward changes the testing procedure. If errors are dependent or heterogeneous, or responses have problematic rank, justify a suitable alternative calibration; simply choosing [[Pillai稳健性边界|Pillai's trace]] is not that justification. A joint rejection still needs the [[多响应联合显著边界|appropriate response-wise interpretation]].
<!-- bilingual-en:end -->
报告应交代被检验效应、统计量及观测值、响应与假设维度、残差自由度、校准方法、适用时的 F 自由度及 p 值。在查看几个竞争性 p 值之前，应有理由地选择统计量；事后只挑最小值会改变整个检验程序。若误差相关、协方差异质或响应秩有问题，就须论证合适的替代校准；仅换成[[Pillai稳健性边界|Pillai trace]]还不够。联合拒绝后的逐响应解释另见[[多响应联合显著边界]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.5. 多个总体均值向量比较：单因子 MANOVA|Course Chapter 6 §1.5–1.6]]: one-way scope, $g-1$ and $N-g$, and calibration as a separate step.
- [Penn State STAT 505, Lesson 8](https://online.stat.psu.edu/stat505/Lesson08), §§8.2–8.3 and 8.8: SSCP degrees, general F approximations, and the single-contrast Wilks conversion. The two-group example is derived explicitly from the pooled Hotelling theorem; its numerical tails are checked from the [NIST F density](https://www.itl.nist.gov/div898/handbook/eda/section3/eda3665.htm).
- [R `summary.manova`](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/summary.manova.html), Details: approximate output, one-df equality and the Roy bound.
- [SAS/STAT 12.3 GLM, Multivariate Analysis of Variance](https://support.sas.com/documentation/cdl/en/statug/66103/HTML/default/statug_glm_details45.htm): error model, estimable linear hypotheses, residual degrees and the documented meaning of `MSTAT=EXACT`. Its displayed covariance formula omits transpose marks in the extracted alternative text; no dimensionally invalid formula is used here.
<!-- bilingual-en:end -->
- 课程第6章§1.5–1.6核对单因素范围、两类自由度与校准步骤；PSU §8.2–8.3、8.8核对对应公式和单对比转换。两组例子另由pooled Hotelling定理推导，数值尾概率由NIST密度直接积分核验。
- R与指定版本SAS官方文档核对软件输出含义；“精确”仍以模型成立为前提。英文来源条目中的链接对应这些原始说明。
