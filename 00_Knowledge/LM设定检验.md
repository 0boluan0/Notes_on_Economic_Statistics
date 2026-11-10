---
aliases:
  - "LM specification test"
  - "拉格朗日乘数设定检验"
  - "LM 遗漏变量检验"
student_os: knowledge-atom
atom_id: ECON-SPEC-006
atom_type: method
status: source-checked
part_of:
  - "[[回归模型设定与函数形式.canvas|回归模型设定与函数形式]]"
requires:
  - "[[残差图边界]]"
contrasts_with:
  - "[[RESET检验边界]]"
---

<!-- bilingual-en:start -->
# LM specification test
<!-- bilingual-en:end -->
*LM 设定检验*

<!-- bilingual-en:start -->
Suppose a straight-line model describes average spending using income. You have a specific concern: perhaps the curve also needs income squared. An **LM specification test** asks whether the errors left by the restricted fit still contain evidence in that proposed direction. Here “restricted” means that the extra coefficient is forced to be zero. The test starts from that simpler fit; it does not search through every possible explanation of a poor model.
<!-- bilingual-en:end -->

假设一个直线模型用收入描述平均消费。你有一个具体疑问：曲线是否还需要收入的平方项？**LM 设定检验**检查：受限模型拟合后留下的偏差，是否在这个候选方向上仍包含证据。“受限”表示把额外项的系数强制设为零。检验从这个较简单的拟合出发，不会搜索模型不佳的所有可能原因。

<!-- bilingual-en:start -->
LM stands for **Lagrange multiplier** and is also called a score test. A [[得分函数|score]] is the slope of the log likelihood as a parameter changes. Broadly, an LM test asks whether the likelihood still slopes substantially toward the excluded directions when the null is imposed. This card develops the common linear-regression version, where that idea becomes an auxiliary regression of residuals.
<!-- bilingual-en:end -->

LM 是 **Lagrange multiplier（拉格朗日乘数）**的缩写，也称得分检验。[[得分函数|得分]]是参数改变时对数似然的斜率。广义上，LM 检验检查：施加零假设后，似然是否仍明显向被排除的方向倾斜。本卡具体说明常见的线性回归版本：把这个想法落实为一次残差辅助回归。

<!-- bilingual-en:start -->
## From a restricted fit to an auxiliary regression
<!-- bilingual-en:end -->
*从受限拟合到辅助回归*

<!-- bilingual-en:start -->
Let $y$ contain the outcomes for $n$ observations. Let $X$ be the $n\times k$ matrix of retained regressors, including an intercept, with [[线性无关|linearly independent]] columns. A candidate matrix $Z$ contains terms chosen before testing, such as income squared or an income–price interaction. In the expanded mean model, the [[零假设与备择假设|null hypothesis]] sets their coefficients to zero:
<!-- bilingual-en:end -->

令 $y$ 装着 $n$ 个观测的结果；$X$ 是 $n\times k$ 的保留回归量矩阵，包含截距，且各列[[线性无关]]。候选矩阵 $Z$ 装着检验前选定的项，例如收入平方、收入与价格的交互项。在扩展的均值模型中，[[零假设与备择假设|零假设]]把它们的系数设为零：

$$y=X\beta+Z\gamma+u,\qquad H_0:\gamma=0.$$

<!-- bilingual-en:start -->
Here $u$ is the model error, while $\beta$ and $\gamma$ are coefficient vectors. Fit $y$ on $X$ by [[普通最小二乘|OLS]], which minimizes squared fitting errors. Save the [[回归残差|residuals]] $\hat u_R=y-X\hat\beta_R$. Then regress these residuals on **both $X$ and $Z$**, using the same observations. The auxiliary response is $\hat u_R$, not $\hat u_R^2$: the question here concerns omitted mean directions, not a variance-pattern test.
<!-- bilingual-en:end -->

其中 $u$ 是模型误差，$\beta,\gamma$ 是系数向量。先用[[普通最小二乘|OLS]] 把 $y$ 对 $X$ 回归，即最小化拟合偏差的平方和，保留[[回归残差|残差]] $\hat u_R=y-X\hat\beta_R$。再用同一批观测，把这些残差对 **$X$ 和 $Z$ 一起**回归。辅助回归的结果变量是 $\hat u_R$，不是 $\hat u_R^2$：这里检查遗漏的均值方向，不是在检验方差模式。

<!-- bilingual-en:start -->
Retaining $X$ accounts for the part of each candidate already represented by the original regressors. Equivalently, first remove that part from $Z$, as in [[FWL残差化定理|FWL residualisation]]. The number of genuinely new directions is the increase in [[矩阵秩|matrix rank]]:
<!-- bilingual-en:end -->

保留 $X$，是为了控制候选项中已经由原回归量表示的部分。等价地，可以先像 [[FWL残差化定理|FWL 残差化]]那样，从 $Z$ 中去掉这部分。真正新增的方向数，是[[矩阵秩]]的增量：

$$q=\operatorname{rank}([X\ Z])-k.$$

<!-- bilingual-en:start -->
Remove redundant candidate columns before computing a joint test. If $q=0$, there is no additional direction to test. For the formulas below, require $q>0$, $n>k+q$, and positive residual sums of squares. Write $RSS_R=\sum_i\hat u_{R,i}^2$ for the restricted residual sum of squares and $RSS_U$ for that from fitting $y$ on the augmented columns. The auxiliary regression has the same final residual as the augmented fit. Because $X$ includes an intercept, the restricted residuals have mean zero, so its ordinary centered $R^2$ is
<!-- bilingual-en:end -->

计算联合检验前应去掉冗余候选列。若 $q=0$，就没有新增方向可检验。以下公式要求 $q>0$、$n>k+q$，且相关残差平方和为正。把受限残差平方和记为 $RSS_R=\sum_i\hat u_{R,i}^2$，把 $y$ 对扩展回归量拟合后的残差平方和记为 $RSS_U$。辅助回归与扩展拟合的最终残差相同。由于 $X$ 含截距，受限残差均值为零，因此辅助回归通常的中心化 $R^2$ 为

$$R_{\mathrm{aux}}^2=\frac{RSS_R-RSS_U}{RSS_R},\qquad LM=nR_{\mathrm{aux}}^2.$$

> [!example] A missed curve in six observations · 六个观测中的遗漏曲线
> <!-- bilingual-en:start -->
> Take $x=(0,0,1,1,2,2)$ and $y=(11,9,16,14,27,25)$. Fitting an intercept and $x$ gives $\hat y_R=9+8x$, residuals $(2,0,-1,-3,2,0)$, and $RSS_R=18$. Adding the proposed $x^2$ gives $\hat y_U=10+2x+3x^2$ and residuals $(1,-1,1,-1,1,-1)$, so $RSS_U=6$. Thus $q=1$, $R_{\mathrm{aux}}^2=2/3$, and $LM=6(2/3)=4$. This small example demonstrates the calculation; it does not make a large-sample approximation accurate at $n=6$.
> <!-- bilingual-en:end -->
>
> 取 $x=(0,0,1,1,2,2)$、$y=(11,9,16,14,27,25)$。拟合截距和 $x$ 得到 $\hat y_R=9+8x$，残差为 $(2,0,-1,-3,2,0)$，故 $RSS_R=18$。加入候选 $x^2$ 后得到 $\hat y_U=10+2x+3x^2$，残差为 $(1,-1,1,-1,1,-1)$，故 $RSS_U=6$。于是 $q=1$、$R_{\mathrm{aux}}^2=2/3$、$LM=6(2/3)=4$。这个小例子展示计算，不意味着 $n=6$ 时大样本近似已经准确。

<!-- bilingual-en:start -->
## Which reference distribution is justified?
<!-- bilingual-en:end -->
*什么条件支持参考分布？*

<!-- bilingual-en:start -->
The familiar reference is an **asymptotic** [[卡方分布|chi-square distribution]], $LM\Rightarrow\chi_q^2$ under the null: the arrow denotes [[依分布收敛|convergence in distribution]] as the sample size grows. One concrete sufficient setting is a fixed number of retained and nonredundant added regressors, fixed uniformly bounded design rows whose normalized cross-product converges to a positive-definite matrix, and [[独立同分布|iid]] errors with mean zero, positive finite variance and a finite $(2+\delta)$ moment, meaning $E|u_i|^{2+\delta}<\infty$ for some $\delta>0$. These conditions ensure enough variation in each tested direction and a valid normal approximation. Random regressors or dependent observations need their corresponding exogeneity, moment and limit conditions; the algebraic $nR^2$ formula alone supplies none of them.
<!-- bilingual-en:end -->

常用参考是**渐近的**[[卡方分布]]：在零假设下，$LM\Rightarrow\chi_q^2$，箭头表示随样本量增大而[[依分布收敛]]。一组具体的充分条件是：保留项与非冗余新增项的数量固定；固定设计的各行一致有界，标准化交叉乘积矩阵收敛到正定矩阵；误差[[独立同分布]]、均值为零、方差有限且为正，并对某个 $\delta>0$ 有有限的 $2+\delta$ 阶矩，即 $E|u_i|^{2+\delta}<\infty$。这些条件保证各检验方向有足够变化，并能使用正态近似。随机回归量或相关观测需要相应的外生性、矩和极限条件；单凭 $nR^2$ 的代数公式不能获得这些保证。

<!-- bilingual-en:start -->
With fixed augmented regressors and independent $N(0,\sigma^2)$ errors, $\sigma^2>0$, a different statistic has the exact finite-sample [[F分布|F reference]] under $H_0$:
<!-- bilingual-en:end -->

若扩展回归量固定，误差独立且服从 $N(0,\sigma^2)$，其中 $\sigma^2>0$，则零假设下另一个统计量具有精确的有限样本 [[F分布|F 参考分布]]：

$$F=\frac{(RSS_R-RSS_U)/q}{RSS_U/(n-k-q)}\sim F_{q,n-k-q}.$$

<!-- bilingual-en:start -->
In the six-observation example, this is $F=12/(6/3)=6$ with degrees of freedom $(1,3)$. It is not $LM=4$. Indeed $LM=nqF/(n-k-q+qF)$, so finite-sample calibrations differ; $LM$ is bounded by $n$ and cannot have an exact, unbounded chi-square law. With [[异方差|heteroskedasticity]] or [[聚类标准误边界|cluster dependence]], use a score or [[Wald联合检验|joint Wald]] procedure whose covariance estimate and calibration are justified for that sampling structure.
<!-- bilingual-en:end -->

六观测例中，这给出 $F=12/(6/3)=6$，自由度为 $(1,3)$；它不是 $LM=4$。实际上 $LM=nqF/(n-k-q+qF)$，有限样本的校准因此不同；$LM$ 不超过 $n$，不可能精确服从没有上界的卡方分布。存在[[异方差]]或[[聚类标准误边界|簇内相关]]时，应使用协方差估计与参考校准确有依据的得分或 [[Wald联合检验|Wald 联合检验]]。

<!-- bilingual-en:start -->
## Why the proposed directions matter
<!-- bilingual-en:end -->
## 为什么它比通用探针更具体

<!-- bilingual-en:start -->
If the prespecified candidates are $x^2$ and an interaction $xz$, a properly calibrated rejection is evidence against omitting those tested directions within the maintained model. It does not identify an untested variable $w$, establish a causal mechanism, or prove that every remaining assumption is correct. Non-rejection does not prove exact zero coefficients. A [[残差图边界|residual plot]] can motivate later candidates, but selecting terms because they look promising in the same sample changes the testing procedure; the ordinary reference does not automatically account for that search. Record exploratory findings as exploratory or use a separately justified confirmation design.
<!-- bilingual-en:end -->

若事先指定的候选项是 $x^2$ 和交互项 $xz$，正确校准下的拒绝，是在其余模型假设下反对遗漏这些方向的证据。它不能定位未检验变量 $w$，不能确立因果机制，也不能证明其余所有假设正确。不拒绝则不能证明系数恰为零。[[残差图边界|残差图]]可以启发后续候选项，但因为同一样本里某些项“看起来有用”才挑选它们，会改变整个检验程序；普通参考分布不会自动校正这次搜索。应把探索结果明确记作探索，或使用另有依据的确认设计。

<!-- bilingual-en:start -->
## Continue
<!-- bilingual-en:end -->
## 继续

- [[RESET检验边界|Ramsey RESET]]：对照拟合值幂这种汇总型非线性探针。
- [[设定修复循环]]：把拒绝转化为可解释、可验证的修订。

## 来源与核验

<!-- bilingual-en:start -->
- [University of Houston, Financial Econometrics, Review & Lecture 6](https://bauer.uh.edu/rsusmel/4397/fec-6-a.pdf), current PDF pp. 27–29, slides 53–57: restricted-residual auxiliary regression, original regressors retained, and the common $nR^2$ construction.
- [Hansen, Econometrics (2010)](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf), §4.13, PDF pp. 68–70; §7.7, pp. 121–122: normal-regression projection independence and specified nonlinear additions. The effective-rank calculation, six-observation example and displayed LM–F identity are derived directly here.
- [[02_Economy/01_Econometrics/04_模型设定.md#3.2. 拟合不足的检验：LM 拉格朗日乘数检验|Course §3.2]] and [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#4.3. LM 检验|matrix-regression §4.3]] supply course scope and notation.
- Further reading: Wooldridge, *Introductory Econometrics*, 6th ed., Chapters 5 and 9. That edition's full text was not reopened for this verification.
<!-- bilingual-en:end -->

- 休斯顿大学讲义现版 PDF 第 27–29 页（幻灯片 53–57）支持受限残差、保留原回归量和常用 $nR^2$ 构造。
- Hansen §4.13、§7.7 支持正态回归的投影独立性和指定非线性候选项；有效秩、六观测例与 LM–F 恒等式在本卡直接推导。
- 上述永久课程 §3.2、矩阵表达 §4.3 保留课程范围和记号。Wooldridge 第六版列为延伸阅读，本次未重开该版全文。
