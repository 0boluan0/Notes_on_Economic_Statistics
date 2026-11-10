---
aliases:
- Pooled two-sample Hotelling T-squared test
- 两独立样本 Hotelling T²
- 共同协方差的多元两样本均值检验
student_os: knowledge-atom
atom_id: STAT-HOT-007
atom_set: hotelling-mean-inference
atom_type: theorem-procedure
status: source-checked
requires:
- '[[样本协方差Wishart律]]'
- '[[正态均值协方差独立]]'
- '[[Gaussian仿射闭包]]'
- '[[Wishart抽样假设]]'
- '[[Wishart秩与可逆性]]'
part_of:
- '[[Hotelling T² 与多元均值推断.canvas]]'
related:
- '[[单样本Hotelling T²]]'
- '[[配对Hotelling T²]]'
- '[[线性约束Hotelling T²]]'
---

<!-- bilingual-en:start -->
# Pooled two-sample Hotelling T-squared
<!-- bilingual-en:end -->

*共同协方差下的两独立样本 Hotelling T²*

<!-- bilingual-en:start -->
Suppose two factories make the same product, and each sampled item is measured for length and width. We want to compare the factories' **population mean vectors**, allowing the measurements on one item to be correlated. Pooled Hotelling's $T^2$ compares the difference between the two sample means with its estimated sampling uncertainty. “Pooled” means that the two groups contribute their **within-group** variation to estimate a common covariance matrix.
<!-- bilingual-en:end -->

假设两家工厂生产同一种产品，每件抽样产品都测量长度与宽度。我们想比较两家工厂的**总体均值向量**，同时允许同一件产品的两项测量相关。Pooled Hotelling $T^2$ 把两组样本均值之差与它的估计抽样误差比较。“Pooled”指合并两组各自的**组内**变动，用来估计共同协方差矩阵。

<!-- bilingual-en:start -->
Let $X_{gi}\in\mathbb R^p$ be observation $i$ in group $g$, where $g=1,2$, $p\ge1$, and each group has $n_g\ge2$ observations. Write $\bar X_g$ for its [[样本均值向量|sample mean]] and $S_g$ for its [[样本协方差矩阵|sample covariance]], calculated around its own group mean with divisor $n_g-1$. For the exact test, assume observations are [[独立同分布|iid within each group]], the two entire samples are independent, and
$$X_{1i}\sim N_p(\mu_1,\Sigma),\qquad X_{2i}\sim N_p(\mu_2,\Sigma),\qquad \Sigma\succ0.$$
The [[联合Gaussian|joint Gaussian]] model concerns the complete vector. The common [[正定矩阵|positive-definite covariance]] gives positive variance to every nonzero direction. Neither the population means nor the observed sample covariance matrices have to be equal.
<!-- bilingual-en:end -->

用 $X_{gi}\in\mathbb R^p$ 表示第 $g$ 组的第 $i$ 个观测，$g=1,2$、$p\ge1$，每组有 $n_g\ge2$ 个观测。$\bar X_g$ 是该组[[样本均值向量]]，$S_g$ 是围绕本组均值、用分母 $n_g-1$ 计算的[[样本协方差矩阵]]。精确检验要求各组内部[[独立同分布|iid]]、两整组样本彼此独立，并满足上述[[联合Gaussian|联合 Gaussian]]模型。共同协方差[[正定矩阵|正定]]意味着每个非零方向都有正方差；这不要求两总体均值相同，也不要求算出的两张样本协方差恰好相同。

<!-- bilingual-en:start -->
## Pool variation around each group's own mean
<!-- bilingual-en:end -->

*先围绕各自均值，再合并变动*

<!-- bilingual-en:start -->
The matrix $(n_g-1)S_g$ is group $g$'s sum of centered outer products. Adding those sums and dividing by their combined residual degrees of freedom gives
$$m=(n_1-1)+(n_2-1),\qquad S_p=\frac{(n_1-1)S_1+(n_2-1)S_2}{m}.$$
Each group loses one residual degree of freedom by estimating its own mean. Thus the weights are $n_1-1$ and $n_2-1$, not necessarily one half each. Computing one covariance around the grand mean after stacking all observations would also include separation between the group means; that is a different quantity.
<!-- bilingual-en:end -->

$(n_g-1)S_g$ 是第 $g$ 组的中心化外积之和。先相加，再除以合计的残差自由度，就得到 $S_p$。每组估计本组均值各消耗一个残差自由度，所以权重为 $n_1-1$ 与 $n_2-1$，不一定各占一半。若把所有观测直接堆起来、围绕总均值算一张协方差，还会混入组均值之间的分离，得到的是另一个量。

<!-- bilingual-en:start -->
Let $\delta=\mu_1-\mu_2$ be the population difference, and let $\delta_0$ be a fixed target. Equality of mean vectors corresponds to $\delta_0=0$. Write $d=\bar X_1-\bar X_2-\delta_0$ for the estimated departure and $h=1/n_1+1/n_2$. Independence between groups makes their [[样本均值协方差|mean covariances]] add:
$$\operatorname{Cov}(\bar X_1-\bar X_2)=h\Sigma.$$
We estimate this covariance by $hS_p$. When $S_p$ is invertible, the statistic is
$$T^2=d^T(hS_p)^{-1}d=\frac{n_1n_2}{n_1+n_2}\,d^TS_p^{-1}d.$$
It is a [[马氏距离|squared Mahalanobis departure]] on the uncertainty scale of the difference of means. The factor $h$ is essential: the covariance of an individual observation is not the covariance of this difference.
<!-- bilingual-en:end -->

令 $\delta=\mu_1-\mu_2$ 为总体均值之差，$\delta_0$ 为固定检验目标；比较均值是否相等时取 $\delta_0=0$。记 $d=\bar X_1-\bar X_2-\delta_0$，$h=1/n_1+1/n_2$。两组独立，因此[[样本均值协方差|均值协方差]]相加，差向量的协方差为 $h\Sigma$，用 $hS_p$ 估计。上式的 $T^2$ 是在这个误差尺度上的[[马氏距离|平方 Mahalanobis 偏离]]。$h$ 不能省略：单个观测的协方差与两个均值之差的协方差并不是同一个量。

> [!important] The exact F calibration · 精确 F 校准
> <!-- bilingual-en:start -->
> Under $H_0:\delta=\delta_0$ and the stated Gaussian sampling model, if $m\ge p$,
> $$F_{\mathrm{obs}}=\frac{m-p+1}{pm}T^2\sim F_{p,m-p+1}.$$
> The [[F分布|F distribution]] therefore has denominator degrees of freedom $m-p+1=n_1+n_2-p-1$. Choose $0<\alpha<1$ before inspecting the test results, and reject when $F_{\mathrm{obs}}$ exceeds the $(1-\alpha)$ [[总体分位数|quantile]] of this F law. A rejection gives evidence against the whole target difference; it does not establish that every coordinate differs.
> <!-- bilingual-en:end -->
>
> 在原假设与上述 Gaussian 抽样模型下，若 $m\ge p$，转换后服从 $F_{p,m-p+1}$。[[F分布|F 分布]]的分母自由度是 $m-p+1=n_1+n_2-p-1$。查看结果前选定 $0<\alpha<1$；超过该分布的 $1-\alpha$ [[总体分位数|分位数]]就拒绝。这提供反对整个目标差向量的证据，不表示每个坐标都不同。

> [!example] Two groups of six · 两组各六个观测
> <!-- bilingual-en:start -->
> Suppose $p=2$, $n_1=n_2=6$, $\bar x_1=(5,4)^T$, $\bar x_2=(2,2)^T$, and the target difference is zero. If
> $$S_1=\operatorname{diag}(1,2),\quad S_2=\operatorname{diag}(3,6),$$
> then $m=10$, $S_p=\operatorname{diag}(2,4)$, $h=1/3$, and $d=(3,2)^T$. Consequently,
> $$T^2=3\left(\frac{3^2}{2}+\frac{2^2}{4}\right)=16.5,\qquad F_{\mathrm{obs}}=\frac9{20}(16.5)=7.425.$$
> Since $F_{2,9}(0.95)\approx4.256$, the 5% test rejects equality under the model. Different realized $S_1,S_2$ do not themselves contradict the assumption of a common population $\Sigma$.
> <!-- bilingual-en:end -->
>
> 此时 $m=10$，合并协方差为 $\operatorname{diag}(2,4)$，均值差为 $(3,2)^T$，误差尺度因子 $h=1/3$。算得 $T^2=16.5$、$F_{\mathrm{obs}}=7.425$，超过 $F_{2,9}$ 的 95% 分位数约 $4.256$，因此在模型下按 5% 水平拒绝均值相等。两张实现的样本协方差不同，本身并不否定共同总体协方差假设。

<!-- bilingual-en:start -->
## What makes pooling exact, and where it stops
<!-- bilingual-en:end -->

*精确性从哪里来，适用范围到哪里*

<!-- bilingual-en:start -->
Each group's centered scatter has the [[样本协方差Wishart律|Wishart sampling law]]. The independent scatters share the same scale $\Sigma$, so their sum $mS_p$ has [[Wishart分布|Wishart distribution]] $W_p(m,\Sigma)$: in its Gaussian-squares construction, one simply joins the two independent collections. [[正态均值协方差独立|Normal-sample mean–covariance independence]] also makes $d$ independent of $S_p$. Under the null, $d/\sqrt h$ is $N_p(0,\Sigma)$. This independent Gaussian/Wishart combination gives the displayed F conversion. Merely averaging two covariance estimates would not establish these distributional facts.
<!-- bilingual-en:end -->

各组中心化散布服从[[样本协方差Wishart律|Wishart 抽样律]]。两个独立散布具有相同尺度 $\Sigma$，所以相加后 $mS_p\sim W_p(m,\Sigma)$；按[[Wishart分布|Wishart 的 Gaussian 平方构造]]，这就是把两组独立贡献接在一起。[[正态均值协方差独立|正态样本的均值—协方差独立性]]还保证 $d$ 与 $S_p$ 独立；原假设下 $d/\sqrt h\sim N_p(0,\Sigma)$。这套独立 Gaussian/Wishart 组合才给出精确 F 转换，单纯把两张协方差取平均不足以推出它。

<!-- bilingual-en:start -->
The [[Wishart秩与可逆性|Wishart rank condition]] is $m\ge p$, equivalently $n_1+n_2>p+1$. Individual $S_1$ or $S_2$ may be singular while their pooled matrix is invertible. At $m=p$ the theorem still has one denominator degree of freedom, although covariance estimation can be unstable. A realized rounded dataset must still be checked for rank; a pseudoinverse does not inherit the same theorem. When $p=1$, $T^2$ reduces to the square of the common-variance two-sample [[Student t分布|Student t]] statistic with $m$ degrees of freedom.
<!-- bilingual-en:end -->

[[Wishart秩与可逆性|Wishart 秩条件]]为 $m\ge p$，等价于 $n_1+n_2>p+1$。某一组的 $S_g$ 可以奇异，但合并矩阵仍可逆。$m=p$ 时定理仍成立，分母自由度为一，但协方差估计可能很不稳定。具体舍入数据仍要检查秩；换成伪逆不会继承同一定理。$p=1$ 时，$T^2$ 化为共同方差两样本 [[Student t分布|Student t]] 统计量的平方，自由度为 $m$。

<!-- bilingual-en:start -->
If the population covariances differ, the true difference covariance is $\Sigma_1/n_1+\Sigma_2/n_2$; the pooled exact F argument is no longer available. Equal group sizes do not restore its assumptions. For [[配对Hotelling T²|paired observations]], dependence within pairs changes the difference covariance, so use the paired-difference model instead. Non-Gaussian sampling or cross-group dependence also requires a separate calibration; large samples alone do not make this finite-sample F identity applicable.
<!-- bilingual-en:end -->

若总体协方差不同，均值差的真实协方差是 $\Sigma_1/n_1+\Sigma_2/n_2$，不能再引用这套 pooled 精确 F 论证；两组样本量相等也不会恢复其前提。对于[[配对Hotelling T²|配对观测]]，配对内部依赖会改变差的协方差，应使用差向量模型。非 Gaussian 抽样或跨组依赖同样需要另行校准；样本量大本身不会使这里的有限样本 F 等式适用。

## 来源与核验

<!-- bilingual-en:start -->
- [Penn State STAT 505, Lesson 7, §§7.1.12–7.1.15 and §7.2.7](https://online.stat.psu.edu/stat505/Lesson07): independent two-group design, common covariance, within-group pooling, exact F formula and the unequal-covariance distinction. The displayed formula and code give $n_1+n_2-p-1$; the following prose drops the final minus one, which is not used here.
- [Stanford STATS305C, Hotelling T2, “Two-sample tests: model,” “Hotelling's T² test,” and “Computing the test statistic”](https://web.stanford.edu/class/stats305c/lectures/Hotelling_T2.html): pooled covariance and F scale. The Gaussian/Wishart argument above uses correctly normalized $d/\sqrt h$ and $mS_p$; inconsistent intermediate scaling in the source's later distribution section is not used.
- [[01_Math/04_多元统计分析/06_比较多个均值向量comparisons of multivariate mean vectors.md#1.4. 两个独立总体均值向量比较|Course §1.4]]: course notation, rank threshold and unequal-covariance boundary. The numerical example is computed directly.
<!-- bilingual-en:end -->

- Penn State 核对独立两组设计、共同协方差、组内合并、F 公式与不等协方差区别；公式和代码中的分母自由度为 $n_1+n_2-p-1$，相邻文字漏掉末尾减一，本卡未沿用。
- Stanford 核对 pooled 协方差与 F 缩放；本卡使用正确归一化的 $d/\sqrt h$ 与 $mS_p$，不沿用原源后续分布段的中间缩放错误。
- 课程第1.4节核对记号、秩门槛和不等协方差边界；数值例子直接计算。
