---
aliases:
- OLS consistency
- OLS consistency conditions
- OLS 一致性条件
student_os: knowledge-atom
atom_id: ECON-OLS-011
atom_type: proposition
status: source-checked
part_of:
- '[[OLS 线性回归.canvas|OLS 线性回归]]'
requires:
- '[[普通最小二乘]]'
- '[[依概率收敛]]'
- '[[连续映射定理]]'
contrasts_with:
- '[[零条件均值无偏性]]'
---

<!-- bilingual-en:start -->
# OLS consistency
<!-- bilingual-en:end -->

*OLS 一致性条件*

<!-- bilingual-en:start -->
[[普通最小二乘|Ordinary least squares (OLS)]] chooses coefficients to make the squared prediction errors in the observed sample as small as possible. **Consistency** asks a different question: as the sample grows under a specified sampling model, does the fitted coefficient become reliably close to the population coefficient we want? More rows help only when they bring information that allows the relevant averages to stabilize.
<!-- bilingual-en:end -->

[[普通最小二乘|普通最小二乘（OLS）]]选择系数，让已经观察到的样本中预测误差的平方和尽可能小。**一致性**问的是另一个问题：在指定抽样模型下，样本越来越大时，拟合系数能否可靠地接近想估计的总体系数？只有新增数据让相关平均量稳定下来，增加行数才有相应作用。

<!-- bilingual-en:start -->
For a fixed number of coefficients, write the target vector as $\beta$ and the estimate using $n$ observations as $\hat\beta_n$. [[估计量一致性|Consistency]] means $\hat\beta_n\xrightarrow p\beta$: for every fixed tolerance $\varepsilon>0$, the probability that the [[欧氏范数|distance]] $\|\hat\beta_n-\beta\|$ exceeds $\varepsilon$ tends to zero. This is a statement about sampling probabilities, not a promise that each newly observed estimate is closer than the preceding one.
<!-- bilingual-en:end -->

先固定系数个数，把目标向量记为 $\beta$，用 $n$ 个观察得到的估计记为 $\hat\beta_n$。[[估计量一致性|一致性]]表示 $\hat\beta_n\xrightarrow p\beta$：任意固定容差 $\varepsilon>0$ 下，[[欧氏范数|距离]] $\|\hat\beta_n-\beta\|$ 超过它的概率都趋于零。它谈的是抽样概率，不保证每次增加观察后，实现的估计值都比上一次更接近目标。

> [!example] A slope whose error really averages out · 斜率误差确实能平均掉
> <!-- bilingual-en:start -->
> Consider the deliberately simple model **without an intercept**, $Y_i=2x_i+u_i$. Each $x_i$ and $u_i$ independently takes $-1$ or $1$ with equal probability, and different observation pairs are independent. The slope target is $2$. Because $x_i^2=1$,
> $$\hat\beta_n=\frac{\sum_i x_iY_i}{\sum_i x_i^2}=2+\frac1n\sum_i x_iu_i.$$
> The products $x_iu_i$ are independent, have mean zero and variance one. Thus $E[\hat\beta_n]=2$ and $\operatorname{Var}(\hat\beta_n)=1/n$. [[Chebyshev不等式|Chebyshev's inequality]] gives $P(|\hat\beta_n-2|>\varepsilon)\le1/(n\varepsilon^2)\to0$. At $n=400$ and $\varepsilon=0.5$, this upper bound is $0.01$. The cancellation is probabilistic; no individual sample is required to have zero average error product.
> <!-- bilingual-en:end -->
>
> 故意采用一个**没有截距**的简单模型 $Y_i=2x_i+u_i$。每个 $x_i,u_i$ 都独立、等概率取 $-1$ 或 $1$，不同观察对也独立，目标斜率为 $2$。因为 $x_i^2=1$，斜率估计等于 $2$ 加上 $x_iu_i$ 的样本平均。各乘积独立、均值零、方差一，所以斜率估计均值为 $2$、方差为 $1/n$。[[Chebyshev不等式]]把它转成上面的超差概率界；$n=400$、容差 $0.5$ 时，上界为 $0.01$。这里依靠概率意义的平均，不要求每份样本的误差乘积平均恰好为零。

<!-- bilingual-en:start -->
## Separate the design from the error average
<!-- bilingual-en:end -->
## 自然解释与边界

<!-- bilingual-en:start -->
In the general model $Y_i=x_i^T\beta+u_i$, $Y_i$ is a scalar outcome, $x_i$ is a column of $k$ regressors, and $\beta$ contains the $k$ target coefficients. The superscript $T$ means [[矩阵转置|transpose]]. An intercept is included by making one coordinate of $x_i$ equal to one. The model error $u_i=Y_i-x_i^T\beta$ uses the population target; it is not the fitted [[回归残差|residual]] $Y_i-x_i^T\hat\beta_n$.
<!-- bilingual-en:end -->

一般模型为 $Y_i=x_i^T\beta+u_i$：$Y_i$ 是一个结果数值，$x_i$ 是含 $k$ 个解释项的列向量，$\beta$ 含相应的 $k$ 个目标系数，上标 $T$ 表示[[矩阵转置|转置]]。需要截距时，让 $x_i$ 的一个坐标恒等于一。模型误差 $u_i=Y_i-x_i^T\beta$ 使用总体目标，与拟合后[[回归残差|残差]] $Y_i-x_i^T\hat\beta_n$ 不同。

<!-- bilingual-en:start -->
Stack the row vectors $x_i^T$ into the design matrix $X$. On samples where its columns are [[满列秩与OLS唯一性|linearly independent]], substitution into the OLS formula gives
$$\hat\beta_n-\beta=Q_n^{-1}g_n,\qquad
Q_n=\frac{X^TX}{n}=\frac1n\sum_{i=1}^n x_ix_i^T,\qquad
g_n=\frac1n\sum_{i=1}^n x_iu_i.$$
Here $Q_n$ is a $k\times k$ matrix of average regressor products; $g_n$ is a $k$-vector of average regressor–error products. The first determines how well the design distinguishes coefficient directions. The second is the part of the model error that shifts the fitted coefficients.
<!-- bilingual-en:end -->

把各行 $x_i^T$ 叠成设计矩阵 $X$。在列[[满列秩与OLS唯一性|线性无关]]的样本上，将模型代入 OLS 公式就得到上式。$Q_n$ 是 $k\times k$ 的解释项乘积平均矩阵，$g_n$ 是 $k$ 维的解释项与误差乘积平均。前者决定设计能否分清不同系数方向，后者是推动系数偏离目标的误差部分。

> [!important] A sufficient large-sample route · 一条充分的大样本路径
> <!-- bilingual-en:start -->
> Keep $k$ fixed. If $Q_n\xrightarrow p Q$ for a fixed finite [[正定矩阵|positive-definite]] matrix $Q$, and $g_n\xrightarrow p0$, then $\hat\beta_n\xrightarrow p\beta$. A positive-definite limit is invertible. The [[连续映射定理|continuous mapping theorem]] applies to matrix inversion near $Q$ and to multiplication, giving $Q_n^{-1}g_n\xrightarrow p Q^{-1}0=0$.
>
> Finite samples can still be singular. These assumptions imply $P(Q_n\text{ is singular})\to0$; define the estimator to be any fixed vector, such as zero, on those samples. This convention makes it a defined random quantity for every $n$ and does not change the probability limit. The conditions above are sufficient, not a list of all possible ways OLS can be consistent.
> <!-- bilingual-en:end -->
>
> 固定 $k$。若 $Q_n\xrightarrow p Q$，其中 $Q$ 是固定、有限的[[正定矩阵]]，并且 $g_n\xrightarrow p0$，就有 $\hat\beta_n\xrightarrow p\beta$。正定极限可逆，[[连续映射定理]]允许在它附近取逆再相乘，得到 $Q_n^{-1}g_n\xrightarrow p Q^{-1}0=0$。
>
> 有限样本仍可能奇异，但这些假设使奇异概率趋于零。可在这些样本上把估计量规定为任意固定向量，例如零，让每个 $n$ 都有定义；这一约定不改变概率极限。上述条件是一条充分路径，并未列尽 OLS 一致的所有可能机制。

<!-- bilingual-en:start -->
## How iid sampling supplies those limits
<!-- bilingual-en:end -->

*iid 抽样怎样提供这些极限*

<!-- bilingual-en:start -->
A common sufficient setup has [[独立同分布|iid]] observation pairs $(x_i,Y_i)$, finite second moments $E\|x_i\|^2<\infty$ and $E[Y_i^2]<\infty$, and $Q=E[x_ix_i^T]$ positive definite. Also require **population orthogonality** $E[x_iu_i]=0$ for the target $\beta$. The second moments make every needed product integrable. The [[弱大数定律|weak law of large numbers]] then applies to each entry: $Q_n\xrightarrow p E[x_ix_i^T]$ and $g_n\xrightarrow p E[x_iu_i]=0$. Normal errors and constant error variance are not required for this consistency result.
<!-- bilingual-en:end -->

常用的充分设定是：观察对 $(x_i,Y_i)$ [[独立同分布|iid]]，二阶矩 $E\|x_i\|^2$、$E[Y_i^2]$ 有限，$Q=E[x_ix_i^T]$ 正定，并对目标 $\beta$ 要求**总体正交** $E[x_iu_i]=0$。二阶矩保证所需乘积可积，[[弱大数定律]]于是逐项给出 $Q_n$ 与 $g_n$ 的极限。本一致性结论不要求误差正态，也不要求误差方差恒定。

<!-- bilingual-en:start -->
Under these moments, $E[u_i\mid x_i]=0$ implies orthogonality by [[条件期望取出已知量|taking out the known regressor]] and [[全期望公式|total expectation]]. The reverse is generally false. If $\beta$ is defined as the population [[条件均值与线性投影|best linear predictor]], orthogonality is a property of that projection; consistency then targets that coefficient. It does not automatically make the whole conditional mean linear or make the coefficient a causal effect. For a separately specified structural coefficient, orthogonality is an assumption to justify.
<!-- bilingual-en:end -->

在这些矩条件下，$E[u_i\mid x_i]=0$ 可通过[[条件期望取出已知量|提出已知解释项]]及[[全期望公式]]推出总体正交，反过来一般不成立。若把 $\beta$ 定义为总体[[条件均值与线性投影|最佳线性预测]]系数，正交就是该投影的性质，一致性指向这个系数；它不会自动让整个条件均值成为线性函数，也不会把系数变成因果效应。若目标是另行指定的结构系数，总体正交则是需要论证的假设。

<!-- bilingual-en:start -->
The limiting design also matters. If two columns are identical, their separate coefficients cannot be identified from their sum. With dependent time-series or clustered observations, iid laws cannot simply be reused: one must establish the two displayed limits under the actual dependence. Heavy tails may invalidate the convenient moment assumptions. The sample identity $X^T\hat u=0$ from OLS optimization cannot replace $E[x_iu_i]=0$ or $g_n\xrightarrow p0$; see [[样本正交与总体外生性|sample orthogonality versus population restrictions]].
<!-- bilingual-en:end -->

极限设计同样重要。若两列相同，只能看到它们系数的和，无法分清各自系数。时间序列或聚类观察有依赖时，不能直接搬用 iid 大数定律，而要在真实依赖下证明上述两个极限；重尾也可能破坏方便使用的矩条件。OLS 优化给出的样本恒等式 $X^T\hat u=0$ 不能替代 $E[x_iu_i]=0$ 或 $g_n\xrightarrow p0$，见[[样本正交与总体外生性]]。

<!-- bilingual-en:start -->
## Unbiasedness and variance do not decide consistency by themselves
<!-- bilingual-en:end -->
## 最小对照

<!-- bilingual-en:start -->
[[零条件均值无偏性|Conditional unbiasedness]] is a finite-sample statement about averaging estimates under a fixed design. Consistency is a limit of error probabilities. For an intercept-only model $Y_i=\beta+Z$ with the **same** fair $Z\in\{-1,1\}$ in every row, OLS gives $\hat\beta_n=\beta+Z$. It is unbiased, but $P(|\hat\beta_n-\beta|>1/2)=1$ for every $n$. Here $Q_n=1$ while $g_n=Z$ never concentrates at zero: copying a shared disturbance does not average it away.
<!-- bilingual-en:end -->

[[零条件均值无偏性|条件无偏]]讨论固定设计下有限样本估计的平均；一致性讨论超差概率的极限。只有截距的模型若为 $Y_i=\beta+Z$，每一行共享**同一个**等概率取 $-1,1$ 的 $Z$，则 OLS 给出 $\hat\beta_n=\beta+Z$。它无偏，但任何 $n$ 下误差超过 $1/2$ 的概率都为一。此时 $Q_n=1$，而 $g_n=Z$ 始终不集中到零：重复记录同一个冲击不会把它平均掉。

> [!warning] Nonvanishing variance is not a general proof of failure · 方差不趋零不是一般的不一致性证明
> <!-- bilingual-en:start -->
> Consider zero-mean errors $Z_n$ taking $n$ and $-n$ with probability $1/(2n^2)$ each, and zero otherwise. Their variance is always one, yet for every fixed $\varepsilon>0$ and $n>\varepsilon$, $P(|Z_n|>\varepsilon)=1/n^2\to0$. Rare large errors keep the second moment large while the error converges in probability to zero. Thus an unbiased estimate can concentrate without its variance tending to zero. Vanishing mean squared error is sufficient for consistency; its failure is not sufficient for inconsistency. The shared-$Z$ example above proves failure through the actual error probability.
> <!-- bilingual-en:end -->
>
> 考虑均值为零的误差 $Z_n$：各以 $1/(2n^2)$ 的概率取 $n$、$-n$，其余时候为零。方差始终为一，但任意固定 $\varepsilon>0$ 及 $n>\varepsilon$ 下，$P(|Z_n|>\varepsilon)=1/n^2\to0$。罕见的巨大误差维持二阶矩，同时误差仍依概率趋零。因此无偏估计可以集中而方差不趋零；均方误差趋零足以推出一致，其不趋零却不足以反推不一致。上面的共享 $Z$ 例子通过真实超差概率证明失败。

<!-- bilingual-en:start -->
Conversely, adding $1/n$ to a consistent unbiased sample mean gives a finite-sample bias while preserving consistency; see [[一致不推出无偏|consistency without unbiasedness]]. Consistency also does not supply a convergence rate, an asymptotic normal distribution, or valid standard errors. Those require further conditions appropriate to the sampling model.
<!-- bilingual-en:end -->

反过来，在一致且无偏的样本均值上加 $1/n$，会产生有限样本偏差，却保持一致，见[[一致不推出无偏]]。一致性也不会自动提供收敛速度、渐近正态分布或有效标准误；这些需要与抽样模型相符的进一步条件。

<!-- bilingual-en:start -->
## Continue with the relevant question
<!-- bilingual-en:end -->
## 继续

- [[零条件均值无偏性|Finite-sample unbiasedness · 有限样本无偏性]]
- [[动态回归规格流程|Dynamic-regression specification · 动态回归规格流程]]

## 来源与核验

<!-- bilingual-en:start -->
- [Hansen, Econometrics, revision 10 January 2010, §§5.1–5.3, printed pp.65–70 / PDF pp.72–77](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf#page=72): finite-second-moment iid projection setup, entrywise laws of large numbers, continuous inversion and Theorem5.3.1. The general $Q_n,g_n$ condition is the same algebraic argument with the convergence hypotheses stated directly.
- [MIT15.450, OLS: Estimation and Standard Errors, PDF pp.2–3](https://ocw.mit.edu/courses/15-450-analytics-of-finance-fall-2010/2710a11d17cae67d02173ffb15859784_MIT15_450F10_rec10.pdf#page=2): model dimensions, population orthogonality and the OLS formula. Its informal orthogonality remark is supplemented here by explicit rank, moment and convergence conditions.
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#3.2. 大样本性质|Permanent course notes, §3.2]]: course notation and the large-sample topic. The slope, shared-shock and rare-error examples are calculated directly; their probability arguments distinguish consistency from moment convergence.
<!-- bilingual-en:end -->

- Hansen 2010稿第5.1–5.3节：核对 iid 二阶矩与投影设定、逐项大数定律、取逆的连续性及一致性定理；一般 $Q_n,g_n$ 条件直接使用同一代数推导。
- MIT15.450第2–3页：核对维数、总体正交与 OLS 公式；正文明确补足其简短说明未展开的秩、矩与收敛条件。
- 永久课程第3.2节保留符号与课程定位；斜率、共同冲击及罕见误差例子直接计算概率和矩，不用方差不收缩替代不一致性证明。
