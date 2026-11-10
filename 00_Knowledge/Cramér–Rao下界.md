---
student_os: knowledge-atom
atom_id: 6b0b2784-4c54-4860-816e-01b7dda7f4bb
status: source-checked
aliases:
  - Cramer–Rao bound
  - Cramér–Rao bound
  - Cramér–Rao inequality
  - CRLB
atom_type: theorem
---

<!-- bilingual-en:start -->
# Cramér–Rao lower bound
<!-- bilingual-en:end -->

*Cramér–Rao 下界*

<!-- bilingual-en:start -->
After choosing an unbiased estimation rule, we may still wonder whether some other rule could have a smaller variance. The Cramér–Rao bound gives a benchmark without listing every possible rule: in a suitable parametric model, the way the data distribution changes with the unknown parameter limits how concentrated an unbiased estimate can be.
<!-- bilingual-en:end -->

选定一个无偏估计规则之后，还可以问：是否存在另一个方差更小的规则？Cramér–Rao 下界不必逐一列举所有规则，就能提供一个基准：在合适的参数模型中，数据分布随未知参数变化的方式，限制了无偏估计值能够集中到什么程度。

> [!important] A lower bound under checked conditions · 核对条件后才能使用的下界
> <!-- bilingual-en:start -->
> For a scalar parameter $\theta$, an unbiased estimator $T$ of $\theta$ satisfies $\operatorname{Var}_\theta(T)\ge1/I(\theta)$ under the regularity and finite-moment conditions stated below. Here $I(\theta)$ is the Fisher information in the **whole observed sample**. A lower bound does not guarantee that any estimator attains it.
> <!-- bilingual-en:end -->
>
> 对标量参数 $\theta$，在下文的正则条件与有限矩条件下，其无偏估计量 $T$ 满足 $\operatorname{Var}_\theta(T)\ge1/I(\theta)$。$I(\theta)$ 是**整个观测样本**中的 Fisher 信息。存在下界，并不保证一定有估计量达到它。

^core

<!-- bilingual-en:start -->
## A benchmark for estimating a normal mean
<!-- bilingual-en:end -->

*先看正态均值的精度基准*

<!-- bilingual-en:start -->
Suppose $X_1,\ldots,X_n$ are [[独立同分布|independent and identically distributed]] as $N(\mu,\sigma^2)$, a [[正态分布|normal distribution]] with unknown mean $\mu$ and known variance $\sigma^2>0$. The average $\bar X=n^{-1}\sum_iX_i$ is unbiased and has variance $\sigma^2/n$. Could a complicated nonlinear unbiased rule have a smaller variance? The bound below answers no **within this normal model**. For example, if $\sigma^2=4$ and $n=4$, every admissible unbiased estimator has variance at least $1$, and the sample average attains $1$.
<!-- bilingual-en:end -->

设 $X_1,\ldots,X_n$ [[独立同分布]]于 $N(\mu,\sigma^2)$，即均值 $\mu$ 未知、方差 $\sigma^2>0$ 已知的[[正态分布]]。平均数 $\bar X=n^{-1}\sum_iX_i$ 无偏，方差为 $\sigma^2/n$。复杂的非线性无偏规则能否取得更小方差？下面的下界说明：**在这个正态模型中**不能。例如 $\sigma^2=4$、$n=4$ 时，每个符合条件的无偏估计量的方差都至少为 $1$，而样本平均数恰好达到 $1$。

<!-- bilingual-en:start -->
## Describe the model and how it changes
<!-- bilingual-en:end -->

*描述模型怎样随参数变化*

<!-- bilingual-en:start -->
Let $X$ now denote the whole observation, possibly a vector, and let $f_\theta(x)$ be its joint [[概率密度函数|density]] or [[概率质量函数|probability mass function]]. The scalar parameter $\theta$ ranges over an open interval, and the model specifies a distribution for each possible value. A [[统计量|statistic]] $T=T(X)$ is one fixed rule computed from the observations; it cannot secretly use the unknown $\theta$. Write $E_\theta$ and $\operatorname{Var}_\theta$ for expectation and variance under the distribution indexed by $\theta$.
<!-- bilingual-en:end -->

现在用 $X$ 表示整个观测，它可以是向量；$f_\theta(x)$ 表示其联合[[概率密度函数|密度]]或[[概率质量函数]]。标量参数 $\theta$ 在一个开区间内变化，模型为每个可能的参数值指定一种分布。[[统计量]] $T=T(X)$ 是从观测算出的同一个固定规则，不能暗中使用未知的 $\theta$。$E_\theta$ 和 $\operatorname{Var}_\theta$ 分别表示在参数值为 $\theta$ 的分布下求期望与方差。

<!-- bilingual-en:start -->
For a fixed possible observation $x$, differentiate the [[自然对数|natural logarithm]] of $f_\theta(x)$ with respect to the parameter, holding $x$ fixed. The resulting **[[得分函数|score]]** measures local relative sensitivity of the model density or mass to the parameter. Evaluating it at random data makes it a random variable. Its expected squared size is the **[[Fisher信息|Fisher information]]**:
<!-- bilingual-en:end -->

固定一个可能的观测 $x$，对 $f_\theta(x)$ 的[[自然对数]]关于参数求导，求导时保持 $x$ 不变。得到的 **[[得分函数|score（得分函数）]]**衡量模型密度或概率质量对参数的局部相对敏感程度。把随机数据代进去，它就成为随机变量；其平方的期望称为 **[[Fisher信息|Fisher 信息]]**：

$$
S_\theta(X)=\frac{\partial}{\partial\theta}\log f_\theta(X),\qquad I(\theta)=E_\theta[S_\theta(X)^2].
$$

<!-- bilingual-en:start -->
## The regularity conditions do real work
<!-- bilingual-en:end -->

*正则条件承担实际作用*

<!-- bilingual-en:start -->
One useful version assumes a common support $\mathcal S$: the set where $f_\theta(x)>0$ is independent of $\theta$, and $f_\theta(x)$ is differentiable in $\theta$ there. Require $0<I(\theta)<\infty$ and $E_\theta[T^2]<\infty$. Crucially, differentiation must be interchangeable with integration (or summation in a discrete model) both for total probability and for the mean of $T$:
<!-- bilingual-en:end -->

一个常用版本假设共同支撑 $\mathcal S$：$f_\theta(x)>0$ 的取值集合不随 $\theta$ 改变，且在该集合内 $f_\theta(x)$ 对 $\theta$ 可微。还要求 $0<I(\theta)<\infty$、$E_\theta[T^2]<\infty$。关键是对总概率与 $T$ 的期望，都能交换参数求导与积分（离散模型中为求和）的次序：

$$
\frac{d}{d\theta}\int_{\mathcal S} f_\theta(x)\,dx=\int_{\mathcal S}\partial_\theta f_\theta(x)\,dx=0,
\qquad\frac{d}{d\theta}E_\theta[T]=\int_{\mathcal S}T(x)\,\partial_\theta f_\theta(x)\,dx.
$$

<!-- bilingual-en:start -->
These are conditions to justify, not automatic consequences of writing a derivative. A standard sufficient route is to bound the relevant derivative integrands, locally in $\theta$, by integrable functions so that differentiation under the integral is valid. Common support and pointwise differentiability alone do not supply that justification. This is a usable sufficient setup, not a claim that every valid information inequality requires this exact presentation.
<!-- bilingual-en:end -->

这些等式需要论证，不能因为写下导数符号就默认成立。一个常用的充分论证方法，是在 $\theta$ 的局部范围内，用可积函数控制相关导数被积项，从而允许积分号下求导；共同支撑与逐点可微本身并不足以完成这一论证。这里给出的是可用的一组充分设定，不是在声称所有信息不等式都只能按这一形式建立。

<!-- bilingual-en:start -->
## The bound follows from one covariance identity
<!-- bilingual-en:end -->

*从一个协方差恒等式得到下界*

<!-- bilingual-en:start -->
Suppose the target is a differentiable function $g(\theta)$ and $T$ is [[估计量无偏性|unbiased]] for it throughout the parameter interval: $E_\theta[T]=g(\theta)$. The first interchange above gives $E_\theta[S_\theta]=0$. The second gives $E_\theta[TS_\theta]=g^{\prime}(\theta)$, so $\operatorname{Cov}_\theta(T,S_\theta)=g^{\prime}(\theta)$. Apply the [[期望的Cauchy–Schwarz不等式|Cauchy–Schwarz inequality for expectations]] to the centred variables:
<!-- bilingual-en:end -->

设目标是可微函数 $g(\theta)$，且 $T$ 在整个参数区间内都对它[[估计量无偏性|无偏]]：$E_\theta[T]=g(\theta)$。上面第一个交换等式给出 $E_\theta[S_\theta]=0$；第二个给出 $E_\theta[TS_\theta]=g^{\prime}(\theta)$，因此 $\operatorname{Cov}_\theta(T,S_\theta)=g^{\prime}(\theta)$。对中心化后的变量应用[[期望的Cauchy–Schwarz不等式]]，得到

$$
[g^{\prime}(\theta)]^2\le\operatorname{Var}_\theta(T)\operatorname{Var}_\theta(S_\theta)=\operatorname{Var}_\theta(T)I(\theta),\qquad \boxed{\operatorname{Var}_\theta(T)\ge\frac{[g^{\prime}(\theta)]^2}{I(\theta)}}.
$$

<!-- bilingual-en:start -->
For the parameter itself, $g(\theta)=\theta$ and $g^{\prime}(\theta)=1$, recovering $1/I(\theta)$. For $n$ independent identically distributed observations in the regular model, the sample score is a sum of independent mean-zero scores, so $I_n(\theta)=nI_1(\theta)$. If the information was already calculated for the full sample, do not multiply it by $n$ again.
<!-- bilingual-en:end -->

若目标就是参数本身，$g(\theta)=\theta$、$g^{\prime}(\theta)=1$，便恢复 $1/I(\theta)$。对正则模型中 $n$ 个独立同分布观测，样本得分是独立零均值得分的和，所以 $I_n(\theta)=nI_1(\theta)$。若已经计算了整个样本的信息量，就不能再乘一次 $n$。

<!-- bilingual-en:start -->
## Compute and attain the normal-mean bound
<!-- bilingual-en:end -->

*计算并达到正态均值下界*

<!-- bilingual-en:start -->
In the opening normal example, the joint log density is a term independent of $\mu$ minus $\sum_i(X_i-\mu)^2/(2\sigma^2)$. Differentiating with respect to $\mu$ gives
<!-- bilingual-en:end -->

开头正态例子的联合对数密度，等于一个不依赖 $\mu$ 的项减去 $\sum_i(X_i-\mu)^2/(2\sigma^2)$。对 $\mu$ 求导得到

$$
S_\mu=\frac{\sum_i(X_i-\mu)}{\sigma^2},\qquad I_n(\mu)=\operatorname{Var}(S_\mu)=\frac{n}{\sigma^2},\qquad \operatorname{Var}(T)\ge\frac{\sigma^2}{n}.
$$

<!-- bilingual-en:start -->
Normal densities have a common support, and their parameter derivatives are polynomials times Gaussian densities; the needed exchanges hold here. The sample average has exactly this variance, so it attains the bound for every $\mu$. Equality in the proof requires $T-g(\theta)=[g^{\prime}(\theta)/I(\theta)]S_\theta$ almost surely. For the normal mean this reads $\bar X-\mu=(\sigma^2/n)S_\mu$, which is true. In another model the equality equation may not produce any statistic independent of the unknown parameter. Failing to attain the bound also does not by itself prove that a better unbiased estimator exists.
<!-- bilingual-en:end -->

正态密度具有共同支撑，参数导数是多项式乘 Gaussian 密度，此处所需交换成立。样本平均数的方差恰好等于这个界，因此对每个 $\mu$ 都达到下界。证明中取等号要求 $T-g(\theta)=[g^{\prime}(\theta)/I(\theta)]S_\theta$ 几乎必然成立；正态均值的情形就是 $\bar X-\mu=(\sigma^2/n)S_\mu$，确实成立。在其他模型中，等号方程可能无法给出一个不使用未知参数的统计量。没有达到下界，本身也不能证明一定存在更好的无偏估计量。

> [!warning] A moving endpoint breaks the shortcut · 取值端点移动时不能机械套用
> <!-- bilingual-en:start -->
> Let one observation $X$ be [[连续均匀分布|uniform]] on $(0,\theta)$, with $\theta>0$. Then $T=2X$ is unbiased, and $\operatorname{Var}(T)=4(\theta^2/12)=\theta^2/3$. Differentiating only the interior formula $\log f_\theta(x)=-\log\theta$ gives $S_\theta=-1/\theta$ and a naive “bound” $\theta^2$, which the valid estimator beats. The theorem has not failed: the support moves with $\theta$, and this score has nonzero mean. Indeed $d[\int_0^\theta(1/\theta)\,dx]/d\theta=0$, whereas integrating only the interior derivative gives $\int_0^\theta(-1/\theta^2)\,dx=-1/\theta$. The omitted moving-endpoint term is essential.
> <!-- bilingual-en:end -->
>
> 设只有一个观测 $X$，在 $(0,\theta)$ 上[[连续均匀分布|均匀分布]]，其中 $\theta>0$。那么 $T=2X$ 无偏，且 $\operatorname{Var}(T)=4(\theta^2/12)=\theta^2/3$。如果只对区间内部公式 $\log f_\theta(x)=-\log\theta$ 求导，会得到 $S_\theta=-1/\theta$ 和伪“下界” $\theta^2$，却被这个合法估计量突破。这不是定理错误：支撑随 $\theta$ 移动，且这个得分的均值不为零。事实上 $d[\int_0^\theta(1/\theta)\,dx]/d\theta=0$，而只积分内部导数得到 $\int_0^\theta(-1/\theta^2)\,dx=-1/\theta$；被漏掉的移动端点项不可省略。

<!-- bilingual-en:start -->
This is a finite-sample variance bound within a specified regular model, not the linear-estimator comparison in the [[经典 Gauss–Markov 定理|classical Gauss–Markov theorem]]. It does not directly rank biased estimators by [[均方误差|mean squared error]]. With several unknown parameters, including nuisance parameters (unknown model quantities other than the target), the appropriate bound generally involves the inverse **information matrix**; treating one diagonal entry as if every other parameter were known can give the wrong bound. The scalar result above keeps that distinction explicit.
<!-- bilingual-en:end -->

这是指定正则模型内的有限样本方差界，不是[[经典 Gauss–Markov 定理]]中的线性估计量比较，也不直接按[[均方误差]]给有偏估计量排序。若有多个未知参数，包括目标之外仍未知的模型量（干扰参数），合适的下界通常需要**信息矩阵的逆**；把某个对角元素当成“其余参数全部已知”时的标量信息，可能给出错误下界。上面的标量结果须与这种情形区分。

## 来源与核验

<!-- bilingual-en:start -->
- Jun Shao, [Stat 610 Lecture 3, Theorem 7.3.9, PDF pp. 9–12 and 15](https://pages.stat.wisc.edu/~shao/stat610/stat610-03.pdf#page=9): supports the two differentiation exchanges, covariance proof, target derivative, sample-information additivity, and normal-mean attainment.
- Hyun Min Kang, [Biostatistics 602 Lecture 12, slides 12–14 / presentation PDF pp. 33–45](https://genome.sph.umich.edu/w/images/5/58/Bios602-wi13-lec12-presentation.pdf#page=33): supports the moving-support failure for a uniform model and the equality/attainability distinction. The displayed one-observation example is the directly checked $n=1$ case.
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 1 - Foundations.pdf#page=58|EC400 PSI Lecture 1, slide 58]] supplies the course-level distinction between BLUE and a regular parametric information bound; the detailed conditions and proof are supplied by the references above.
<!-- bilingual-en:end -->

- Shao 的 Stat 610 Lecture 3（PDF pp. 9–12、15）核对两种求导交换、协方差证明、目标导数、样本信息可加性与正态均值达到下界。
- Kang 的 Biostatistics 602 Lecture 12（slides 12–14 / 动画版 PDF pp. 33–45）核对均匀模型支撑移动导致的条件失效，以及等号和可达到性的区别；正文单观测算例取 $n=1$ 并直接核算。
- [[01_Math/08_EC400_MathsCamp/2026_Course_Materials/04_PSI/Lectures/Lecture 1 - Foundations.pdf#page=58|EC400 PSI Lecture 1，slide 58]]提供课程中 BLUE 与正则参数模型信息界的区别；详细条件和证明由上面的权威讲义补充。
