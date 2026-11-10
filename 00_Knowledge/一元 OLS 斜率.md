---
aliases:
  - Simple OLS slope
  - 一元回归斜率公式
student_os: knowledge-atom
atom_id: ECON-OLS-007
atom_type: proposition
status: source-checked
part_of:
  - "[[OLS 线性回归.canvas|OLS 线性回归]]"
requires:
  - "[[OLS正规方程]]"
contrasts_with:
  - "[[多元回归系数]]"
---

<!-- bilingual-en:start -->
# The simple OLS slope
<!-- bilingual-en:end -->

一元 OLS 斜率

<!-- bilingual-en:start -->
A slope says how much a fitted straight line rises when its horizontal coordinate increases by one unit. In a regression of consumption on income, the slope measures the change in **fitted consumption** per unit of income. [[普通最小二乘|Ordinary least squares (OLS)]] chooses the line whose squared vertical prediction errors have the smallest sum. The slope is a description of that fitted line; interpreting it as the effect of changing someone's income needs an additional argument.
<!-- bilingual-en:end -->

斜率表示横坐标增加一个单位时，拟合直线上升多少。在消费对收入的回归中，它衡量每单位收入变化对应的**拟合消费**变化。[[普通最小二乘|普通最小二乘法（OLS）]]选择使纵向预测误差的平方和最小的直线。斜率首先描述这条拟合线；要把它解释成改变某个人收入所产生的效果，还需要额外论证。

<!-- bilingual-en:start -->
## From a fitted line to the covariance ratio
<!-- bilingual-en:end -->
## 自然解释与边界

<!-- bilingual-en:start -->
Suppose we have $n\ge2$ finite real observations $(x_i,y_i)$, where $i$ labels an observation. Fit $\hat y_i=\hat a+\hat b x_i$: $\hat a$ is the intercept, $\hat b$ the slope, and a hat marks an estimated quantity. Write $\bar x=n^{-1}\sum_i x_i$ and $\bar y=n^{-1}\sum_i y_i$ for the two sample means. For a given slope $b$, the [[OLS正规方程|OLS normal equations]] make the residuals sum to zero, so the best intercept is $a=\bar y-b\bar x$. Substituting it leaves the one-variable problem
<!-- bilingual-en:end -->

设有 $n\ge2$ 个有限实数观测 $(x_i,y_i)$，$i$ 是观测编号。拟合 $\hat y_i=\hat a+\hat b x_i$：$\hat a$ 是截距，$\hat b$ 是斜率，帽号表示估计量。记 $\bar x=n^{-1}\sum_i x_i$、$\bar y=n^{-1}\sum_i y_i$ 为两个样本均值。对给定斜率 $b$，[[OLS正规方程]]要求残差之和为零，因此最优截距为 $a=\bar y-b\bar x$。代入后只需解

$$
\min_b\sum_{i=1}^n\big[(y_i-\bar y)-b(x_i-\bar x)\big]^2.
$$

<!-- bilingual-en:start -->
Let $S_{xx}=\sum_i(x_i-\bar x)^2$ and $S_{xy}=\sum_i(x_i-\bar x)(y_i-\bar y)$. Expanding the square gives a constant minus $2bS_{xy}$ plus $b^2S_{xx}$. When $S_{xx}>0$, completing the square shows that its unique minimum occurs at $b=S_{xy}/S_{xx}$.
<!-- bilingual-en:end -->

令 $S_{xx}=\sum_i(x_i-\bar x)^2$、$S_{xy}=\sum_i(x_i-\bar x)(y_i-\bar y)$。展开平方，目标函数是一个常数减去 $2bS_{xy}$，再加上 $b^2S_{xx}$。当 $S_{xx}>0$ 时，配方可知唯一最小值出现在 $b=S_{xy}/S_{xx}$。

> [!important] The slope with an intercept / 含截距的斜率
> <!-- bilingual-en:start -->
> If $x$ varies in the sample, the unique fitted coefficients are
> <!-- bilingual-en:end -->
> 若 $x$ 在样本中有变化，唯一的拟合系数为
>
> $$\hat b=\frac{S_{xy}}{S_{xx}}=\frac{s_{xy}}{s_x^2},\qquad \hat a=\bar y-\hat b\bar x.$$
>
> <!-- bilingual-en:start -->
> Here $s_{xy}=S_{xy}/(n-1)$ is the sample [[协方差|covariance]], measuring signed co-movement; $s_x^2=S_{xx}/(n-1)$ is the sample [[方差|variance]], measuring the spread of $x$. Their common divisor cancels. Using $n$ in **both** definitions gives the same ratio; mixing divisors does not.
> <!-- bilingual-en:end -->
> 其中 $s_{xy}=S_{xy}/(n-1)$ 是衡量同向或反向变化的样本[[协方差]]，$s_x^2=S_{xx}/(n-1)$ 是衡量 $x$ 分散程度的样本[[方差]]。共同的分母会约掉；**两者都**用 $n$ 作分母仍得到同一比值，混用分母则不行。

<!-- bilingual-en:start -->
The numerator records whether above-average $x$ tends to accompany above-average $y$. Dividing by the variation in $x$ converts that co-movement into units of $y$ per unit of $x$. This finite-data identity needs no normality, independence, or random-sampling assumption. Such assumptions become relevant when the fitted coefficient is used to estimate a population quantity or to construct a standard error.
<!-- bilingual-en:end -->

分子记录高于均值的 $x$ 是否倾向于伴随高于均值的 $y$；除以 $x$ 的变动，就把这种共同变化换算成“每单位 $x$ 对应多少单位 $y$”。这条有限数据恒等式不需要正态、独立或随机抽样假设。要用拟合系数估计总体量或构造标准误时，才需要考虑相应统计假设。

<!-- bilingual-en:start -->
## A line you can calculate by hand
<!-- bilingual-en:end -->
## 最小例子

> [!example] Three observations / 三个观测
> <!-- bilingual-en:start -->
> Take $(x,y)=(1,2),(2,2),(3,5)$. Then $\bar x=2$, $\bar y=3$,
> <!-- bilingual-en:end -->
> 取 $(x,y)=(1,2),(2,2),(3,5)$。于是 $\bar x=2$、$\bar y=3$，
>
> $$S_{xy}=(-1)(-1)+0(-1)+1(2)=3,\qquad S_{xx}=1+0+1=2.$$
>
> <!-- bilingual-en:start -->
> Thus $\hat b=1.5$ and $\hat a=0$. The fitted values are $(1.5,3,4.5)$ and the [[回归残差|residuals]], observed minus fitted values, are $(0.5,-1,0.5)$. The observations need not lie on the fitted line: increasing $x$ by one changes the **line's prediction** by $1.5$, not every observed response by $1.5$.
> <!-- bilingual-en:end -->
> 因而 $\hat b=1.5$、$\hat a=0$。拟合值为 $(1.5,3,4.5)$，观测值减拟合值得到[[回归残差]] $(0.5,-1,0.5)$。观测不必落在拟合线上：$x$ 增加一单位，**直线的预测值**增加 $1.5$，并不是每个实际结果都增加 $1.5$。

<!-- bilingual-en:start -->
In the consumption example, a slope of $0.8$ predicts $8$ additional units of consumption along the fitted line for $10$ additional units of income. Calling this an estimate of a structural marginal propensity to consume requires a suitable model and [[条件均值因果边界|causal identification]]. The raw slope does not automatically adjust for household size, wealth, or other omitted factors.
<!-- bilingual-en:end -->

在消费例子中，斜率为 $0.8$ 意味着沿拟合线，收入增加 $10$ 单位对应预测消费增加 $8$ 单位。若要把它称为结构性边际消费倾向的估计，还需要适当模型和[[条件均值因果边界|因果识别]]。原始斜率不会自动调整家庭规模、财富或其他遗漏因素。

> [!warning] Check the intercept and the denominator / 检查截距与分母
> <!-- bilingual-en:start -->
> If every $x_i=c$, then $S_{xx}=0$. The data determine the best constant fitted value $a+bc=\bar y$, but cannot distinguish infinitely many pairs $(a,b)$. If the intercept is deliberately omitted instead, the slope is $\sum_i x_i y_i/\sum_i x_i^2$ when the denominator is positive; this is a different fitting problem, so the centered covariance ratio generally does not apply.
> <!-- bilingual-en:end -->
> 若所有 $x_i=c$，则 $S_{xx}=0$。数据能确定最优常数拟合值 $a+bc=\bar y$，却不能区分无穷多组 $(a,b)$。若有意删去截距，分母为正时斜率是 $\sum_i x_i y_i/\sum_i x_i^2$；这是另一个拟合问题，一般不能再用中心化后的协方差之比。

<!-- bilingual-en:start -->
A zero slope says only that the centered cross-product is zero. It does not rule out a curved relationship: for $x=(-1,0,1)$ and $y=x^2$, the fitted slope is zero although $y$ is completely determined by $x$.
<!-- bilingual-en:end -->

斜率为零只表示中心化乘积之和为零，并不排除曲线关系。例如 $x=(-1,0,1)$、$y=x^2$ 的拟合斜率为零，但 $y$ 完全由 $x$ 决定。

<!-- bilingual-en:start -->
## Adding controls
<!-- bilingual-en:end -->
## 继续

<!-- bilingual-en:start -->
Adding controls changes which variation the slope compares. [[多元回归系数|Multiple-regression coefficients]] explain the resulting held-fixed comparison. The [[FWL残差化定理|Frisch–Waugh–Lovell theorem]] recovers a simple-slope calculation after removing the same controls' linear predictions from both variables.
<!-- bilingual-en:end -->

加入控制变量后，斜率比较的变动会改变。[[多元回归系数]]解释“保持其他变量不变”的读法；[[FWL残差化定理]]则说明，先从两个变量中移除同一组控制变量的线性预测，就能重新用一元斜率计算。

## 来源与核验

<!-- bilingual-en:start -->
- Hansen, [*Econometrics* (2010)](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf), §2.10 (PDF pp. 26–27) and §3.3 (PDF pp. 36–38): the covariance/variance representation and sample normal equations. The centered finite-sample derivation and numerical examples above are also checked directly.
- [[02_Economy/01_Econometrics/02_一元线性回归.md#3.3. $\hat\beta_0$ 和 $\hat\beta_1$ 的解法|Local course: solving the simple-regression coefficients]]: the course notation and sample covariance ratio; the consumption application follows the course's interpretation context.
<!-- bilingual-en:end -->

- Hansen，[《Econometrics》（2010）](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf)，§2.10（PDF 第 26–27 页）及 §3.3（PDF 第 36–38 页）：协方差与方差的表示、样本正规方程。上文中心化的有限样本推导及数值例子另作直接核算。
- [[02_Economy/01_Econometrics/02_一元线性回归.md#3.3. $\hat\beta_0$ 和 $\hat\beta_1$ 的解法|本地课程：一元 OLS 解法]]：课程记号和样本协方差之比；消费应用承接课程解释语境。
