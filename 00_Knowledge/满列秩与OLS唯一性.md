---
aliases:
- OLS rank condition
- OLS 唯一解条件
- OLS coefficient uniqueness
- OLS系数唯一性
student_os: knowledge-atom
atom_id: ECON-OLS-004
atom_type: proposition
status: source-checked
part_of:
- '[[OLS 线性回归.canvas|OLS 线性回归]]'
- '[[多重共线性.canvas|多重共线性与设计矩阵诊断]]'
- '[[虚拟变量与交互项.canvas|虚拟变量与交互项]]'
requires:
- '[[普通最小二乘]]'
related:
- '[[最小二乘解唯一性]]'
implies:
- '[[OLS正规方程]]'
---

<!-- bilingual-en:start -->
# When are OLS coefficients unique?
<!-- bilingual-en:end -->

*满列秩与 OLS 系数唯一性*

<!-- bilingual-en:start -->
A fitted regression can have one definite fitted value for every observed row while still having several equally good coefficient vectors. **Full column rank** rules out this ambiguity: each predictor column supplies a direction that cannot be reproduced exactly by combining the other columns. The condition concerns the actual design matrix, including the intercept and all coded terms.
<!-- bilingual-en:end -->

一份回归可以给每个已观察行确定唯一拟合值，却仍有多组同样好的系数。**满列秩**排除这种歧义：每个解释变量列提供的方向，都不能由其他列精确组合出来。这个条件针对实际设计矩阵，包括截距和全部编码项。

<!-- bilingual-en:start -->
## The same group means, several coefficient vectors
<!-- bilingual-en:end -->

## 最小反例

<!-- bilingual-en:start -->
Consider the original two-indicator example: in this dataset every row is coded in exactly one of the groups named `male` and `female`. This is a condition on these recorded categories. Let $m_i$ and $f_i$ be their zero-or-one indicators, so $m_i+f_i=1$ on every row. With an intercept, the fitted expression is
<!-- bilingual-en:end -->

考虑原来的双指示变量例子：这份数据的每行恰好被编码到名为 `male`、`female` 的两组之一。这是对本数据记录类别的约定。令 $m_i,f_i$ 为相应的零一指示值，则每行都有 $m_i+f_i=1$。若再加入截距，拟合式为

$$b_0+b_m m_i+b_f f_i.$$

<!-- bilingual-en:start -->
Take four outcomes $(2,4,6,8)$, with the first two in the first group and the last two in the second. Least squares fits the group means $3$ and $7$. It therefore determines $b_0+b_m=3$ and $b_0+b_f=7$, but cannot separate each sum into its components. For every real number $c$,
<!-- bilingual-en:end -->

取四个结果 $(2,4,6,8)$，前两个属于第一组，后两个属于第二组。最小二乘拟合的组均值是 $3$ 和 $7$，所以它确定 $b_0+b_m=3$、$b_0+b_f=7$，却不能把每个和拆成唯一的分量。对任意实数 $c$，

$$(b_0,b_m,b_f)=(c,3-c,7-c)$$

<!-- bilingual-en:start -->
gives the same fitted outcomes $(3,3,7,7)$, residuals $(-1,1,-1,1)$ and squared total $4$. For instance, $(0,3,7)$ and $(3,0,4)$ are distinct coefficient vectors with identical fits. Their group contrast $b_f-b_m=4$ is fixed even though individual coefficients are not. There are four observations and only three listed coefficients, yet the coefficients are not unique: counting rows alone does not check the rank condition.
<!-- bilingual-en:end -->

都会得到同一组拟合值 $(3,3,7,7)$、残差 $(-1,1,-1,1)$ 和平方和 $4$。例如 $(0,3,7)$ 与 $(3,0,4)$ 是不同系数向量，拟合却完全相同。个别系数虽不唯一，组间对比 $b_f-b_m=4$ 却已确定。此处有四个观测、只列了三个系数，系数仍不唯一；只数行数不能核对秩条件。

<!-- bilingual-en:start -->
## The exact condition
<!-- bilingual-en:end -->

*准确条件*

<!-- bilingual-en:start -->
Let real $X$ have $n\ge1$ rows and $k\ge1$ columns, and let $y\in\mathbb R^n$ hold the observed outcomes. Each column is one fitted term; an intercept counts as a column. [[矩阵秩|Rank]] counts how many independent column directions remain. **Full column rank** means $\operatorname{rank}(X)=k$, equivalently that the columns are [[线性无关|linearly independent]]. In particular, it requires $n\ge k$, but that count alone is not sufficient.

For unrestricted [[普通最小二乘|OLS]] with squared Euclidean loss, the coefficient vector is unique **if and only if** $X$ has full column rank. In that case the [[OLS正规方程|normal equations]] give
<!-- bilingual-en:end -->

令实数矩阵 $X$ 有 $n\ge1$ 行、$k\ge1$ 列，$y\in\mathbb R^n$ 存放观测结果。每列是一个拟合项，截距也占一列。[[矩阵秩|秩]]统计剩下多少独立的列方向。**满列秩**表示 $\operatorname{rank}(X)=k$，等价于各列[[线性无关]]。它必然要求 $n\ge k$，但这个数量条件并不充分。

对不约束系数、使用欧氏平方损失的[[普通最小二乘|OLS]]，系数向量唯一，**当且仅当** $X$ 满列秩。此时由[[OLS正规方程]]得到

$$\boxed{\hat\beta=(X'X)^{-1}X'y.}$$

<!-- bilingual-en:start -->
Here a prime is [[矩阵转置|transpose]], $X'X$ is a $k\times k$ matrix, and the superscript $-1$ denotes its [[逆矩阵|inverse]]. To see why it is invertible, let $v\in\mathbb R^k$ be a nonzero coefficient direction. Full column rank means $Xv\ne0$, so
<!-- bilingual-en:end -->

撇号是[[矩阵转置|转置]]，$X'X$ 是 $k\times k$ 矩阵，上标 $-1$ 表示[[逆矩阵]]。为什么它可逆？取任一非零系数方向 $v\in\mathbb R^k$。满列秩使 $Xv\ne0$，因此

$$v'X'Xv=(Xv)'(Xv)=\lVert Xv\rVert_2^2>0.$$

<!-- bilingual-en:start -->
This is the [[Gram矩阵正定判据|positive-definite Gram-matrix condition]], which ensures invertibility. Conversely, if the columns are dependent, there is a nonzero $v$ with $Xv=0$. Such a vector belongs to the [[零空间|nullspace]], the directions erased by multiplication by $X$. Whenever $\hat\beta$ minimizes the squared loss, every $\hat\beta+t v$, for real $t$, has the same fitted vector and loss. Thus rank deficiency gives infinitely many coefficient minimizers for every outcome vector $y$; it is not a failure of the minimum to exist.
<!-- bilingual-en:end -->

这就是保证可逆性的 [[Gram矩阵正定判据|Gram 矩阵正定条件]]。反过来，若各列相关，就存在非零 $v$ 使 $Xv=0$。这样的方向属于[[零空间]]，也就是被乘以 $X$ 后抹去的方向。只要 $\hat\beta$ 是最小化解，任意实数 $t$ 对应的 $\hat\beta+t v$ 都产生同样的拟合向量和损失。因此，秩亏时对每个结果向量 $y$ 都有无穷多个系数最小化解，不是最小值不存在。

<!-- bilingual-en:start -->
For comparison, the design with rows $(1,0),(1,1),(1,2)$ has two independent columns. It gives $X'X=\begin{bmatrix}3&3\\3&5\end{bmatrix}$, whose determinant is $6\ne0$. With outcomes $(1,2,2)$, the unique intercept and slope are $(7/6,1/2)$, as in the OLS example.
<!-- bilingual-en:end -->

作为对照，行分别为 $(1,0),(1,1),(1,2)$ 的设计有两个独立列，其 $X'X=\begin{bmatrix}3&3\\3&5\end{bmatrix}$，行列式为 $6\ne0$。对结果 $(1,2,2)$，唯一截距与斜率正是 OLS 例中的 $(7/6,1/2)$。

<!-- bilingual-en:start -->
## Unique fit, coding choices and precision
<!-- bilingual-en:end -->

## 使用条件与含义边界

<!-- bilingual-en:start -->
Even under rank deficiency, the fitted vector $X\hat\beta$ **on the observed design** is unique. All possible fitted vectors form the [[列空间|column space]] of $X$, and least squares selects its unique nearest vector to $y$. The general result is [[最小二乘解唯一性|least-squares uniqueness]]. Uniqueness of this fitted vector should not be turned into uniqueness of every coefficient or of every prediction at an arbitrary new predictor configuration.

In the two-group example, remove the `female` column while retaining the intercept. The remaining two columns are independent when both groups are observed; the intercept becomes $7$ and the `male` coefficient $-4$. Removing `male` instead gives intercept $3$ and `female` coefficient $4$. The fitted group means agree, but the coefficients now express different reference-group comparisons. A suitable explicit constraint or a [[最小范数最小二乘解|minimum-norm selection rule]] can also select a representative. Such a convention does not make the original unconstrained coefficients uniquely determined by the squared-loss criterion.
<!-- bilingual-en:end -->

即使秩亏，**原来观察到的设计上**的拟合向量 $X\hat\beta$ 仍唯一。所有可能拟合向量组成 $X$ 的[[列空间]]，最小二乘选取其中距离 $y$ 最近的唯一向量；一般结论见[[最小二乘解唯一性]]。但不能把这个拟合向量唯一，扩大成每个系数、或任意新解释变量组合处的预测都唯一。

双组例子中，保留截距、删除 `female` 列；两组都实际出现时，剩下两列独立，截距变成 $7$，`male` 系数为 $-4$。若改为删除 `male`，则截距为 $3$，`female` 系数为 $4$。拟合组均值一致，系数却表达不同的参考组比较。合适的显式约束或[[最小范数最小二乘解|最小范数选择规则]]也能选定一个代表；这种约定不会使原来无约束的各个系数变成由平方损失准则唯一确定。

> [!warning] Unique is not the same as precise / 唯一不等于精确
> <!-- bilingual-en:start -->
> Nearly dependent columns can still have full rank in exact arithmetic. The answer is then unique, but it may be sensitive to small numerical or data changes; under an appropriate sampling model, some coefficient estimates may also have large variance. See [[近似共线性精度|precision under near collinearity]]. Exact rank deficiency, numerical conditioning and statistical precision are separate questions. Full rank alone also establishes neither unbiasedness nor causality.
> <!-- bilingual-en:end -->
>
> 几乎相关的列，在精确算术下仍可能满秩。此时答案唯一，却可能对数值误差或数据微小变化敏感；在相应抽样模型下，某些系数估计也可能有很大方差。详见[[近似共线性精度]]。精确秩亏、数值条件性、统计精度是不同问题，满秩本身也不能证明无偏或因果含义。

<!-- bilingual-en:start -->
When $n=k$ and $X$ is invertible, the coefficient fit is unique and interpolates every outcome. Nevertheless, no residual degrees of freedom remain to estimate an unknown common error variance by dividing the residual sum of squares by $n-k$. The requirement for [[OLS经典协方差估计|classical covariance estimation]] is stronger than the requirement for a unique coefficient fit.
<!-- bilingual-en:end -->

当 $n=k$ 且 $X$ 可逆时，系数拟合唯一，并能精确经过全部结果；但没有剩余自由度，不能再用残差平方和除以 $n-k$ 估计未知共同误差方差。[[OLS经典协方差估计]]的条件比“能唯一拟合系数”更强。

## 继续

- [[OLS正规方程|OLS normal equations · OLS 正规方程]]
- [[多重共线性|Multicollinearity · 多重共线性]]
- [[近似共线性精度|Precision under near collinearity · 近似共线性精度]]

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.3sum.pdf#page=3|MIT 18.06SC Session 2.3 summary]], p.3, and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S03_Lecture_Lecture_16_Projection_Matrices_and_Least_Squares.pdf#page=15|Lecture 16 transcript]], pp.15–17: independent columns and invertibility of $X'X$ through the squared-length identity. The nullspace and two-group calculations here give the nonunique direction explicitly.
- Hansen, [Econometrics, January 2010](https://users.ssc.wisc.edu/~xshi28/Econometrics_Hansen.pdf), §§3.3–3.6, PDF pp.36–42, and §4.11, PDF pp.65–66: inverse formula and the distinction between exact and near collinearity. The singular case here is stated as nonuniqueness of coefficients, not absence of fitted values.
- [[02_Economy/01_Econometrics/05_多元回归模型的矩阵表达.md#2.1. OLS|Permanent matrix-OLS course section]]: local matrix formula and its invertibility condition.
- Further reading retained from the original card: Wooldridge, *Introductory Econometrics*, 6th ed., Chapter 3, MLR.3 and §§3.2a,3.3.
<!-- bilingual-en:end -->

- MIT 指定页核对独立列与 $X'X$ 可逆的平方长度论证；本文零空间和两组计算直接展示不唯一方向。
- Hansen 指定页核对逆矩阵公式、精确与近共线性区别；奇异情形明确为系数不唯一，不是拟合值不存在。
- 永久矩阵 OLS 课程保留本地公式及可逆条件；Wooldridge 为原卡保留的延伸阅读。
