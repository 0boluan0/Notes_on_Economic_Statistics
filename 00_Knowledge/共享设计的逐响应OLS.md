---
aliases:
  - Response-wise OLS equivalence
  - Separate OLS equivalence
  - 共享设计下的逐响应最小二乘
student_os: knowledge-atom
atom_id: STAT-MVR-004
atom_set: multivariate-linear-regression
atom_type: theorem-boundary
status: source-checked
part_of:
  - "[[多元线性回归.canvas|多元线性回归]]"
requires:
  - "[[多响应线性回归]]"
  - "[[普通最小二乘]]"
  - "[[Frobenius范数]]"
leads_to:
  - "[[多响应残差与误差SSP]]"
related:
  - "[[满列秩与OLS唯一性]]"
  - "[[矩阵正态误差结构]]"
  - "[[相关行下的多响应GLS]]"
  - "[[相关行不容独立推断]]"
---

<!-- bilingual-en:start -->
# Response-wise least squares
<!-- bilingual-en:end -->

*共享设计下的逐响应最小二乘*

<!-- bilingual-en:start -->
When mathematics and reading scores use the same study-time predictors, ordinary multivariate least squares can fit one regression for mathematics and another for reading. Combining the two coefficient columns gives exactly the joint least-squares solution. Correlation between the outcomes does not change this calculation; it does affect what statistical conclusions can be drawn from it.
<!-- bilingual-en:end -->

数学与阅读成绩使用同一套学习时间预测变量时，普通多响应最小二乘可以分别拟合数学和阅读，再把两列系数放到一起，结果恰好等于联合最小二乘解。响应之间的相关性不改变这个计算规则，却会影响随后能作出的统计推断。

<!-- bilingual-en:start -->
## Why the objective separates
<!-- bilingual-en:end -->

*为什么目标可以拆开*

<!-- bilingual-en:start -->
Use the [[多响应线性回归|shared-design model]] with $Y\in\mathbb R^{n\times q}$, $X\in\mathbb R^{n\times k}$ and coefficients $B\in\mathbb R^{k\times q}$. Write response columns $Y=[y_1,\ldots,y_q]$ and coefficient columns $B=[b_1,\ldots,b_q]$. [[普通最小二乘|Ordinary least squares]] minimizes squared observed-minus-fitted differences. The squared [[Frobenius范数|Frobenius norm]] sums every entry’s square, so
<!-- bilingual-en:end -->

使用[[多响应线性回归|共享设计模型]]：$Y\in\mathbb R^{n\times q}$、$X\in\mathbb R^{n\times k}$、$B\in\mathbb R^{k\times q}$。把响应列写成 $Y=[y_1,\ldots,y_q]$，系数列写成 $B=[b_1,\ldots,b_q]$。[[普通最小二乘]]最小化“观测减拟合”的平方；[[Frobenius范数]]平方把所有元素的平方相加，因此

$$L(B)=\|Y-XB\|_F^2
=\sum_{j=1}^q\|y_j-Xb_j\|_2^2.$$

<!-- bilingual-en:start -->
Changing $b_1$ changes only the first term, changing $b_2$ only the second, and so on. If every column minimizes its own term, their sum is minimal; if one column does not, improving it reduces the sum without affecting the others. This proves the equivalence without a probability model.
<!-- bilingual-en:end -->

改变 $b_1$ 只改变第一项，改变 $b_2$ 只改变第二项，依此类推。每一列都把自己的项最小化，整项之和就最小；如果有一列没有做到，单独改进它就能降低总和而不影响其他项。这直接证明了等价性，不需要概率模型。

> [!important] What is shared / 共享的条件
> <!-- bilingual-en:start -->
> All responses use the same aligned observations and design, ordinary unweighted squared loss, and freely chosen coefficient columns. Cross-response constraints or a coupled penalty can invalidate the separation. No independence or Gaussian assumption is used in the algebra above.
> <!-- bilingual-en:end -->
>
> 所有响应使用相同且对齐的观测、同一设计矩阵、普通未加权平方损失，且各列系数可以自由选择。跨响应约束或耦合惩罚可能破坏分离；上述代数没有使用独立性或 Gaussian 假设。

<!-- bilingual-en:start -->
## A common formula and a small calculation
<!-- bilingual-en:end -->

*同一个公式与一个小计算*

<!-- bilingual-en:start -->
When $X$ has [[满列秩与OLS唯一性|full column rank]], meaning its $k$ columns are linearly independent, the [[OLS正规方程|normal equations]] are $X^TX\hat B=X^TY$. Since $X^TX$ is invertible,
<!-- bilingual-en:end -->

当 $X$ [[满列秩与OLS唯一性|满列秩]]，即 $k$ 列线性独立时，[[OLS正规方程|正规方程]]为 $X^TX\hat B=X^TY$；$X^TX$ 可逆，因而

$$\hat B=(X^TX)^{-1}X^TY.$$

<!-- bilingual-en:start -->
The inverse multiplies every response column in the same way. This is a mathematical formula, not a recommendation to form the inverse in software; QR or SVD can solve least-squares problems directly. The [[正规方程条件数平方|conditioning of normal equations]] explains the numerical distinction.
<!-- bilingual-en:end -->

同一个左侧矩阵作用在每个响应列上。这是数学公式，不是建议软件显式构造逆矩阵；QR 或 SVD 可以直接求解最小二乘。[[正规方程条件数平方]]解释了数值上的区别。

> [!example] Fit the two score columns / 拟合两列成绩
> <!-- bilingual-en:start -->
> Use the three observations from the model card:
> $$X=\begin{bmatrix}1&0\\1&1\\1&2\end{bmatrix},\qquad Y=\begin{bmatrix}51&39\\54&44\\63&43\end{bmatrix}.$$
> Then $X^TX=\begin{bmatrix}3&3\\3&5\end{bmatrix}$ and $X^TY=\begin{bmatrix}168&126\\180&130\end{bmatrix}$, giving
> $$\hat B=\begin{bmatrix}50&40\\6&2\end{bmatrix},\qquad R=Y-X\hat B=\begin{bmatrix}1&-1\\-2&2\\1&-1\end{bmatrix}.$$
> Each residual column has squared length $6$, so the joint loss is $12$. Each column is orthogonal to both design columns: $X^TR=0$. The fitted mathematics slope is $6$; the illustrative population slope in the model card was $5$. An estimate from one sample need not equal the true coefficient.
> <!-- bilingual-en:end -->
>
> 对模型卡中的三行成绩计算 $X^TX$ 与 $X^TY$，得到上述系数和残差。每个残差列的平方和为 $6$，总损失为 $12$；每列残差都与两列设计正交，$X^TR=0$。这里估出的数学斜率是 $6$，模型卡示意总体斜率是 $5$：一次样本的估计不必等于真值。

<!-- bilingual-en:start -->
If $X$ is rank deficient, the column-wise argument still works. Each fitted column is the unique orthogonal projection onto the [[列空间|column space]] of $X$, even though its coefficient vector may be nonunique. For instance, duplicating the intercept lets its total coefficient be split arbitrarily between two identical columns. A generalized inverse selects one solution; it does not identify those coefficients separately.
<!-- bilingual-en:end -->

若 $X$ 秩亏，逐列论证仍成立。每个拟合列仍是到 $X$ [[列空间]]的唯一正交投影，但系数向量可能不唯一。例如重复截距列后，同一个总截距可以任意分给两列。广义逆能选一个解，却不能让这些系数分别被识别。

<!-- bilingual-en:start -->
## Equal estimates do not mean independent inference
<!-- bilingual-en:end -->

*估计相同不表示推断相互独立*

<!-- bilingual-en:start -->
With fixed full-rank $X$, zero-mean error rows uncorrelated across observations, and common finite within-row covariance $\Sigma$, write $A=(X^TX)^{-1}X^T$. Then $\hat b_a-b_a=A u_a$ for response $a$, where $u_a$ is column $a$ of the error matrix $U$. Since $\operatorname{Cov}(u_a,u_b\mid X)=\Sigma_{ab}I_n$,
<!-- bilingual-en:end -->

固定满列秩 $X$，设误差行均值为零、不同观测行不相关，并具有共同有限行内协方差 $\Sigma$。写 $A=(X^TX)^{-1}X^T$，第 $a$ 个响应满足 $\hat b_a-b_a=A u_a$，其中 $u_a$ 是误差矩阵 $U$ 的第 $a$ 列。因为 $\operatorname{Cov}(u_a,u_b\mid X)=\Sigma_{ab}I_n$，

$$\operatorname{Cov}(\hat b_a,\hat b_b\mid X)
=\Sigma_{ab}AA^T=\Sigma_{ab}(X^TX)^{-1}.$$

<!-- bilingual-en:start -->
This follows by the [[协方差仿射变换|covariance transformation rule]]. Nonzero response covariance therefore produces cross-response coefficient covariance, even though the two estimates were computed separately. Joint hypotheses such as [[多响应一般线性假设|$CBA=D$]] use this dependence; separate marginal p-values do not automatically give a valid joint test.
<!-- bilingual-en:end -->

这由[[协方差仿射变换|协方差变换规则]]直接得到。响应协方差非零会产生跨响应系数协方差，即使两个估计是分别算出来的。[[多响应一般线性假设|$CBA=D$]] 等联合假设需要使用这种依赖；分别算出的边际 p 值不会自动变成有效联合检验。

<!-- bilingual-en:start -->
## A weighted objective may have the same minimizer
<!-- bilingual-en:end -->

*加权目标也可能有相同的最小点*

<!-- bilingual-en:start -->
Let $W$ be a fixed symmetric [[正定矩阵|positive-definite]] $q\times q$ response weight, so every nonzero response vector $v$ satisfies $v^TWv>0$, and use $L_W(B)=\operatorname{tr}((Y-XB)W(Y-XB)^T)$; trace means the diagonal sum. Off-diagonal entries of $W$ introduce cross-products, so this objective is not the displayed column sum. Nevertheless, it has the same OLS minimizers under the shared unconstrained design.
<!-- bilingual-en:end -->

取固定对称[[正定矩阵|正定]]的 $q\times q$ 响应权重 $W$，即每个非零响应向量 $v$ 都满足 $v^TWv>0$，使用 $L_W(B)=\operatorname{tr}((Y-XB)W(Y-XB)^T)$，其中迹是对角元之和。$W$ 的非对角元会引入交叉乘积，因此这个目标并非前面的逐列和；但在共享、无约束设计下，其最小点仍与 OLS 相同。

<!-- bilingual-en:start -->
To see this, put $B=\hat B+\Delta$ and $R=Y-X\hat B$. Because $X^TR=0$, the cross terms vanish, leaving $L_W(B)=L_W(\hat B)+\operatorname{tr}(X\Delta W\Delta^TX^T)$. The added term is $\|X\Delta W^{1/2}\|_F^2\ge0$, where $W^{1/2}$ is the [[半正定主平方根|principal square root]]: the symmetric positive-definite matrix whose square is $W$. It is zero exactly when $X\Delta=0$. Thus fitted values agree, and full rank makes coefficients agree too.
<!-- bilingual-en:end -->

令 $B=\hat B+\Delta$、$R=Y-X\hat B$。由 $X^TR=0$，展开后的交叉项消失，得到 $L_W(B)=L_W(\hat B)+\operatorname{tr}(X\Delta W\Delta^TX^T)$。增加项等于 $\|X\Delta W^{1/2}\|_F^2\ge0$，其中 $W^{1/2}$ 是[[半正定主平方根|主平方根]]，即平方等于 $W$ 的对称正定矩阵；它恰在 $X\Delta=0$ 时为零。所以拟合值相同，满秩时系数也相同。

<!-- bilingual-en:start -->
For independent Gaussian error rows with a **fixed** positive-definite response covariance, the [[矩阵正态误差结构|matrix-normal likelihood]] uses $W=\Sigma^{-1}$ and therefore gives this same estimate of $B$. Joint maximum likelihood with an unrestricted unknown covariance additionally needs a positive-definite fitted residual scatter matrix. The [[Gaussian协方差MLE边界|analogous scatter-rank issue for iid observations with unknown mean]] explains why covariance estimation has a separate existence condition; in regression, check the fitted residual scatter and its residual degrees of freedom, rather than copying that card’s sample-size threshold.
<!-- bilingual-en:end -->

对独立 Gaussian 误差行和**固定**正定响应协方差，[[矩阵正态误差结构|矩阵正态似然]]使用 $W=\Sigma^{-1}$，因而给出相同的 $B$ 估计。若还要联合极大似然估计不受约束的未知协方差，就需要拟合残差散布矩阵正定。[[Gaussian协方差MLE边界|iid 未知均值模型的类似散布秩问题]]说明了为何协方差估计有独立的存在条件；回归中要查拟合残差散布及其自由度，不能直接搬用该卡的样本量门槛。

<!-- bilingual-en:start -->
Weighting observation rows is different. Known row covariance $\Omega$ generally leads to [[相关行下的多响应GLS|GLS with a different row metric]]; $\Omega$ proportional to the identity is a sufficient case for agreement with OLS, not a necessary condition for every particular design. Nonseparable joint error covariance can also couple responses in efficient GLS. The ordinary Frobenius-loss identity remains true whichever sampling model generated the data.
<!-- bilingual-en:end -->

给观测行加权是另一件事。已知行协方差 $\Omega$ 一般会得到[[相关行下的多响应GLS|使用不同观测行度量的 GLS]]；$\Omega$ 与单位矩阵成比例是 GLS 与 OLS 相同的充分情形，却不是每个具体设计都必须满足的条件。不可分离的整体误差协方差也可能使有效 GLS 跨响应耦合。不管数据来自什么模型，原来的 Frobenius 平方损失恒等式都仍成立。

<!-- bilingual-en:start -->
Response-specific missing rows, predictors or row weights no longer match this one common-design problem. Separate regressions may still be possible, but the displayed common formula need not describe them. A low-rank coefficient constraint, a shared-slope restriction or a coupled penalty can also make one response’s optimum depend on the others. Not every penalty does so: a sum of independent column penalties remains separable.
<!-- bilingual-en:end -->

不同响应有不同缺失行、预测变量或观测权重，就不再是这一张共同设计的问题。仍可能分别回归，但前面的统一公式未必描述它们。系数低秩约束、跨响应共享斜率或耦合惩罚也可能让一个响应的最优解依赖另一个。不是所有惩罚都会破坏分离：独立列惩罚之和仍可分离。

## 来源与核验

<!-- bilingual-en:start -->
- [statsmodels, Multivariate Linear Model](https://www.statsmodels.org/stable/examples/notebooks/generated/multivariate_ls.html), introduction: common-design estimates agree with separate OLS while multivariate inference is distinct. [The `mv_test` reference](https://www.statsmodels.org/stable/generated/statsmodels.multivariate.multivariate_ols.MultivariateLSResults.mv_test.html) gives the coefficient/response hypothesis structure.
- [Boyd, Stanford EE263 Lecture5](https://ee263.stanford.edu/archive/ls.pdf), slides3–9: projection, full-rank normal equations, residual orthogonality, completion of squares and QR. Column separation, the weighted completion-of-squares proof, rank-deficient extension and the numerical example are explicitly derived here.
- [SciPy `matrix_normal`, Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.matrix_normal.html): the Gaussian trace-quadratic likelihood with row and response covariance. Fixing response covariance and identity row covariance gives the stated weighted objective; a general row metric gives the GLS distinction.
- [[01_Math/04_多元统计分析/07_多元线性回归multivariate linear regression.md#1.3. 最小二乘估计|Permanent course Chapter7 §1.3]]: single-response formula and covariance. The cross-response covariance above is calculated from the explicitly stated error covariance, rather than assumed from the fitted columns.
<!-- bilingual-en:end -->

statsmodels 核对共享设计的逐响应等价与联合假设；Stanford 指定页核对正规方程、投影、正交与配方证明。SciPy 原密度支持固定协方差下的加权目标，未据此声称联合协方差 MLE 总存在。数值、加权配方及跨响应协方差均在正文直接推导；永久课程用于单响应符号衔接。
