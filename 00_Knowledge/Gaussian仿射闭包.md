---
aliases:
  - "Affine closure and marginals of a multivariate Gaussian"
  - "多元正态的仿射变换"
  - "Gaussian affine transformations"
student_os: knowledge-atom
atom_id: STAT-MVN-002
atom_set: multivariate-normal
atom_type: theorem
status: source-checked
requires:
  - "[[联合Gaussian]]"
part_of:
  - "[[多元正态分布.canvas]]"
related:
  - "[[Gaussian条件分布]]"
  - "[[联合高斯独立判据]]"
---

<!-- bilingual-en:start -->
# Gaussian affine transformations

A Gaussian model stays Gaussian when we take fixed weighted sums of its coordinates and add constants. An average, a difference, or a change of units can therefore be described by another Gaussian model. Here **Gaussian** means [[联合Gaussian|jointly Gaussian]]: the coordinates belong to one random vector and every real linear combination is normal, including a constant as a degenerate normal.
<!-- bilingual-en:end -->

*Gaussian 仿射变换*

对一个 Gaussian 向量的坐标取固定加权和，再加上常数，结果仍是 Gaussian。因此，平均值、差值和单位换算都可以得到新的 Gaussian 模型。这里的 Gaussian 指[[联合Gaussian|联合 Gaussian]]：这些坐标属于同一个随机向量，而且每个实线性组合都是正态，常数也作为退化正态包含在内。

<!-- bilingual-en:start -->
## Start with a sum and a difference

Let $Z_1,Z_2$ be independent [[正态分布|standard normals]]: each has mean $0$ and variance $1$. Construct two measurements
$$
X_1=1+Z_1,\qquad X_2=2+Z_1+2Z_2.
$$
They share the noise $Z_1$, so they are dependent. Their [[均值向量|mean vector]] lists the coordinate means, and their [[协方差矩阵|covariance matrix]] records their variances and shared variation:
$$
\mu=\begin{pmatrix}1\\2\end{pmatrix},\qquad
\Sigma=\begin{pmatrix}1&1\\1&5\end{pmatrix}.
$$
Now keep their sum and their difference:
$$
Y_1=X_1+X_2=3+2Z_1+2Z_2,\qquad
Y_2=X_2-X_1=1+2Z_2.
$$
These are still fixed combinations of independent normal noises, so together they are Gaussian. Direct calculation gives means $3,1$, variances $8,4$, and covariance $4$.
<!-- bilingual-en:end -->

*从和与差开始*

设 $Z_1,Z_2$ 是相互独立的[[正态分布|标准正态变量]]，每个的均值为 $0$、方差为 $1$。构造两个测量量
$$
X_1=1+Z_1,\qquad X_2=2+Z_1+2Z_2.
$$
它们共享噪声 $Z_1$，因此并不独立。[[均值向量]]按坐标排列均值，[[协方差矩阵]]记录各自的方差和共同变化：
$$
\mu=\begin{pmatrix}1\\2\end{pmatrix},\qquad
\Sigma=\begin{pmatrix}1&1\\1&5\end{pmatrix}.
$$
现在保留它们的和与差：
$$
Y_1=X_1+X_2=3+2Z_1+2Z_2,\qquad
Y_2=X_2-X_1=1+2Z_2.
$$
它们仍是独立正态噪声的固定组合，所以合起来仍联合 Gaussian。直接计算可得均值为 $3,1$，方差为 $8,4$，协方差为 $4$。

<!-- bilingual-en:start -->
## The matrix records the chosen weights

Writing $Y=AX+b$ collects these operations. The [[矩阵乘法|matrix product]] makes row $i$ of $A$ the list of weights for output $Y_i$; $b_i$ is its added constant. In the example,
$$
A=\begin{pmatrix}1&1\\-1&1\end{pmatrix},\qquad b=0.
$$
More generally, let $X\sim N_p(\mu,\Sigma)$ have $p$ coordinates. Choose a **fixed real** matrix $A\in\mathbb R^{q\times p}$ and fixed $b\in\mathbb R^q$, giving $q$ output coordinates. Then
<!-- bilingual-en:end -->

*矩阵记录每个输出的权重*

写成 $Y=AX+b$，就能把这些操作放在一起。[[矩阵乘法]]使 $A$ 的第 $i$ 行成为输出 $Y_i$ 的权重表，$b_i$ 则是给它加上的常数。本例中，
$$
A=\begin{pmatrix}1&1\\-1&1\end{pmatrix},\qquad b=0.
$$
一般地，设 $X\sim N_p(\mu,\Sigma)$ 有 $p$ 个坐标。取**固定实矩阵** $A\in\mathbb R^{q\times p}$ 和固定向量 $b\in\mathbb R^q$，得到 $q$ 个输出坐标，则

> [!important]
> <!-- bilingual-en:start -->
> $$AX+b\sim N_q(A\mu+b,\ A\Sigma A^T).$$
>
> Here $A^T$ is the [[矩阵转置|transpose]]. The [[均值向量仿射变换|mean transformation]] and [[协方差仿射变换|covariance transformation]] hold more generally when the required moments exist. The extra Gaussian conclusion identifies the **entire distribution** of the output.
> <!-- bilingual-en:end -->
>
> $$AX+b\sim N_q(A\mu+b,\ A\Sigma A^T).$$
>
> 其中 $A^T$ 是[[矩阵转置|转置]]。[[均值向量仿射变换|均值变换]]与[[协方差仿射变换|协方差变换]]本来就在相应矩存在时成立；Gaussian 闭包进一步确定了输出的**完整分布**。

<!-- bilingual-en:start -->
For the sum and difference above, this gives
$$
A\mu=\begin{pmatrix}3\\1\end{pmatrix},\qquad
A\Sigma A^T=\begin{pmatrix}8&4\\4&4\end{pmatrix}.
$$
The result allows rectangular or rank-deficient $A$. It also allows a singular input covariance. One proof uses the Gaussian construction $X=\mu+BZ$, where $Z$ has independent standard-normal coordinates and $BB^T=\Sigma$:
$$
AX+b=(A\mu+b)+(AB)Z.
$$
The output has exactly the same form. No inverse, change-of-variables density formula, or full-dimensional density is needed.
<!-- bilingual-en:end -->

对上面的和与差，公式给出
$$
A\mu=\begin{pmatrix}3\\1\end{pmatrix},\qquad
A\Sigma A^T=\begin{pmatrix}8&4\\4&4\end{pmatrix}.
$$
这里允许 $A$ 是矩形矩阵或秩不足的矩阵，也允许输入协方差奇异。可以直接用 Gaussian 构造来证明：把 $X$ 写成 $\mu+BZ$，其中 $Z$ 的各坐标是独立标准正态，且 $BB^T=\Sigma$，则
$$
AX+b=(A\mu+b)+(AB)Z.
$$
输出仍具有同样的构造形式。这不需要逆矩阵、密度换元公式，也不要求分布有全维密度。

<!-- bilingual-en:start -->
## Selecting or repeating coordinates

A marginal distribution keeps some coordinates and ignores the others. This is a special affine transformation: a selection matrix $P$ has one $1$ in each row, in the column of the coordinate to keep, and $0$ elsewhere. Thus
$$
PX\sim N(P\mu,P\Sigma P^T).
$$
For example, $P=(0\ \ 1)$ selects $X_2\sim N(2,5)$ from our pair.

Repeating a coordinate also remains valid. With $A=\begin{pmatrix}1&0\\1&0\end{pmatrix}$, the output is $(X_1,X_1)^T$, with covariance $\begin{pmatrix}1&1\\1&1\end{pmatrix}$. Its two coordinates always agree, so their difference has variance zero. This is a [[Gaussian密度与奇异支撑|singular Gaussian supported on a line]], not a failure of Gaussian closure.
<!-- bilingual-en:end -->

*选取坐标或重复坐标*

边际分布保留部分坐标、忽略其余坐标，这是仿射变换的特殊情形：选择矩阵 $P$ 的每一行在要保留的坐标位置放一个 $1$，其余位置放 $0$。因此
$$
PX\sim N(P\mu,P\Sigma P^T).
$$
例如 $P=(0\ \ 1)$ 从本例中选出 $X_2\sim N(2,5)$。

重复坐标也完全合法。取 $A=\begin{pmatrix}1&0\\1&0\end{pmatrix}$，输出便是 $(X_1,X_1)^T$，协方差为 $\begin{pmatrix}1&1\\1&1\end{pmatrix}$。两个坐标总相等，所以它们之差的方差为零。这是[[Gaussian密度与奇异支撑|支撑在直线上的奇异 Gaussian]]，并不违反闭包结论。

> [!warning]
> <!-- bilingual-en:start -->
> The input must be **jointly** Gaussian; normal marginal distributions alone do not guarantee that their sum is normal. The weights and shifts here are deterministic. Nonlinear operations have no such general closure: for $Z\sim N(0,1)$, $Z^2$ is nonnegative and nonconstant, so it cannot be normal. It has the [[卡方分布|chi-square distribution]] with one degree of freedom.
> <!-- bilingual-en:end -->
>
> 输入必须**联合** Gaussian；各坐标分别正态，不能保证它们的和正态。这里的权重与平移量都是确定的。非线性运算也没有同样的一般闭包性质：若 $Z\sim N(0,1)$，则 $Z^2$ 非负且不是常数，因此不可能正态；它服从自由度为 $1$ 的[[卡方分布]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.3.2. 仿射变换|Multivariate statistics course, §1.3.2]]: course notation and affine mean/covariance formulas.
- [MIT 6.436J Lecture 14, pp. 3–6, Theorem 1(c)](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf): Gaussian representations and arbitrary deterministic affine transformations, including degenerate cases.
- [MIT 18.443 Lecture 4, p. 3](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf): transformed covariance and singular Gaussian support. The numerical example is calculated directly above.
<!-- bilingual-en:end -->

- [[01_Math/04_多元统计分析/04_多元正态分布The Multivariate Normal Distribution.md#1.3.2. 仿射变换|多元统计课程 §1.3.2]]：课程记号及仿射均值、协方差公式。
- [MIT 6.436J Lecture 14，第 3–6 页，Theorem 1(c)](https://ocw.mit.edu/courses/6-436j-fundamentals-of-probability-fall-2018/ceb423b6a703abd60429367b733f45bf_MIT6_436JF18_lec14.pdf)：Gaussian 构造和任意确定仿射变换，包括退化情形。
- [MIT 18.443 Lecture 4，第 3 页](https://ocw.mit.edu/courses/18-443-statistics-for-applications-fall-2006/da928274a46632ce0d35d2012a1c8e74_lecture4.pdf)：变换后的协方差及奇异 Gaussian 支撑。正文数值例子由所列公式直接计算。
