---
aliases:
- Spectral bound for stochastic matrices
- Markov 矩阵单位圆谱边界
student_os: knowledge-atom
atom_id: LA-EIG-040
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
- '[[Markov矩阵]]'
- '[[谱半径]]'
related:
- '[[Markov矩阵必有特征值一]]'
- '[[Markov单位圆特征值半单]]'
- '[[Markov稳态唯一判据]]'
- '[[Markov矩阵幂收敛判据]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# The spectrum of a stochastic matrix
<!-- bilingual-en:end -->

*Markov 矩阵的谱边界*

<!-- bilingual-en:start -->
A [[Markov矩阵|stochastic matrix]] redistributes probability without creating negative probabilities or changing their total. This restricts its eigenvalues: none can have magnitude greater than one. But “no growing eigenvalue” still allows a mode to keep changing sign or rotating instead of settling down.
<!-- bilingual-en:end -->

[[Markov矩阵]]重新分配概率，同时保持各概率非负、总和不变。这会限制它的特征值：没有一个特征值的模能大于一。不过，“没有增长的特征值”仍然允许某个模式不断变号或旋转，而不稳定在一个值上。

> [!example] Swapping two states · 两个状态不断交换
> <!-- bilingual-en:start -->
> Consider the matrix $F$ and two columns below. A superscript $T$ denotes [[矩阵转置|transpose]], so the coordinates written in a row become a column.
>
> $$F=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad
> F\begin{pmatrix}1\\1\end{pmatrix}=\begin{pmatrix}1\\1\end{pmatrix},\qquad
> F\begin{pmatrix}1\\-1\end{pmatrix}=-\begin{pmatrix}1\\-1\end{pmatrix}.$$
>
> Its eigenvalues are $1$ and $-1$. Starting from probability vector $(1,0)^T$, successive distributions alternate between $(1,0)^T$ and $(0,1)^T$. The eigenvector $(1,-1)^T$ records a difference between states; it is not itself a probability distribution. The eigenvalue $-1$ changes the sign of this difference at every step without enlarging it.
> <!-- bilingual-en:end -->
>
> 考察上面的矩阵 $F$。上标 $T$ 表示[[矩阵转置|转置]]，把横写的坐标变成一列。它的特征值是 $1$ 和 $-1$。若从概率向量 $(1,0)^T$ 开始，后续分布就在 $(1,0)^T$ 与 $(0,1)^T$ 之间交替。特征向量 $(1,-1)^T$ 记录两个状态之间的差异，本身不是概率分布。特征值 $-1$ 使这种差异每一步变号，但不放大。

<!-- bilingual-en:start -->
## The closed disk and the spectral radius
<!-- bilingual-en:end -->

*闭单位圆与谱半径*

<!-- bilingual-en:start -->
Let $P$ be a real $n\times n$ matrix, with integer $n\ge1$, whose entries are nonnegative and whose columns each sum to one. This is the [[Markov矩阵左右约定|column-stochastic convention]]: a probability column $p$ evolves to $Pp$. We consider all complex [[特征对|eigenvalues]], meaning numbers $\lambda$ for which $Pv=\lambda v$ has a nonzero complex vector $v$. For $\lambda=a+ib$, where $i^2=-1$, its magnitude is $|\lambda|=\sqrt{a^2+b^2}$.
<!-- bilingual-en:end -->

设 $P$ 为实 $n\times n$ 矩阵，整数 $n\ge1$，各元素非负、每列之和为一。这是[[Markov矩阵左右约定|列随机约定]]：概率列 $p$ 更新为 $Pp$。这里检查全部复[[特征对|特征值]]，即存在非零复向量 $v$ 满足 $Pv=\lambda v$ 的数 $\lambda$。若 $\lambda=a+ib$、$i^2=-1$，其模为 $|\lambda|=\sqrt{a^2+b^2}$。

> [!important] The bound includes the boundary · 边界也允许取到
> <!-- bilingual-en:start -->
> Every eigenvalue satisfies $|\lambda|\le1$. Moreover, a finite stochastic matrix [[Markov矩阵必有特征值一|always has eigenvalue one]], so its [[谱半径|spectral radius]], the largest eigenvalue magnitude, is exactly
>
> $$\rho(P)=1.$$
>
> The same result holds for a row-stochastic matrix. Neither strict positivity of all entries nor [[不可约链|irreducibility]] is required.
> <!-- bilingual-en:end -->
>
> 每个特征值都满足 $|\lambda|\le1$。有限随机矩阵又[[Markov矩阵必有特征值一|必有特征值一]]，所以最大的特征值模，即[[谱半径]]，恰好为一。行随机矩阵也满足同样结论；不要求每个条目严格为正，也不要求[[不可约链|不可约]]。

<!-- bilingual-en:start -->
## Why redistribution cannot enlarge total absolute size
<!-- bilingual-en:end -->

*为什么重新分配不会放大绝对值总量*

<!-- bilingual-en:start -->
For any real or complex column $z=(z_1,\ldots,z_n)^T$, define $\|z\|_1=\sum_{j=1}^n|z_j|$: add the magnitudes of all its coordinates. This is a measure of size that also applies to signed differences and complex eigenvectors. By the triangle inequality and the nonnegative entries of $P$,

$$\|Pz\|_1
=\sum_i\left|\sum_jP_{ij}z_j\right|
\le\sum_i\sum_jP_{ij}|z_j|
=\sum_j\left(\sum_iP_{ij}\right)|z_j|
=\|z\|_1.$$

The last equality uses the column sums. Substituting an eigenvector gives $|\lambda|\,\|v\|_1\le\|v\|_1$. Since $v\ne0$, its size is positive and division gives $|\lambda|\le1$.
<!-- bilingual-en:end -->

对任意实或复列向量 $z=(z_1,\ldots,z_n)^T$，定义 $\|z\|_1=\sum_{j=1}^n|z_j|$，即把各坐标的模相加。这种大小度量也适用于带正负号的差异与复特征向量。由三角不等式及 $P$ 元素非负，得到上面的不等式；最后一步用了每列之和为一。代入特征向量后，$|\lambda|\,\|v\|_1\le\|v\|_1$。由于 $v\ne0$，可以除以正数 $\|v\|_1$，得出闭单位圆界。

<!-- bilingual-en:start -->
Repeating the same inequality gives $\|P^kz\|_1\le\|z\|_1$ for every integer $k\ge0$, with $P^0=I_n$, the [[单位矩阵|identity matrix]]. In matrix-norm notation this reads $\|P^k\|_1=1$, where the matrix one-norm is the largest absolute column sum. Indeed, every power remains nonnegative, and $\mathbf1^TP^k=\mathbf1^T$, where $\mathbf1$ is the all-ones column. Thus its absolute column sums are all exactly one. This uniform bound will rule out certain [[Markov单位圆特征值半单|Jordan growth at boundary eigenvalues]].
<!-- bilingual-en:end -->

反复使用同一不等式，便有全部整数 $k\ge0$ 下 $\|P^kz\|_1\le\|z\|_1$；其中 $P^0=I_n$ 是[[单位矩阵]]。用矩阵范数记号写，就是 $\|P^k\|_1=1$，这里矩阵的1-范数指最大的绝对列和。因为每个幂仍非负，且 $\mathbf1^TP^k=\mathbf1^T$，其中 $\mathbf1$ 是全一列，所以每列的绝对值之和都恰好为一。这个统一的有界性还会排除[[Markov单位圆特征值半单|边界特征值处的某些 Jordan 增长]]。

<!-- bilingual-en:start -->
For a row-stochastic matrix, the analogous size is $\|z\|_\infty=\max_j|z_j|$. Each output coordinate is a weighted average with nonnegative weights summing to one, so $\|Pz\|_\infty\le\|z\|_\infty$. Its powers have matrix infinity-norm $\|P^k\|_\infty=1$, the largest absolute row sum. The proof therefore covers either convention without mixing their left and right equations.
<!-- bilingual-en:end -->

对行随机矩阵，对应的大小是 $\|z\|_\infty=\max_j|z_j|$。每个输出坐标都是非负权重、权重和为一的加权平均，所以 $\|Pz\|_\infty\le\|z\|_\infty$。每个幂的矩阵无穷范数 $\|P^k\|_\infty=1$，即最大的绝对行和为一。因此两套约定都能证明同一个谱界，无须混用左右方程。

> [!warning] A closed disk does not promise convergence · 闭盘不保证收敛
> <!-- bilingual-en:start -->
> The swap matrix has $-1$ on the boundary and oscillates. A cyclic permutation of three states similarly has eigenvalues $1$ and $(-1\pm i\sqrt3)/2$, all of magnitude one, and returns to its starting state every three steps. The sharper [[Markov矩阵幂收敛判据|criterion for convergence of stochastic matrix powers]] is a separate question. Also, $\rho(P)=1$ rules out $P^k\to0$; probabilities retain total mass one. A distribution can approach a nonzero stationary distribution instead.
> <!-- bilingual-en:end -->
>
> 交换矩阵在边界上有 $-1$，所以会振荡。三个状态循环置换时，特征值为 $1$ 与 $(-1\pm i\sqrt3)/2$，模都等于一，每三步回到起始状态。更精确的[[Markov矩阵幂收敛判据|随机矩阵幂收敛条件]]属于另一个问题。此外，$\rho(P)=1$ 排除了 $P^k\to0$；概率仍要保留总质量一。分布可能趋近的是一个非零稳态分布。

## 来源与核验

<!-- bilingual-en:start -->
- [Levin–Peres with Wilmer, Markov Chains and Mixing Times, 2nd ed., Lemma 12.1(i), PDF p.177 / printed p.161](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=177): the closed-unit-disk bound for every finite transition matrix. The absolute-sum proof and row/column norm equalities are derived above.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]], p.1 and p.3: the column convention, stochasticity of powers and the transpose convention. The swap and three-cycle boundary cases are checked directly; no strict bound for all other eigenvalues is assumed.
<!-- bilingual-en:end -->

- Levin–Peres 等第2版，引理12.1(i)，PDF第177页／印刷161页：核对任意有限转移矩阵的闭单位圆界；正文自行展开绝对值和证明及两种范数等式。
- MIT第2.11节摘要第1、3页：核对列随机约定、幂仍为随机矩阵与转置约定。交换及三循环边界直接核算，不把其他特征值一律视为严格在单位圆内。
