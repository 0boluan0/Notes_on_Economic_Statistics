---
aliases:
- Semisimplicity of unit-circle eigenvalues of stochastic matrices
- Markov 单位圆谱半单性
student_os: knowledge-atom
atom_id: LA-EIG-049
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
- '[[Markov矩阵]]'
- '[[离散系统有界判据]]'
- '[[半单特征值]]'
related:
- '[[Markov矩阵谱边界]]'
- '[[Markov矩阵幂收敛判据]]'
- '[[Markov稳态唯一判据]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# Semisimple boundary eigenvalues of stochastic matrices
<!-- bilingual-en:end -->

*Markov 矩阵的单位圆特征值为何半单*

<!-- bilingual-en:start -->
A stochastic matrix can keep a mode oscillating forever, but it cannot attach a growing Jordan-chain contribution to a mode of magnitude one. The reason is concrete: its repeated probability redistributions remain uniformly bounded. This is stronger than merely knowing that every eigenvalue lies in the [[Markov矩阵谱边界|closed unit disk]].
<!-- bilingual-en:end -->

随机矩阵可以让某个模式一直振荡，却不能在模为一的模式上附加不断增长的 Jordan 链贡献。原因很具体：反复重新分配概率，仍然一致有界。这比只知道全部特征值位于[[Markov矩阵谱边界|闭单位圆内]]更强。

<!-- bilingual-en:start -->
## The extra growth that a Jordan block would create
<!-- bilingual-en:end -->

*Jordan 块会带来什么额外增长*

<!-- bilingual-en:start -->
The size-two [[Jordan块|Jordan block]] $J$ below adds the second coordinate into the first. Its powers are

$$J=\begin{pmatrix}1&1\\0&1\end{pmatrix},\qquad
J^k=\begin{pmatrix}1&k\\0&1\end{pmatrix},\qquad
J^k\begin{pmatrix}0\\1\end{pmatrix}=\begin{pmatrix}k\\1\end{pmatrix}.$$

Here $k$ is a nonnegative integer; the formula follows by multiplication or the [[Jordan块的幂|Jordan-block power formula]]. Although both eigenvalues equal one, one coordinate grows with $k$. This displayed $J$ is not a stochastic matrix: its second column sums to two. The issue is whether such a block could be hidden inside a stochastic matrix after a change of basis. It cannot.
<!-- bilingual-en:end -->

上面的二阶 [[Jordan块]] $J$ 把第二坐标加进第一坐标。对非负整数 $k$，直接乘法或[[Jordan块的幂|块幂公式]]给出所示结果。虽然两个特征值都是一，一个坐标却随着 $k$ 增长。展示的 $J$ 本身不是随机矩阵，因为第二列之和为二。真正要问的是：随机矩阵换基之后，会不会隐藏这样的块？答案仍然是否定的。

> [!important] Every boundary eigenvalue is semisimple · 每个边界特征值都半单
> <!-- bilingual-en:start -->
> Let $P$ be a finite real row- or column-stochastic matrix of size $n\times n$, with $n\ge1$. For each complex eigenvalue $\lambda$ satisfying $|\lambda|=1$, all associated complex Jordan blocks have size one. Equivalently, $\lambda$ is a [[半单特征值|semisimple eigenvalue]]: its [[代数重数与几何重数|geometric and algebraic multiplicities]] are equal. This conclusion does not require irreducibility.
> <!-- bilingual-en:end -->
>
> 设 $P$ 为有限维实矩阵，尺寸 $n\times n$、$n\ge1$，并且满足行随机或列随机条件。对每个满足 $|\lambda|=1$ 的复特征值，它对应的复 Jordan 块都只有一阶。等价地，$\lambda$ 是[[半单特征值]]：它的[[代数重数与几何重数|几何重数与代数重数]]相等。这个结论不要求不可约。

<!-- bilingual-en:start -->
## A two-vector contradiction
<!-- bilingual-en:end -->

*用两个向量看出矛盾*

<!-- bilingual-en:start -->
Use the column-stochastic convention first. For any complex vector $z$, write $\|z\|_1=\sum_j|z_j|$, the sum of its coordinate magnitudes. The [[Markov矩阵谱边界|absolute-sum argument]] gives $\|P^kz\|_1\le\|z\|_1$ for all $k\ge0$, even when $z$ is not a probability vector. Suppose there were a Jordan block longer than one at a value $\lambda$ on the unit circle. The first two vectors of its [[Jordan链|Jordan chain]] would give vectors $v\ne0$ and $w$ with

$$Pv=\lambda v,\qquad Pw=\lambda w+v.$$

Repeated multiplication then gives

$$P^kw=\lambda^kw+k\lambda^{k-1}v\qquad(k\ge1).$$

Each step keeps adding another copy of the preceding chain direction. Since $|\lambda|=1$, the triangle inequality, applied in reverse, yields

$$\|P^kw\|_1\ge k\|v\|_1-\|w\|_1.$$

The right side becomes arbitrarily large because $\|v\|_1>0$, whereas stochasticity bounds the left side by the fixed number $\|w\|_1$. This contradiction excludes every chain of length at least two, hence every Jordan block larger than one.
<!-- bilingual-en:end -->

先采用列随机约定。对任意复向量 $z$，记 $\|z\|_1=\sum_j|z_j|$，即各坐标的模相加。[[Markov矩阵谱边界|绝对值和论证]]给出全部 $k\ge0$ 下 $\|P^kz\|_1\le\|z\|_1$，即使 $z$ 不是概率向量也成立。假设单位圆上的某个 $\lambda$ 对应长度大于一的 Jordan 块，其 [[Jordan链]]的前两个向量就会满足上面的 $Pv=\lambda v$、$Pw=\lambda w+v$，且 $v\ne0$。反复相乘得到 $P^kw=\lambda^kw+k\lambda^{k-1}v$：每一步都会再累加一份前一个链方向。由于 $|\lambda|=1$，反向使用三角不等式，得到 $\|P^kw\|_1\ge k\|v\|_1-\|w\|_1$。右侧因为 $\|v\|_1>0$ 而无限增大，左侧却被固定的 $\|w\|_1$ 界住，矛盾。因此不存在长度至少为二的链，也不存在大于一阶的对应块。

<!-- bilingual-en:start -->
For a row-stochastic matrix, repeat the same proof with $\|z\|_\infty=\max_j|z_j|$. Nonnegative row sums equal to one give $\|P^kz\|_\infty\le\|z\|_\infty$, and the same reverse-triangle lower bound contradicts it. This is the stochastic-matrix instance of the general [[离散系统有界判据|criterion for bounded matrix powers]].
<!-- bilingual-en:end -->

行随机矩阵使用 $\|z\|_\infty=\max_j|z_j|$ 重复同一证明。非负行元素之和为一，保证 $\|P^kz\|_\infty\le\|z\|_\infty$；反向三角不等式给出的同样下界与之矛盾。这正是一般[[离散系统有界判据|矩阵幂有界判据]]在随机矩阵上的应用。

<!-- bilingual-en:start -->
## What the conclusion leaves open
<!-- bilingual-en:end -->

*这个结论还没有决定什么*

<!-- bilingual-en:start -->
Semisimple does not mean simple. For the [[单位矩阵|identity matrix]] $I_2$, eigenvalue one occurs twice and has two independent eigenvectors: it is semisimple but has algebraic multiplicity two. The theorem allows multiple stationary directions. It also allows persistent oscillation: the swap matrix $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ has semisimple eigenvalues $1$ and $-1$ but alternates between itself and $I_2$. Thus it does not settle [[Markov稳态唯一判据|stationary uniqueness]] or [[Markov矩阵幂收敛判据|convergence of the powers]].
<!-- bilingual-en:end -->

半单不等于单根。对[[单位矩阵]] $I_2$，特征值一出现两次，也有两个独立特征向量：它半单，但代数重数为二。所以定理允许多个稳态方向，也允许持续振荡：交换矩阵 $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ 的特征值 $1$、$-1$ 都半单，其幂却在自身和 $I_2$ 之间交替。因此定理没有决定[[Markov稳态唯一判据|稳态是否唯一]]，也没有决定[[Markov矩阵幂收敛判据|矩阵幂是否收敛]]。

> [!example] A longer block can remain inside the disk · 单位圆内部仍可有长块
> <!-- bilingual-en:start -->
> Consider the column-stochastic matrix
>
> $$Q=\begin{pmatrix}1/2&0&0\\1/2&1/2&0\\0&1/2&1\end{pmatrix}.$$
>
> Its [[三角矩阵行列式|triangular determinant]] gives characteristic polynomial $(t-1/2)^2(t-1)$, hence eigenvalues $1/2,1/2,1$. Solving $(Q-\tfrac12I_3)x=0$ gives $x_1=0$ and $x_2=-x_3$, only one independent direction. Thus $1/2$ has a size-two Jordan block, and $Q$ is not diagonalizable. This causes no contradiction: the block's extra factor $k$ is multiplied by $(1/2)^{k-1}$, which tends to zero. The theorem restricts blocks on the unit circle, not all blocks everywhere.
> <!-- bilingual-en:end -->
>
> 考虑上面的列随机矩阵 $Q$。由[[三角矩阵行列式]]可得特征多项式 $(t-1/2)^2(t-1)$，因此特征值为 $1/2,1/2,1$。解 $(Q-\tfrac12I_3)x=0$，得到 $x_1=0$、$x_2=-x_3$，只有一个独立方向。因此 $1/2$ 对应一个二阶 Jordan 块，整个 $Q$ 不能对角化。这并不矛盾：块幂的额外因子 $k$ 还要乘 $(1/2)^{k-1}$，最终趋零。定理限制的是单位圆上的块，不是所有位置上的块。

## 来源与核验

<!-- bilingual-en:start -->
- [Gallager, MIT 6.262, Chapter 3, Exercise 3.15(c), PDF p.44 / printed p.146](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf#page=44): bounds powers of a stochastic matrix to exclude nontrivial Jordan blocks on the unit circle. The direct chain proof above makes the modulus condition explicit. Exercise 3.14, PDF p.43, supplies the transpose of the interior-block example.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]], p.1: powers remain stochastic. [[半单特征值|Semisimple eigenvalues]] and [[Jordan块的幂|Jordan-block powers]] provide the precise terminology and block calculation; all displayed examples and the two-vector estimate are checked directly.
<!-- bilingual-en:end -->

- Gallager，MIT6.262第3章，练习3.15(c)，PDF第44页／印刷146页：核对用随机矩阵幂有界排除单位圆上的非平凡 Jordan 块。正文用直接链证明明确模的条件。练习3.14，PDF第43页，给出内部二阶块例子的转置版本。
- MIT18.06SC第2.11节摘要第1页：核对幂仍为随机矩阵。半单定义与块幂前置提供准确术语及计算；展示的例子和两向量下界均直接核验。
