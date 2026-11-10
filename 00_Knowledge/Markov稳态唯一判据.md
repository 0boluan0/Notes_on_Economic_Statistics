---
aliases:
- Stationary distribution uniqueness
- Algebraic uniqueness criterion for stationary distributions
- Markov 稳态唯一性判据
student_os: knowledge-atom
atom_id: LA-EIG-041
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
- '[[Markov稳态分布]]'
- '[[有限Markov链稳态存在]]'
- '[[零空间]]'
- '[[特征空间]]'
related:
- '[[有限链平稳分解]]'
- '[[Markov单位圆特征值半单]]'
- '[[有限不可约链稳态]]'
- '[[稳态唯一不推收敛]]'
part_of:
- '[[特征值、对角化与线性动力系统.canvas]]'
---

<!-- bilingual-en:start -->
# Uniqueness of a stationary distribution
<!-- bilingual-en:end -->

*Markov 稳态分布的唯一性*

<!-- bilingual-en:start -->
A transition rule can admit one balanced probability distribution, or many. **Uniqueness** means that there is exactly one way to assign nonnegative probabilities summing to one that remain unchanged after a step. This card explains how to test uniqueness by solving a linear system, and why the test still works when some states have zero stationary probability.
<!-- bilingual-en:end -->

同一套转移规则可能只有一种平衡的概率分配，也可能有许多种。**唯一性**是指：非负、总和为一、再走一步仍不变的概率分配，恰好只有一个。本卡解释怎样用线性方程组判断唯一性，以及为什么有些状态的稳态概率为零时，判据仍然成立。

<!-- bilingual-en:start -->
## From a balance equation to a probability distribution
<!-- bilingual-en:end -->

*先解平衡方程，再归一化*

<!-- bilingual-en:start -->
Use the [[Markov矩阵左右约定|column convention]] for a finite [[Markov矩阵|Markov matrix]] $P$: its real entries are nonnegative and each column sums to one. There are $n\ge1$ states, and $P_{ij}$ is the probability of moving from state $j$ to state $i$. A [[Markov稳态分布|stationary distribution]] is a column $\pi$ satisfying
<!-- bilingual-en:end -->

对有限[[Markov矩阵]] $P$，本卡采用[[Markov矩阵左右约定|列随机约定]]：实元素非负，每列和为一。状态数 $n\ge1$，$P_{ij}$ 表示从状态 $j$ 到状态 $i$ 的概率。[[Markov稳态分布|稳态分布]]是满足下式的列 $\pi$：

$$
P\pi=\pi,\qquad \pi_i\ge0,\qquad \sum_{i=1}^{n}\pi_i=1.
$$

<!-- bilingual-en:start -->
For example, let $P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix}$. Writing $\pi=(a,b)^T$, where $T$ denotes [[矩阵转置|transpose]], the first balance equation gives $0.9a+0.2b=a$, or $a=2b$. The second gives the same restriction. Normalization $a+b=1$ then gives $\pi=(2/3,1/3)^T$. Before normalization, the real fixed vectors form the whole line $\{t(2,1)^T:t\in\mathbb R\}$; most vectors on that line are not probability distributions.
<!-- bilingual-en:end -->

例如取 $P=\begin{bmatrix}0.9&0.2\\0.1&0.8\end{bmatrix}$。把 $\pi$ 写成 $(a,b)^T$，其中 $T$ 表示[[矩阵转置]]。第一个平衡方程给出 $0.9a+0.2b=a$，即 $a=2b$；第二个方程给出同一限制。再用 $a+b=1$ 归一化，就得到 $\pi=(2/3,1/3)^T$。归一化之前，实固定向量组成整条直线 $\{t(2,1)^T:t\in\mathbb R\}$；这条线上多数向量并不是概率分布。

> [!important] A one-dimensional fixed-point space · 一维固定点空间
> <!-- bilingual-en:start -->
> Every finite Markov matrix has [[有限Markov链稳态存在|at least one stationary distribution]]. Under the column convention, it has **exactly one** if and only if
> <!-- bilingual-en:end -->
>
> 每个有限 Markov 矩阵都[[有限Markov链稳态存在|至少存在一个稳态分布]]。在列随机约定下，稳态分布**唯一**当且仅当
>
> $$\dim\ker(P-I_n)=1.$$
>
> <!-- bilingual-en:start -->
> Here $I_n$ is the [[单位矩阵|identity matrix]], $\ker(P-I_n)$ is the [[零空间|nullspace]] consisting of all real solutions of $(P-I_n)v=0$, and its [[维数|dimension]] counts independent fixed directions. It is also the [[特征空间|eigenspace]] for eigenvalue $1$. For a row-stochastic matrix $R$, use $\ker(R^T-I_n)$ for stationary columns instead.
> <!-- bilingual-en:end -->
>
> 这里 $I_n$ 是[[单位矩阵]]，$\ker(P-I_n)$ 是由 $(P-I_n)v=0$ 的全部实解组成的[[零空间]]，其[[维数]]计算独立固定方向的数量。它也是特征值 $1$ 的[[特征空间]]。若用行随机矩阵 $R$，稳态列对应的是 $\ker(R^T-I_n)$。

<!-- bilingual-en:start -->
## Why signed fixed vectors cannot hide extra dimensions
<!-- bilingual-en:end -->

*为什么带正负号的固定向量不会隐藏额外维数*

<!-- bilingual-en:start -->
One direction is immediate: if the fixed-point space is one-dimensional, any two stationary distributions are scalar multiples. Both have total mass one, so that scalar must be one. Existence ensures that there is a distribution to normalize.
<!-- bilingual-en:end -->

一个方向很直接：固定点空间若为一维，任意两个稳态分布都互为倍数。两者总质量都为一，因此倍数只能为一。存在性保证确实有一个分布可以归一化。

<!-- bilingual-en:start -->
The converse needs care: an extra fixed vector might contain negative entries. Let $v$ be any real fixed vector, so $Pv=v$, and let $|v|$ mean taking the absolute value of each component. Nonnegative matrix entries and the triangle inequality give a componentwise inequality
<!-- bilingual-en:end -->

反方向需要多想一步：多出来的固定向量可能带负分量。任取实固定向量 $v$，满足 $Pv=v$；用 $|v|$ 表示逐个分量取绝对值。矩阵元素非负，再用三角不等式，可逐个分量得到

$$
|v|=|Pv|\le P|v|.
$$

<!-- bilingual-en:start -->
But multiplying by a column-stochastic matrix preserves the sum of components. Thus $P|v|$ and $|v|$ have the same total sum. Their difference has only nonnegative entries and sums to zero, so every entry of that difference is zero: $P|v|=|v|$.
<!-- bilingual-en:end -->

然而，乘列随机矩阵会保持分量总和。因此 $P|v|$ 与 $|v|$ 的总和相同；两者之差的分量都非负，总和却为零，每个分量就只能为零。于是 $P|v|=|v|$。

<!-- bilingual-en:start -->
Now split $v$ into its nonnegative positive and negative parts:
<!-- bilingual-en:end -->

现在把 $v$ 拆成两个非负向量，即它的正部与负部：

$$
v^+=\frac{|v|+v}{2},\qquad v^-=\frac{|v|-v}{2},\qquad v=v^+-v^-.
$$

<!-- bilingual-en:start -->
Both parts are fixed by $P$, because both $v$ and $|v|$ are fixed. Every nonzero part has a positive sum, and dividing by that sum makes it a stationary distribution. If the stationary distribution is unique, both nonzero parts must therefore be multiples of the same $\pi$; a zero part is already its zero multiple. Consequently every fixed $v$ is a multiple of $\pi$. The fixed-point space is exactly one-dimensional. This argument does not require every entry of $\pi$ to be positive.
<!-- bilingual-en:end -->

由于 $v$ 和 $|v|$ 都固定，两个部分也都被 $P$ 固定。任何非零部分都有正的分量总和，除以该总和就成为稳态分布。若稳态唯一，每个非零部分就只能是同一个 $\pi$ 的倍数；零部分本来就是它的零倍。因此每个固定向量 $v$ 都是 $\pi$ 的倍数，固定点空间恰好为一维。这个证明没有要求 $\pi$ 的每个分量都为正。

<!-- bilingual-en:start -->
## One closed class is enough
<!-- bilingual-en:end -->

*只需一个闭沟通类，不必整个链不可约*

<!-- bilingual-en:start -->
There is also a graph interpretation. A [[马氏链沟通类|communicating class]] groups states that can reach one another through positive-probability paths. A [[闭沟通类|closed class]] has no possible transition to a state outside it. For a finite chain, a stationary distribution is unique **if and only if there is exactly one closed communicating class**. Other states may feed into that class.
<!-- bilingual-en:end -->

也可以从转移图理解这个结论。[[马氏链沟通类|沟通类]]把能沿正概率路径互相到达的状态归在一起；[[闭沟通类]]一旦进入，就不会跳到类外。对有限链，稳态分布唯一**当且仅当恰好有一个闭沟通类**。其他状态仍然可以通向这个闭类。

<!-- bilingual-en:start -->
Why do closed classes matter? Each closed class supports its own stationary distribution, extended by zeros outside the class. Hence two closed classes already give two different stationary distributions. Conversely, stationary probability cannot remain outside all closed classes: in a finite chain there is a common finite number of steps and a positive lower bound on the probability of reaching a closed class from any outside state. Repeated blocks of those steps drive the outside mass to zero. Within one closed communicating class, irreducibility gives a unique stationary distribution. These facts yield the equivalence; the full description when there are several classes is developed in [[有限链平稳分解|finite-chain stationary decomposition]].
<!-- bilingual-en:end -->

闭类为什么关键？每个闭类本身都支持一个稳态分布，把类外坐标补零，便得到整个链的稳态。因此只要有两个闭类，就至少有两个不同稳态。反过来，稳态概率不能留在全部闭类之外：链是有限的，所以从任意类外状态出发，都能在某个统一的有限步数内，以一个统一的正概率下界到达某个闭类。不断重复这样长度的时间段，类外质量就趋于零。在一个闭沟通类内部，不可约性保证稳态唯一。合起来便得到该等价；多个闭类时的完整描述见[[有限链平稳分解]]。

> [!example] Reducible, but with a unique stationary distribution · 可约仍可以稳态唯一
> <!-- bilingual-en:start -->
> For $P=\begin{bmatrix}1&1\\0&0\end{bmatrix}$, either starting state moves to state $1$ in one step. State $1$ is the only closed class; state $2$ can leave and never return. The fixed equation forces the second coordinate to be zero, so $\ker(P-I_2)=\{(a,0)^T:a\in\mathbb R\}$ and the unique stationary distribution is $(1,0)^T$.
> <!-- bilingual-en:end -->
>
> 对 $P=\begin{bmatrix}1&1\\0&0\end{bmatrix}$，无论从哪个状态出发，一步后都在状态 $1$。状态 $1$ 是唯一闭类，状态 $2$ 离开后就无法返回。固定方程迫使第二坐标为零，所以 $\ker(P-I_2)=\{(a,0)^T:a\in\mathbb R\}$，唯一稳态为 $(1,0)^T$。

<!-- bilingual-en:start -->
An [[不可约链|irreducible chain]] has all states in one communicating class, so finite irreducibility is sufficient for uniqueness; the example shows it is not necessary. At the other extreme, $P=I_n$ with $n\ge2$ has one closed singleton class per state, the entire space as its fixed-point space, and every probability vector as a stationary distribution.
<!-- bilingual-en:end -->

[[不可约链]]把全部状态放在同一个沟通类内，所以有限不可约是唯一性的充分条件；上例表明它并非必要条件。另一端的例子是 $P=I_n$，其中 $n\ge2$：每个状态单独成为一个闭类，整个空间都是固定点空间，每个概率向量都平稳。

<!-- bilingual-en:start -->
## What eigenvalue simplicity adds—and what it does not
<!-- bilingual-en:end -->

*单根的等价说法，以及它没有保证的事*

<!-- bilingual-en:start -->
For finite Markov matrices, [[Markov单位圆特征值半单|eigenvalues on the unit circle are semisimple]]. In particular, at $1$ the eigenspace dimension equals the root multiplicity in the characteristic polynomial; this is the meaning of [[半单特征值|semisimplicity]]. Thus uniqueness is also equivalent to $1$ being an algebraically simple root. The implication from a one-dimensional eigenspace to a simple root uses semisimplicity for stochastic matrices and is false for arbitrary square matrices. Finally, [[稳态唯一不推收敛|uniqueness does not imply convergence]] from every starting distribution: other eigenvalues, such as $-1$, can still sustain oscillation.
<!-- bilingual-en:end -->

对有限 Markov 矩阵，[[Markov单位圆特征值半单|单位圆上的特征值都是半单的]]。特别地，$1$ 的特征空间维数等于它在特征多项式中的根重数；这就是[[半单特征值]]的含义。因此唯一性也等价于 $1$ 是代数单根。“一维特征空间推出单根”在这里依靠随机矩阵在一处的半单性，不能无条件套给任意方阵。最后，[[稳态唯一不推收敛|唯一性没有保证从每个初始分布出发都收敛]]：$-1$ 等其他特征值仍可能维持振荡。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Levin and Peres, with Wilmer, *Markov Chains and Mixing Times*, 2nd ed.](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=25), §1.5 printed pp.9–13 / PDF pp.25–29 and §1.7 printed pp.15–17 / PDF pp.31–33, especially Corollary 1.17 and Propositions 1.28–1.29: existence, irreducible uniqueness, zero stationary mass outside closed classes, and uniqueness exactly for a single closed class. The source uses probability rows; this card consistently transposes to columns. The signed-vector proof above is given directly, so the algebraic criterion does not depend on accepting the decomposition card first.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|MIT 18.06 Lecture 24 transcript]], PDF pp.1–7: the column convention and the connection between stationary vectors and eigenvalue $1$. The nullspace here is explicitly that of $P-I_n$.
- [[Markov单位圆特征值半单|Semisimplicity of unit-circle Markov eigenvalues]] supplies the bounded-power argument used for the simple-root reformulation. All displayed examples and the positive/negative-part proof are checked directly.
<!-- bilingual-en:end -->

- Levin、Peres 与 Wilmer，《Markov Chains and Mixing Times》第2版，§1.5 印刷第9–13页 / PDF第25–29页，§1.7 印刷第15–17页 / PDF第31–33页，特别是推论1.17、命题1.28–1.29：支持存在性、不可约时唯一、闭类之外的稳态质量为零，以及唯一性恰好对应一个闭类。原书用概率行，本卡统一转为概率列。正文直接给出带正负号向量的证明，所以代数判据不必以先接受分解卡为前提。
- MIT 18.06 Lecture 24讲稿，PDF第1–7页：支持列随机约定和稳态向量与特征值一的联系；本卡明确求解的是 $P-I_n$ 的零空间。
- [[Markov单位圆特征值半单]]给出单根改写所需的幂有界论证。全部展示例子以及正负部分证明均直接核验。
