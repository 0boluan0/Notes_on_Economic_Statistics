---
aliases:
  - Spectral convergence criterion for Markov matrix powers
  - Markov 矩阵幂收敛判据
student_os: knowledge-atom
atom_id: LA-EIG-042
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
  - "[[Markov矩阵谱边界]]"
  - "[[Markov单位圆特征值半单]]"
  - "[[Markov矩阵必有特征值一]]"
  - "[[矩阵幂趋零判据]]"
related:
  - "[[稳态唯一不推收敛]]"
  - "[[有限链逐步收敛]]"
  - "[[周期链Cesaro平均]]"
leads_to:
  - "[[Markov幂的秩一稳态极限]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# When do Markov matrix powers converge?
<!-- bilingual-en:end -->

*Markov 矩阵幂的收敛判据*

<!-- bilingual-en:start -->
Repeatedly applying a [[Markov矩阵|Markov matrix]] redistributes probability one step at a time. **Convergence of its powers** means that every entry of the transition matrix eventually approaches a fixed number. It allows different starting distributions to approach different limits; requiring one common limit is a further condition.
<!-- bilingual-en:end -->

反复施加 [[Markov矩阵]]，就是一步一步重新分配概率。**矩阵幂收敛**指转移矩阵的每个条目最终都趋近一个固定数。它允许不同的初始分布趋向不同的极限；要求所有初始分布趋向同一个极限，是进一步的条件。

<!-- bilingual-en:start -->
## Settling down can preserve a choice of starting state
<!-- bilingual-en:end -->

*稳定下来，也可以保留初态的区别*

<!-- bilingual-en:start -->
Consider three states and the column-stochastic matrix below. Each column lists the next-state probabilities from one starting state. States 1 and 2 stay where they are; state 3 moves to either of them with equal probability.
<!-- bilingual-en:end -->

考虑三个状态，以及下面这个列随机矩阵。每一列列出从一个出发状态到下一步各状态的概率。状态1、2留在原地；状态3以相同概率进入二者之一。

$$
Q=\begin{pmatrix}1&0&1/2\\0&1&1/2\\0&0&0\end{pmatrix},\qquad
Q\begin{pmatrix}a\\b\\c\end{pmatrix}
=\begin{pmatrix}a+c/2\\b+c/2\\0\end{pmatrix}.
$$

<!-- bilingual-en:start -->
For a probability column, $a,b,c\ge0$ and $a+b+c=1$. After one step there is no probability in state 3, so another step changes nothing: $Q^2=Q$ and $Q^k=Q$ for every integer $k\ge1$. Starting entirely in state 1 or entirely in state 2 gives different limits. Thus convergent matrix powers need not erase the starting choice.
<!-- bilingual-en:end -->

概率列要求 $a,b,c\ge0$ 且 $a+b+c=1$。一步之后，状态3的概率已经为零，再走一步不会改变任何东西：$Q^2=Q$，因此对每个整数 $k\ge1$ 都有 $Q^k=Q$。完全从状态1出发与完全从状态2出发，极限不同。所以矩阵幂收敛不一定消除初态的区别。

<!-- bilingual-en:start -->
## Check every eigenvalue, including complex ones
<!-- bilingual-en:end -->

*检查全部特征值，包括复特征值*

<!-- bilingual-en:start -->
Let $P$ be a real $n\times n$ matrix, with integer $n\ge1$, nonnegative entries, and each column summing to one. Under this [[Markov矩阵左右约定|column convention]], a probability column $p_0$ evolves as $p_k=P^kp_0$. The exponent $k$ counts repeated matrix multiplication, and $P^0=I_n$, the [[单位矩阵|identity matrix]]. We write $P^k\to L$ when each entry converges to the corresponding entry of $L$.
<!-- bilingual-en:end -->

设 $P$ 是实 $n\times n$ 矩阵，整数 $n\ge1$，元素非负且每列之和为一。在这一[[Markov矩阵左右约定|列随机约定]]下，概率列 $p_0$ 经过 $k$ 步变成 $p_k=P^kp_0$。指数 $k$ 计算重复矩阵相乘的次数，$P^0=I_n$ 是[[单位矩阵]]。$P^k\to L$ 表示每个条目都收敛到 $L$ 中对应的条目。

<!-- bilingual-en:start -->
An [[特征对|eigenvalue]] $\lambda$ has a nonzero vector $v$ satisfying $Pv=\lambda v$. Even though $P$ is real, we check this equation over the complex numbers. For $\lambda=a+ib$, where $i^2=-1$, its magnitude is $|\lambda|=\sqrt{a^2+b^2}$. The [[Markov矩阵谱边界|stochastic-matrix spectral bound]] already gives $|\lambda|\le1$, and [[Markov矩阵必有特征值一|one is always an eigenvalue]]. The remaining question is what happens on the boundary $|\lambda|=1$.
<!-- bilingual-en:end -->

[[特征对|特征值]] $\lambda$ 对应非零向量 $v$，满足 $Pv=\lambda v$。即使 $P$ 是实矩阵，也要在复数范围检查这个方程。若 $\lambda=a+ib$、$i^2=-1$，其模是 $|\lambda|=\sqrt{a^2+b^2}$。[[Markov矩阵谱边界]]已经给出 $|\lambda|\le1$，且[[Markov矩阵必有特征值一|一必定是特征值]]。剩下要判断的是边界 $|\lambda|=1$ 上的情况。

> [!important] Only the value one may remain on the unit circle · 单位圆上只允许留下数值一
> <!-- bilingual-en:start -->
> For a finite row- or column-stochastic matrix,
>
> $$P^k\text{ converges}\quad\Longleftrightarrow\quad
> |\lambda|<1\ \text{for every eigenvalue }\lambda\ne1.$$
>
> The value $1$ may occur more than once in the characteristic polynomial. There is no assumption of irreducibility or of a complete eigenvector basis.
> <!-- bilingual-en:end -->
>
> 对有限行随机或列随机矩阵，矩阵幂收敛当且仅当每个不等于一的特征值都满足 $|\lambda|<1$。数值一在特征多项式中可以重复出现；不要求不可约，也不要求存在一整组特征向量基。

<!-- bilingual-en:start -->
## Why the condition is sufficient
<!-- bilingual-en:end -->

*为什么条件足够*

<!-- bilingual-en:start -->
A [[Jordan标准形|Jordan basis over the complex numbers]] separates the action into blocks for different eigenvalues. A block can include the extra coupling described by a [[Jordan链|Jordan chain]], so using these blocks does not assume that $P$ is diagonalizable. For stochastic matrices, [[Markov单位圆特征值半单|every unit-circle eigenvalue is semisimple]]: its blocks all have size one. In particular, the blocks at $\lambda=1$ simply leave their coordinates unchanged. This is the local meaning of [[半单特征值|semisimplicity]], even if the value one repeats.
<!-- bilingual-en:end -->

在复数域取 [[Jordan标准形|Jordan 基]]，可以把矩阵作用分成对应于不同特征值的块。块内部可能带有 [[Jordan链]]描述的额外耦合，因此使用这种分解没有假设 $P$ 可以对角化。对随机矩阵，[[Markov单位圆特征值半单|单位圆上的每个特征值都半单]]：对应块的尺寸都是一。特别地，$\lambda=1$ 的块只是让自己的坐标保持不变。这就是[[半单特征值]]在局部的含义，即使数值一重复出现也一样。

<!-- bilingual-en:start -->
Every other block has $|\lambda|<1$. Its powers tend to zero: for $\lambda\ne0$, the [[Jordan块的幂|block-power formula]] gives fixed-degree polynomial factors in $k$ multiplied by decaying powers of $\lambda$; when $\lambda=0$, a sufficiently high power is exactly zero. This is the mechanism behind the [[矩阵幂趋零判据|vanishing-power criterion]]. Larger Jordan blocks **inside** the disk are therefore allowed. In this basis, the limiting matrix is identity on the blocks at one and zero on the remaining blocks; changing back to the original coordinates gives a limit $L$.
<!-- bilingual-en:end -->

其余块都满足 $|\lambda|<1$，所以它们的幂趋于零：当 $\lambda\ne0$ 时，[[Jordan块的幂|块幂公式]]给出关于 $k$ 的固定次数多项式因子乘上衰减的 $\lambda$ 的幂；当 $\lambda=0$ 时，足够高的幂直接为零。这正是[[矩阵幂趋零判据]]的机制。因此，单位圆**内部**仍然允许较大的 Jordan 块。在这组基中，极限在特征值一的块上是恒等，在其余块上为零；换回原坐标，就得到极限 $L$。

<!-- bilingual-en:start -->
## Why another boundary eigenvalue prevents convergence
<!-- bilingual-en:end -->

*为什么其他边界特征值会阻止收敛*

<!-- bilingual-en:start -->
Suppose $Pv=\lambda v$ with $v\ne0$, $|\lambda|=1$, and $\lambda\ne1$. Repeated application gives $P^kv=\lambda^kv$. If $P^k$ converged, this vector sequence would converge, so the difference between consecutive terms would tend to zero. But
<!-- bilingual-en:end -->

假设 $Pv=\lambda v$，其中 $v\ne0$、$|\lambda|=1$、$\lambda\ne1$。反复施加矩阵得到 $P^kv=\lambda^kv$。如果 $P^k$ 收敛，这个向量序列也会收敛，相邻两项之差就应该趋于零。然而

$$
P^{k+1}v-P^kv=\lambda^k(\lambda-1)v.
$$

<!-- bilingual-en:start -->
Choose a nonzero coordinate $v_j$. The magnitude of that coordinate in the difference is always $|\lambda-1|\,|v_j|>0$, because $|\lambda^k|=1$. This is impossible for a convergent sequence. A complex $v$ is legitimate here: convergence of all real entries of $P^k$ also implies convergence after multiplying by any fixed complex vector. Checking only real eigenvalues could miss a persistent rotation. For $\lambda=-1$, the obstruction is the familiar alternating sign.
<!-- bilingual-en:end -->

选取一个非零坐标 $v_j$。由于 $|\lambda^k|=1$，差向量在这个坐标上的模始终为 $|\lambda-1|\,|v_j|>0$，不可能趋于零。这里允许使用复向量 $v$：$P^k$ 的全部实条目收敛，同样保证它乘任何固定复向量之后收敛。若只检查实特征值，就可能漏掉持续旋转；$\lambda=-1$ 则是熟悉的交替变号。

<!-- bilingual-en:start -->
## The limit keeps exactly the fixed-point component
<!-- bilingual-en:end -->

*极限恰好保留固定点部分*

<!-- bilingual-en:start -->
Write $E_1(P)=\{x\in\mathbb R^n:Px=x\}$, the [[特征空间|eigenspace]] for one, including zero. If $P^k\to L$, then $PL=LP=L$ by taking limits in $PP^k=P^kP=P^{k+1}$. Hence every output $Lx$ is fixed by $P$. Conversely, if $Px=x$, then $P^kx=x$ at every step and therefore $Lx=x$. These observations give
<!-- bilingual-en:end -->

记 $E_1(P)=\{x\in\mathbb R^n:Px=x\}$，即包含零向量的特征值一的[[特征空间]]。若 $P^k\to L$，对 $PP^k=P^kP=P^{k+1}$ 取极限，得到 $PL=LP=L$，所以每个输出 $Lx$ 都被 $P$ 固定。反过来，若 $Px=x$，则每一步都有 $P^kx=x$，因此 $Lx=x$。合起来得到

$$
\operatorname{range}(L)=E_1(P),\qquad L^2=L,\qquad
\operatorname{rank}(L)=\dim E_1(P).
$$

<!-- bilingual-en:start -->
Here the [[列空间|range]] is the set of possible outputs, and [[矩阵秩|rank]] is its dimension. Thus $L$ is an [[幂等矩阵|idempotent projection]]: it keeps fixed vectors and removes the components belonging to the other generalized eigenspaces, the directions built from [[广义特征向量|generalized eigenvectors]]. This particular choice of what to retain and remove is called the **spectral projection** at one. Projection here does not assert perpendicularity; for the standard real inner product it would be [[正交投影矩阵判别|orthogonal only if $L^T=L$]], where $T$ denotes [[矩阵转置|transpose]]. The opening $Q$ is already a nonsymmetric projection of rank two.
<!-- bilingual-en:end -->

这里[[列空间|值域]]指全部可能输出，[[矩阵秩|秩]]是它的维数。因此 $L$ 是一个[[幂等矩阵|幂等投影]]：保留固定向量，消去其余广义特征空间的分量，即由[[广义特征向量]]构成的相应方向。按这种方式选择保留和消去的部分，称为特征值一处的**谱投影**。此处“投影”并不保证垂直；在标准实内积下，还需 $L^T=L$ [[正交投影矩阵判别|才是正交投影]]，其中 $T$ 表示[[矩阵转置]]。开头的 $Q$ 已经是一个不对称、秩为二的投影。

<!-- bilingual-en:start -->
The limit remains nonnegative with each column summing to one, so its columns are [[Markov稳态分布|stationary probability distributions]]. They need not be equal. If the stationary distribution is unique, they all coincide and the projection has rank one; that further conclusion is developed in [[Markov幂的秩一稳态极限|the rank-one stationary limit]]. With a row-stochastic matrix, the same convergence criterion applies; transpose the entire description to interpret probability rows.
<!-- bilingual-en:end -->

极限仍然元素非负、每列之和为一，所以各列都是[[Markov稳态分布|稳态概率分布]]，但不一定相同。若稳态唯一，各列才会相同，投影才为秩一；这一进一步结论见[[Markov幂的秩一稳态极限]]。行随机矩阵也适用同一收敛判据；解释概率行时，把整套描述一起转置即可。

> [!question]- Does the identity converge? · 单位矩阵的幂收敛吗？
> <!-- bilingual-en:start -->
> For $P=I_2$, every power equals $I_2$, so the powers converge. The eigenvalue one repeats twice and every probability vector is stationary. The limit has rank two, not one. At the other extreme, a unique stationary distribution by itself [[稳态唯一不推收敛|does not guarantee convergence]] if another unit-circle eigenvalue remains.
> <!-- bilingual-en:end -->
>
> 对 $P=I_2$，每次幂都等于 $I_2$，因此收敛。特征值一重复两次，每个概率向量都平稳，极限的秩为二而非一。另一个边界是：如果仍有其他单位圆特征值，仅有唯一稳态也[[稳态唯一不推收敛|不能保证收敛]]。

<!-- bilingual-en:start -->
## Interactive exploration
<!-- bilingual-en:end -->

<!-- visual-lab:links:start -->
## 交互理解

- [[07_Programme/可视化实验室/实验/gnn|图神经网络的邻居聚合]]
<!-- visual-lab:links:end -->

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Levin–Peres with Wilmer, *Markov Chains and Mixing Times*, 2nd ed., Lemma 12.1(i), PDF p.177 / printed p.161](https://pages.uoregon.edu/dlevin/MARKOV/mcmt2e.pdf#page=177): the complex spectral bound for a finite stochastic matrix. Proposition 1.32, PDF p.36, distinguishes a stationary limit of averages from convergence of the unaveraged powers.
- [Axler, *Linear Algebra Done Right*, 4th ed., §8B and §8C, PDF pp.322–323 and 336–338](https://linear.axler.net/LADR4e.pdf#page=322): generalized-eigenspace decomposition and Jordan bases over the complex numbers. Combined with the bounded-power argument in [[Markov单位圆特征值半单|semisimplicity for Markov matrices]], these support the block proof given here, without assuming diagonalizability.
- [Gallager, *Discrete Stochastic Processes*, Chapter 3, §§3.3.4–3.4, PDF pp.15–20 / printed pp.117–122](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3558b08622765d26c2b0a7d2eeeac885_MIT6_262S11_chap03.pdf#page=15): limiting transition matrices, multiple recurrent classes, and spectral modes. The source uses probability rows; the column proof and all displayed examples above are checked directly.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses2.11sum.pdf|MIT 18.06SC Session 2.11 summary]], PDF pp.1–3, and [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S11_Lecture_Lecture_24_Markov_Matrices_Fourier_Series.pdf|Lecture 24 transcript]], PDF pp.1–7: course scope, the column convention, and fixed versus decaying modes. Strict decay of all other modes is a condition here, not a consequence of stochasticity alone.
- [Nick Higham, *What Is a Stochastic Matrix?*](https://nhigham.com/2022/12/13/what-is-a-stochastic-a-matrix/): supplementary exposition of the boundary condition and the identity/swap examples; the textbook arguments above carry the proof.
<!-- bilingual-en:end -->

- Levin–Peres 与 Wilmer，第2版引理12.1(i)，PDF第177页 / 印刷第161页：有限随机矩阵的复谱边界。命题1.32，PDF第36页，区分平均序列的稳态极限与未取平均的矩阵幂收敛。
- Axler 第4版 §8B、§8C，PDF第322–323、336–338页：复数域的广义特征空间分解与 Jordan 基；结合[[Markov单位圆特征值半单]]的幂有界论证，支持本文不预设可对角化的块证明。
- Gallager 第3章 §§3.3.4–3.4，PDF第15–20页 / 印刷第117–122页：转移矩阵极限、多个常返类与谱模式。原文采用概率行；本文的列约定证明和全部展示例子直接核验。
- MIT Session 2.11摘要第1–3页及 Lecture 24讲稿第1–7页：课程范围、列约定、固定模式与衰减模式。“其余模式严格衰减”在这里是条件，并非随机性本身自动保证。
- Higham 文章作为边界条件及单位矩阵、交换矩阵例子的补充说明；核心证明由上述教材论证承担。
