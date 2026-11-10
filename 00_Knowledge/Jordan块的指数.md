---
aliases:
  - Matrix exponential of a Jordan block
  - Jordan 块的指数
student_os: knowledge-atom
atom_id: LA-EIG-032
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
  - "[[Jordan块]]"
  - "[[矩阵指数]]"
related:
  - "[[Jordan块的幂]]"
  - "[[对角化计算矩阵指数]]"
  - "[[连续系统谱稳定性]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Exponential of a Jordan block
<!-- bilingual-en:end -->

*Jordan 块的矩阵指数*

<!-- bilingual-en:start -->
The [[矩阵指数|matrix exponential]] combines all matrix powers into $e^{tJ}=I+tJ+t^2J^2/2!+\cdots$. For a [[Jordan块|Jordan block]], this infinite definition reduces to a scalar exponential times a finite polynomial matrix. The finite polynomial records movement along the chain. It is exact, not a numerical truncation, and it is not obtained by exponentiating each entry.
<!-- bilingual-en:end -->

[[矩阵指数]]把各次矩阵幂组合成 $e^{tJ}=I+tJ+t^2J^2/2!+\cdots$。对[[Jordan块]]，这个无限级数可以化成标量指数乘以有限多项式矩阵。有限多项式记录沿链的移动；它是精确结果，不是数值截断，也不是对每个元素分别取指数。

> [!example] A chain with no scalar stretch · 没有标量伸缩的链
> <!-- bilingual-en:start -->
> Take $N=\begin{pmatrix}0&1\\0&0\end{pmatrix}$. Multiplication gives $N^2=0$, so every higher term in the defining series vanishes. At $t=3$ the resulting matrix sends $(a,b)^T$ to $(a+3b,b)^T$, where $T$ means [[矩阵转置|transpose]].
> <!-- bilingual-en:end -->
>
> 取 $N=\begin{pmatrix}0&1\\0&0\end{pmatrix}$，相乘得 $N^2=0$，定义级数的所有更高项因此消失。$t=3$ 时，所得矩阵把 $(a,b)^T$ 变成 $(a+3b,b)^T$，其中 $T$ 表示[[矩阵转置|转置]]。
>
> $$e^{tN}=I+tN=\begin{pmatrix}1&t\\0&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
## Factor out the scalar exponential
<!-- bilingual-en:end -->

*先提出标量指数*

<!-- bilingual-en:start -->
Let $J=\lambda I_r+N$ be a size-$r$ block, with integer $r\ge1$, real or complex $\lambda$, and [[单位矩阵|identity]] $I_r$. The shift matrix $N$ has ones on the first superdiagonal and satisfies $N^r=0$. Let $t$ be real or complex; the long-time interpretation below takes real $t\ge0$. Since $\lambda I_r$ commutes with $N$, the convergent series can be regrouped to give
<!-- bilingual-en:end -->

设 $J=\lambda I_r+N$ 是大小为 $r$ 的块：整数 $r\ge1$，$\lambda$ 为实数或复数，$I_r$ 为[[单位矩阵]]。移位矩阵 $N$ 在第一条超对角线上取1，并满足 $N^r=0$。$t$ 可以为实数或复数；后面的长期解释采用实时间 $t\ge0$。由于 $\lambda I_r$ 与 $N$ 可交换，可以重组收敛级数，得到

$$e^{tJ}=e^{t\lambda I_r}e^{tN}=e^{\lambda t}e^{tN}.$$

<!-- bilingual-en:start -->
In the first factor, every matrix power is a scalar times $I_r$, leaving the ordinary scalar exponential. In the second factor, $N^j=0$ for $j\ge r$. Hence
<!-- bilingual-en:end -->

第一个因子中的各次矩阵幂都是标量乘 $I_r$，最终留下普通标量指数；第二个因子中，$j\ge r$ 时 $N^j=0$。所以

> [!important] A finite polynomial times an exponential · 有限多项式乘指数
> <!-- bilingual-en:start -->
> For every $t$, the exact formula is
> <!-- bilingual-en:end -->
>
> 对每个 $t$，精确公式为
>
> $$e^{tJ}=e^{\lambda t}\sum_{j=0}^{r-1}\frac{t^j}{j!}N^j.$$

<!-- bilingual-en:start -->
Here $j!=1\cdot2\cdots j$, with $0!=1$, and $N^0=I_r$. The coefficient on the $j$th superdiagonal is $e^{\lambda t}t^j/j!$. At $t=0$ this gives $I_r$; for $r=1$ it gives the one-entry matrix $[e^{\lambda t}]$. If $\lambda=0$, the scalar factor is $1$, so the whole answer is a polynomial.
<!-- bilingual-en:end -->

这里 $j!=1\cdot2\cdots j$，$0!=1$，且 $N^0=I_r$。第 $j$ 条超对角线上的系数为 $e^{\lambda t}t^j/j!$。$t=0$ 时得到 $I_r$；$r=1$ 时得到只有一项的矩阵 $[e^{\lambda t}]$；$\lambda=0$ 时标量因子为1，整个结果就是多项式。

<!-- bilingual-en:start -->
For size two and three, this reads
<!-- bilingual-en:end -->

二阶与三阶的具体形式为

$$e^{tJ_2(\lambda)}=e^{\lambda t}\begin{pmatrix}1&t\\0&1\end{pmatrix},\qquad e^{tJ_3(\lambda)}=e^{\lambda t}\begin{pmatrix}1&t&t^2/2\\0&1&t\\0&0&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
For instance, $e^{2J_3(-1)}=e^{-2}\begin{pmatrix}1&2&2\\0&1&2\\0&0&1\end{pmatrix}$. The highest polynomial degree in a size-$r$ block is $r-1$, even though some entries and some initial vectors use smaller degrees.
<!-- bilingual-en:end -->

例如 $e^{2J_3(-1)}=e^{-2}\begin{pmatrix}1&2&2\\0&1&2\\0&0&1\end{pmatrix}$。大小为 $r$ 的块包含的最高多项式次数是 $r-1$，但某些元素以及某些初始向量只涉及更低次数。

<!-- bilingual-en:start -->
## Check it without an infinite sum
<!-- bilingual-en:end -->

*不用无限求和也能验算*

<!-- bilingual-en:start -->
Denote the finite sum by $P(t)$. Differentiating its finitely many terms gives $P'(t)=NP(t)$: the would-be final term on the right vanishes because $N^r=0$. Thus $F(t)=e^{\lambda t}P(t)$ satisfies $F'(t)=JF(t)$ and $F(0)=I_r$. These are the identities used by the [[常系数线性系统的矩阵指数解|matrix-exponential solution]] $x(t)=e^{tJ}x(0)$ of $x'=Jx$.
<!-- bilingual-en:end -->

把有限和记作 $P(t)$。逐项求导得到 $P'(t)=NP(t)$：右边原本多出的最后一项因 $N^r=0$ 而消失。因此 $F(t)=e^{\lambda t}P(t)$ 满足 $F'(t)=JF(t)$、$F(0)=I_r$。这些正是[[常系数线性系统的矩阵指数解]] $x(t)=e^{tJ}x(0)$ 所使用的恒等式，对应方程 $x'=Jx$。

<!-- bilingual-en:start -->
## Growth depends on both the scalar and the chain
<!-- bilingual-en:end -->

*增长同时取决于标量与链*

<!-- bilingual-en:start -->
For real $t\to+\infty$, $|e^{\lambda t}|=e^{t\operatorname{Re}\lambda}$, where $\operatorname{Re}\lambda$ means the real part. If it is negative, [[幂函数慢于指数函数|exponential decay dominates every fixed polynomial]], and every entry tends to zero. If it is zero and $r>1$, the first superdiagonal has magnitude $t$, making the matrix family unbounded. For $r=1$ at that boundary, the magnitude remains one. These are block-level facts behind [[连续系统谱稳定性|continuous-time spectral stability]].
<!-- bilingual-en:end -->

对实时间 $t\to+\infty$，有 $|e^{\lambda t}|=e^{t\operatorname{Re}\lambda}$，其中 $\operatorname{Re}\lambda$ 表示实部。若实部为负，由[[幂函数慢于指数函数|指数衰减压过固定多项式]]，每个元素都趋于零。若实部为零且 $r>1$，第一条超对角线的模为 $t$，矩阵族就无界；这个边界上若 $r=1$，模则始终为一。这些是[[连续系统谱稳定性]]背后的块层面事实。

> [!warning] Some trajectories can still be bounded · 有些轨迹仍可以有界
> <!-- bilingual-en:start -->
> For $J_2(0)$, the initial column $(0,1)^T$ evolves to $(t,1)^T$ and grows, whereas $(1,0)^T$ stays fixed. An unbounded matrix family means some initial directions grow; it does not mean every initial vector grows. Also, the factorization of the exponential above uses commuting summands. It is not a general rule for arbitrary matrix sums.
> <!-- bilingual-en:end -->
>
> 对 $J_2(0)$，初始列向量 $(0,1)^T$ 演化为 $(t,1)^T$ 并增长，而 $(1,0)^T$ 保持不动。矩阵族无界意味着存在增长的初始方向，并不意味着每个初始向量都增长。此外，上面的指数分解使用了两个加数可交换的条件，不能作为任意矩阵和的通用规则。

## 来源与核验

<!-- bilingual-en:start -->
- [Stanford EE263, Lecture 12, slides 12–8 to 12–10](https://web.stanford.edu/class/archive/ee/ee263/ee263.1082/lectures/jcf.pdf#page=8): superdiagonal coefficients of the block exponential and polynomial-times-exponential generalized modes.
- [MIT 18.06, Matrix Exponentials, §§3–5, PDF pp.4–6](https://web.mit.edu/18.06/www/Spring17/Matrix-Exponentials.pdf#page=4): power-series definition, derivative identity and the sufficient commutativity condition for factoring exponentials. [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U2_S10_Lecture_Lecture_23_Differential_Equations_and_expAt.pdf|MIT Lecture 23]] also introduces the convergent series.
- [[03_Positive Definite Matrices and Applications#Jordan块的幂与Jordan块的指数|Permanent course calculation]]: course notation and stability scope. [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=1|MIT MCS Corollary 13.7.4]]: exponential domination. Finite differentiation, numerical examples and boundary trajectories above are checked directly.
<!-- bilingual-en:end -->

- Stanford EE263第12讲12–8至12–10页核对超对角系数及多项式乘指数的广义模态。
- MIT矩阵指数讲义第3–5节核对幂级数、导数恒等式及可交换时可以分解指数；Lecture23也介绍该收敛级数。
- 永久课程段落核对记号与稳定性范围；MIT MCS推论13.7.4核对指数压过固定幂。有限求导、数值例子与边界轨迹直接核算。
