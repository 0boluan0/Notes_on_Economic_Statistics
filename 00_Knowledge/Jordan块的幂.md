---
aliases:
  - Powers of a Jordan block
  - Jordan 块的幂
student_os: knowledge-atom
atom_id: LA-EIG-013
atom_set: eigenvalues-linear-dynamics
atom_type: theorem
status: source-checked
requires:
  - "[[Jordan块]]"
related:
  - "[[对角化计算矩阵幂]]"
  - "[[Jordan块的指数]]"
  - "[[矩阵幂趋零判据]]"
  - "[[离散系统谱稳定性]]"
part_of:
  - "[[特征值、对角化与线性动力系统.canvas]]"
---

<!-- bilingual-en:start -->
# Powers of a Jordan block
<!-- bilingual-en:end -->

*Jordan 块的幂*

<!-- bilingual-en:start -->
Taking a matrix power means applying the same linear action repeatedly, using [[矩阵乘法|matrix multiplication]], rather than raising each entry separately. A [[Jordan块|Jordan block]] combines a scalar stretch with movement along a chain. Its powers therefore contain both repeated scalar multiplication and counts of how often a chain step can occur.
<!-- bilingual-en:end -->

矩阵的幂表示按[[矩阵乘法]]反复施加同一个线性作用，而不是把每个元素分别乘方。[[Jordan块]]把标量伸缩与沿链移动结合在一起。因此，它的幂同时包含反复的标量乘法，以及沿链移动可以出现多少次的计数。

> [!example] A shear applied three times · 剪切连续作用三次
> <!-- bilingual-en:start -->
> The block $J=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ sends the column $(a,b)^T$ to $(a+b,b)^T$. Here $T$ denotes [[矩阵转置|transpose]]. The second coordinate stays fixed, while every application adds another copy of $b$ to the first. Thus $J^2(a,b)^T=(a+2b,b)^T$ and $J^3(a,b)^T=(a+3b,b)^T$.
> <!-- bilingual-en:end -->
>
> 块 $J=\begin{pmatrix}1&1\\0&1\end{pmatrix}$ 把列向量 $(a,b)^T$ 变成 $(a+b,b)^T$，其中 $T$ 表示[[矩阵转置|转置]]。第二个坐标不变，每作用一次，第一个坐标就再加一份 $b$。所以 $J^2(a,b)^T=(a+2b,b)^T$，$J^3(a,b)^T=(a+3b,b)^T$。
>
> $$J^0=I,\qquad J^3=\begin{pmatrix}1&3\\0&1\end{pmatrix}.$$

<!-- bilingual-en:start -->
## Separate scaling from shifting
<!-- bilingual-en:end -->

*把伸缩与移位分开*

<!-- bilingual-en:start -->
Write an $r\times r$ block as $J=\lambda I_r+N$, where $r\ge1$ is an integer, $\lambda$ is real or complex, and $I_r$ is the [[单位矩阵|identity]]. The matrix $N$ has ones immediately above the diagonal and zeros elsewhere. Its powers move farther along the chain: $N^j$ has ones on the $j$th superdiagonal for $0\le j<r$, with $N^0=I_r$, and $N^r=0$. This last property is called nilpotence.
<!-- bilingual-en:end -->

把 $r\times r$ 块写成 $J=\lambda I_r+N$：$r\ge1$ 为整数，$\lambda$ 为实数或复数，$I_r$ 是[[单位矩阵]]。$N$ 仅在紧挨主对角线上方取1，其余为零。$N^j$ 表示沿链移动得更远：当 $0\le j<r$ 时，第 $j$ 条超对角线为1，其中 $N^0=I_r$；而 $N^r=0$。最后这个性质称为幂零。

<!-- bilingual-en:start -->
The two parts commute: $(\lambda I_r)N=N(\lambda I_r)=\lambda N$. When [[矩阵乘法分配律|distributing]] the $m$ factors in $(\lambda I_r+N)^m$, choose $j$ positions to contribute $N$. There are [[组合数|$\binom mj$ choices]], and commutativity makes every such product equal to $\lambda^{m-j}N^j$. This is the [[二项式定理|binomial theorem]] applied to these commuting matrices. Terms with $j\ge r$ vanish.
<!-- bilingual-en:end -->

两部分可交换：$(\lambda I_r)N=N(\lambda I_r)=\lambda N$。用[[矩阵乘法分配律|分配律]]展开 $(\lambda I_r+N)^m$ 中的 $m$ 个因子，从中选 $j$ 个位置贡献 $N$，共有[[组合数|$\binom mj$ 种选择]]。交换性使每种乘积都等于 $\lambda^{m-j}N^j$；这就是把[[二项式定理]]用于这对可交换矩阵。$j\ge r$ 的项全部为零。

> [!important] Exact finite formula · 精确的有限公式
> <!-- bilingual-en:start -->
> For every nonnegative integer $m$,
> <!-- bilingual-en:end -->
>
> 对每个非负整数 $m$，
>
> $$J^m=\sum_{j=0}^{\min(m,r-1)}\binom mj\lambda^{m-j}N^j.$$

<!-- bilingual-en:start -->
The upper limit also keeps $m-j\ge0$. The coefficient is $\binom mj=m!/[j!(m-j)!]$, where $k!=1\cdot2\cdots k$ and $0!=1$. Zeroth powers here denote empty products: a scalar contributes $1$ and a matrix contributes $I_r$. Thus $m=0$ gives $J^0=I_r$ even for a zero block. If $\lambda=0$, use the simpler statement $J=N$: its powers become zero from $m=r$ onward.
<!-- bilingual-en:end -->

求和上限还保证 $m-j\ge0$。系数为 $\binom mj=m!/[j!(m-j)!]$，其中 $k!=1\cdot2\cdots k$，$0!=1$。这里的零次幂按空乘积理解：标量贡献1，矩阵贡献 $I_r$。因此 $m=0$ 给出 $J^0=I_r$，零块也一样。若 $\lambda=0$，可以直接使用 $J=N$，从 $m=r$ 起幂就全为零。

<!-- bilingual-en:start -->
## Read the successive superdiagonals
<!-- bilingual-en:end -->

*逐条读超对角线*

<!-- bilingual-en:start -->
For a size-two block and $m\ge1$, only $I$ and $N$ remain. A size-three block additionally has an $N^2$ term. For example, for $J_3(2)$ and $m=3$, the coefficients are $2^3=8$, $3\cdot2^2=12$, and $\binom32 2=6$:
<!-- bilingual-en:end -->

二阶块在 $m\ge1$ 时只留下 $I$ 与 $N$；三阶块还会留下 $N^2$。例如 $J_3(2)$ 的三次幂，各项系数为 $2^3=8$、$3\cdot2^2=12$、$\binom32 2=6$：

$$J_2(\lambda)^m=\begin{pmatrix}\lambda^m&m\lambda^{m-1}\\0&\lambda^m\end{pmatrix}\quad(m\ge1),\qquad J_3(2)^3=\begin{pmatrix}8&12&6\\0&8&12\\0&0&8\end{pmatrix}.$$

<!-- bilingual-en:start -->
At $m=1,\lambda=0$, the superdiagonal is $1$: it is the term with no scalar factors left. At $m=0$, use $I$ directly instead of introducing $\lambda^{-1}$ into the size-two display. Negative powers, when they exist, require a different formula; they are outside the nonnegative-power sum above.
<!-- bilingual-en:end -->

当 $m=1,\lambda=0$ 时，超对角元为1，因为对应项已经没有标量因子。当 $m=0$ 时直接用 $I$，不要往二阶显示式中引入 $\lambda^{-1}$。负次幂即使存在，也需要另一条公式，不属于上述非负次幂和式。

<!-- bilingual-en:start -->
## Why block size affects repeated dynamics
<!-- bilingual-en:end -->

*为什么块大小影响长期反复作用*

<!-- bilingual-en:start -->
For fixed nonzero $\lambda$, the coefficient of $N^j$ is $\lambda^m\lambda^{-j}\binom mj$: an exponential factor times a polynomial in $m$ of degree $j$. Hence, if $0<|\lambda|<1$, every entry tends to zero because [[幂函数慢于指数函数|fixed powers grow more slowly than exponentials]]. The case $\lambda=0$ becomes zero after finitely many steps. These give the block-level explanation of the [[矩阵幂趋零判据|matrix-power convergence criterion]].
<!-- bilingual-en:end -->

固定非零 $\lambda$ 后，$N^j$ 的系数为 $\lambda^m\lambda^{-j}\binom mj$：指数因子乘以关于 $m$ 的 $j$ 次多项式。因此当 $0<|\lambda|<1$ 时，由[[幂函数慢于指数函数]]，每个元素都趋于零；$\lambda=0$ 时则经过有限步直接归零。这给出[[矩阵幂趋零判据]]的块层面解释。

> [!warning] Unit-modulus eigenvalues do not settle boundedness · 模为一还不够
> <!-- bilingual-en:start -->
> If $|\lambda|=1$ and $r>1$, the first superdiagonal has magnitude $m$, so the matrix powers are unbounded. The opening block has only the eigenvalue $1$, yet produces linear growth. Some initial vectors still stay bounded: its first coordinate basis vector is fixed. The claim concerns the matrices, or the existence of growing trajectories, not every initial vector.
> <!-- bilingual-en:end -->
>
> 若 $|\lambda|=1$ 且 $r>1$，第一条超对角线的模为 $m$，矩阵幂就无界。开头的块只有特征值1，却产生线性增长；但有些初始向量仍有界，例如第一个坐标基向量保持不动。这里说的是矩阵幂无界，或存在增长轨迹，并非每个初始向量都增长。

## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/07-Mathematics for Computer Science/MIT-6-042j-pdf.pdf#page=678|MIT Mathematics for Computer Science, Theorem 15.6.4, PDF p.678]]: position-counting proof of the binomial theorem; the displayed commuting-matrix argument and all block calculations are checked directly.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.4sum.pdf#page=4|MIT 18.06SC Session 3.4, p.4]]: Jordan-block structure. [[03_Positive Definite Matrices and Applications#Jordan块的幂与Jordan块的指数|Permanent course calculation]]: the finite-power formula and dynamics scope.
- [[01_Math/07-Mathematics for Computer Science/MIT_OCW_6.042J_Materials/01_Session_Readings/MIT6_042JS15_Session24.pdf#page=1|MIT MCS Session 24, Corollary 13.7.4]]: exponential domination of fixed powers. Unit-circle growth follows directly from the first superdiagonal.
<!-- bilingual-en:end -->

- MIT MCS 定理15.6.4核对选择因子位置的二项式证明；可交换矩阵的推广及所有块算例直接核算。
- MIT Session3.4第4页核对块结构；永久课程对应段落核对有限幂公式与动力系统范围。
- MIT MCS Session24推论13.7.4核对指数压过固定幂；单位圆上的增长直接由第一条超对角线得到。
