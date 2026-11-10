---
aliases:
- Cholesky factorization criterion
- SPD Cholesky factorization
student_os: knowledge-atom
atom_id: LA-SPD-016
atom_set: symmetric-positive-definite
atom_type: equivalence
status: source-checked
requires:
- '[[LDL正定判别]]'
- '[[正定矩阵]]'
related:
- '[[半正定主平方根]]'
- '[[LDL零主元边界]]'
part_of:
- '[[对称矩阵与正定二次型.canvas]]'
---

<!-- bilingual-en:start -->
# The Cholesky criterion
<!-- bilingual-en:end -->
*Cholesky 正定判据*

<!-- bilingual-en:start -->
For real symmetric matrices, Cholesky factorization expresses a positive-definite matrix as a triangular matrix times its transpose. In the associated quadratic expression, this turns the value into a sum of ordinary squares. The triangular factor provides both a test for positivity and a convenient way to solve linear equations.
<!-- bilingual-en:end -->

在实对称情形，Cholesky 分解把正定矩阵写成一个三角矩阵乘以它的转置。在对应的二次表达式里，这会把函数值变成普通平方项之和。因此，三角因子既可以检验正定性，也便于求解线性方程。

<!-- bilingual-en:start -->
Start with the same matrix used in the [[LDL正定判别|positive-pivot criterion]]:
<!-- bilingual-en:end -->

先看 [[LDL正定判别]]中使用的同一个矩阵：

$$A=\begin{pmatrix}4&2\\2&5\end{pmatrix},\qquad
L=\begin{pmatrix}2&0\\1&2\end{pmatrix},\qquad
LL^T=\begin{pmatrix}4&2\\2&5\end{pmatrix}=A.$$

<!-- bilingual-en:start -->
The symbol $T$ denotes [[矩阵转置|transpose]]. The factor $L$ is [[三角矩阵|lower triangular]], with zeros above the main diagonal, and its diagonal entries $2,2$ are positive. For a column $x=(s,t)^T$,
<!-- bilingual-en:end -->

$T$ 表示[[矩阵转置|转置]]。因子 $L$ 是[[三角矩阵|下三角矩阵]]，主对角线上方全零，对角元 $2,2$ 都正。对列向量 $x=(s,t)^T$，

$$x^TAx=(L^Tx)^T(L^Tx)=(2s+t)^2+(2t)^2.$$

<!-- bilingual-en:start -->
This is positive unless $s=t=0$. It proves that $A$ is [[正定矩阵|positive definite]], meaning that every nonzero real input gives a positive quadratic value. Unlike the unit-diagonal $L$ in an $LDL^T$ decomposition, a Cholesky factor generally does not have diagonal entries $1$; the square coefficients have been absorbed into it.
<!-- bilingual-en:end -->

除非 $s=t=0$，这个和严格为正，证明 $A$ [[正定矩阵|正定]]，即每个非零实输入都得到正二次值。与 $LDL^T$ 中对角元为一的 $L$ 不同，Cholesky 因子的对角元一般不为一，因为平方项的系数已经吸收进它里面。

<!-- bilingual-en:start -->
## The exact existence and uniqueness statement
<!-- bilingual-en:end -->
*精确的存在唯一性结论*

<!-- bilingual-en:start -->
Let $A$ be a [[实对称矩阵|real symmetric]] $n\times n$ matrix, with finite integer $n\ge1$. The criterion is
<!-- bilingual-en:end -->

设 $A$ 是 $n\times n$ [[实对称矩阵]]，$n\ge1$ 为有限整数。判据为

> [!important] Cholesky criterion · Cholesky 判据
> $$A\succ0\quad\Longleftrightarrow\quad
> A=LL^T\text{ for a lower triangular }L\text{ with }L_{ii}>0\text{ for every }i.$$
> <!-- bilingual-en:start -->
> For the given matrix in the given coordinate order, this $L$ is unique. Equivalently, $A=R^TR$ with $R=L^T$ upper triangular and positive on the diagonal. Positive diagonal entries make these factors invertible.
> <!-- bilingual-en:end -->
>
> 对给定坐标次序下的给定矩阵，这个 $L$ 唯一。等价地，$A=R^TR$，其中 $R=L^T$ 是正对角上三角矩阵。对角元严格正保证这些因子可逆。

<!-- bilingual-en:start -->
For a complex [[Hermitian 矩阵|Hermitian]] matrix, the corresponding statement is $A\succ0$ if and only if $A=LL^*$ with $L$ lower triangular and **real positive** diagonal entries. The star means conjugate transpose. Its quadratic value is $z^*Az=\sum_i|(L^*z)_i|^2$, where $|w|^2=\overline w w$. The real-positive diagonal convention also fixes the complex phases and makes the factor unique.
<!-- bilingual-en:end -->

对复 [[Hermitian 矩阵]]，相应结论是：$A\succ0$ 当且仅当 $A=LL^*$，其中 $L$ 下三角且对角元为**正实数**；星号表示共轭转置。二次值为 $z^*Az=\sum_i|(L^*z)_i|^2$，其中 $|w|^2=\overline w w$。正实对角元的约定也固定了复数相位，使因子唯一。

<!-- bilingual-en:start -->
## Building the factor and proving the converse
<!-- bilingual-en:end -->
*构造因子，并证明反向*

<!-- bilingual-en:start -->
By the [[LDL正定判别|positive-pivot theorem]], positive-definite $A$ has $A=\widetilde L D\widetilde L^T$ with unit lower triangular $\widetilde L$ and $D=\operatorname{diag}(d_1,\ldots,d_n)$, all $d_i>0$. The tilde distinguishes this unit-diagonal factor from the Cholesky factor. Here $\operatorname{diag}$ denotes a [[对角矩阵|diagonal matrix]], with zeros away from the listed diagonal. Take the ordinary positive square root of each $d_i$:
<!-- bilingual-en:end -->

根据 [[LDL正定判别]]，正定 $A$ 有分解 $A=\widetilde L D\widetilde L^T$，其中 $\widetilde L$ 单位下三角，$D=\operatorname{diag}(d_1,\ldots,d_n)$ 且全部 $d_i>0$。波浪号用于区分单位对角的因子与 Cholesky 因子。$\operatorname{diag}$ 表示[[对角矩阵]]，列出的对角元之外全为零。对每个 $d_i$ 取通常的正平方根：

$$D^{1/2}=\operatorname{diag}(\sqrt{d_1},\ldots,\sqrt{d_n}),\qquad
L=\widetilde L D^{1/2},\qquad LL^T=A.$$

<!-- bilingual-en:start -->
Multiplication by $D^{1/2}$ scales each column of $\widetilde L$, preserving its triangular shape and giving diagonal entries $\sqrt{d_i}>0$. In the opening example, $\widetilde L=\begin{pmatrix}1&0\\1/2&1\end{pmatrix}$ and $D=\operatorname{diag}(4,4)$ produce the displayed $L$.
<!-- bilingual-en:end -->

右乘 $D^{1/2}$ 逐列缩放 $\widetilde L$，保留三角形状，并使对角元为 $\sqrt{d_i}>0$。开头例子的 $\widetilde L=\begin{pmatrix}1&0\\1/2&1\end{pmatrix}$ 与 $D=\operatorname{diag}(4,4)$ 恰好产生所展示的 $L$。

<!-- bilingual-en:start -->
Conversely, if $A=LL^T$ with positive diagonal triangular $L$, successive substitution shows that $L$ has an [[逆矩阵|inverse]]. Thus $x\ne0$ implies $L^Tx\ne0$, and
<!-- bilingual-en:end -->

反过来，若 $A=LL^T$ 且 $L$ 为正对角三角矩阵，逐次代入求解便知 $L$ 有[[逆矩阵]]。因此 $x\ne0$ 蕴含 $L^Tx\ne0$，并且

$$x^TAx=\|L^Tx\|_2^2>0.$$

<!-- bilingual-en:start -->
The squared [[欧氏范数|Euclidean norm]] $\|y\|_2^2=\sum_i y_i^2$ is positive exactly when the real vector $y$ is nonzero. Also $(LL^T)^T=LL^T$, so the product is symmetric. Replacing transpose by conjugate transpose and squares by squared magnitudes proves the complex converse too.
<!-- bilingual-en:end -->

[[欧氏范数]]的平方 $\|y\|_2^2=\sum_i y_i^2$ 恰好在实向量 $y$ 非零时为正。又有 $(LL^T)^T=LL^T$，所以乘积对称。把转置换成共轭转置，把平方换成模平方，就得到复数情形的反向证明。

<!-- bilingual-en:start -->
## Why the positive diagonal gives uniqueness
<!-- bilingual-en:end -->
*正对角元为什么保证唯一*

<!-- bilingual-en:start -->
Suppose $LL^*=MM^*$, where both factors are lower triangular with positive real diagonals; this includes the real case. Set $Q=M^{-1}L$. An invertible lower triangular matrix has a lower triangular inverse, as follows by solving its columns successively; [[下三角乘积封闭性|products remain lower triangular]]. Therefore $Q$ is lower triangular with positive diagonal, and $QQ^*=I$, where $I$ is the [[单位矩阵|identity matrix]]. Consequently $Q^{-1}=Q^*$ is both lower triangular and upper triangular. It must be diagonal, and so must $Q$. Each positive diagonal entry $q_{ii}$ satisfies $|q_{ii}|^2=1$, forcing $q_{ii}=1$. Hence $Q=I$ and $L=M$.
<!-- bilingual-en:end -->

假设 $LL^*=MM^*$，两个因子均下三角且对角元为正实数，这也包含实数情形。令 $Q=M^{-1}L$。通过逐列求解可知，可逆下三角矩阵的逆仍下三角；再由[[下三角乘积封闭性]]，$Q$ 下三角且对角元正，并满足 $QQ^*=I$，其中 $I$ 为[[单位矩阵]]。所以 $Q^{-1}=Q^*$ 同时是下三角和上三角，只能是对角矩阵，$Q$ 也只能是对角矩阵。每个正对角元满足 $|q_{ii}|^2=1$，于是 $q_{ii}=1$，从而 $Q=I$、$L=M$。

<!-- bilingual-en:start -->
The normalization matters: replacing a real factor $L$ by $-L$ leaves $LL^T$ unchanged but violates the positive-diagonal convention. Uniqueness also refers to a fixed variable order. Simultaneously reordering rows and columns preserves positive definiteness but generally changes the triangular factor in the reordered coordinates.
<!-- bilingual-en:end -->

这个规范确实必要：把实因子 $L$ 换为 $-L$，乘积 $LL^T$ 不变，却不再满足正对角约定。唯一性也针对固定变量顺序；同时重排行和列保持正定性，但一般会改变新坐标次序下的三角因子。

<!-- bilingual-en:start -->
## Singular factors and numerical tests
<!-- bilingual-en:end -->
*奇异因子与数值检验*

<!-- bilingual-en:start -->
Every real symmetric [[半正定矩阵|positive-semidefinite]] matrix has some $A=BB^T$, for instance $B=A^{1/2}$ from the [[半正定主平方根|principal square root]]. If $A$ is singular, no such square factor can be invertible. For $A=\operatorname{diag}(1,0)$, taking $B=\operatorname{diag}(1,0)$ gives $A=BB^T$, but the nonzero vector $(0,1)^T$ has quadratic value zero. Thus a product of this form alone does not prove strict positivity. The principal square root is itself symmetric and satisfies $B^2=A$; a Cholesky factor need not be symmetric and generally does not satisfy $L^2=A$.
<!-- bilingual-en:end -->

每个实对称[[半正定矩阵]]都有某个 $A=BB^T$，例如取[[半正定主平方根]] $B=A^{1/2}$。若 $A$ 奇异，这样的方形因子不可能可逆。对 $A=\operatorname{diag}(1,0)$，取 $B=\operatorname{diag}(1,0)$ 就有 $A=BB^T$，但非零向量 $(0,1)^T$ 的二次型值为零。因此，仅有这种乘积形式并不证明严格正定。主平方根本身对称且满足 $B^2=A$；Cholesky 因子不必对称，一般也不满足 $L^2=A$。

<!-- bilingual-en:start -->
For a verified real symmetric input, the exact unpivoted algorithm successively computes
<!-- bilingual-en:end -->

对已确认实对称的输入，精确的不换主元算法逐步计算

$$r_j=a_{jj}-\sum_{k<j}l_{jk}^2,\qquad
l_{jj}=\sqrt{r_j},\qquad
l_{ij}=\frac{a_{ij}-\sum_{k<j}l_{ik}l_{jk}}{l_{jj}}\quad(i>j).$$

<!-- bilingual-en:start -->
Indices $i,j,k$ label rows and columns; $l_{ij}$ denotes an entry of $L$, and an empty sum at $j=1$ is zero. Every $r_j$ must be positive. A negative value has no real square root; zero violates the required positive diagonal and can make the next division fail. In exact arithmetic either event rules out positive definiteness, although zero may belong to a semidefinite case needing a different treatment.
<!-- bilingual-en:end -->

$i,j,k$ 是行列下标，$l_{ij}$ 是 $L$ 的元素；$j=1$ 时空和为零。每个 $r_j$ 都必须正：负数没有实平方根，零则违反正对角要求，并可能使接下来的除法失效。在精确算术下，两种情况都排除正定，但零可能来自需要另行处理的半正定情形。

<!-- bilingual-en:start -->
Floating-point calculation adds input rounding and arithmetic error. For example, $A_\varepsilon=\begin{pmatrix}1&1\\1&1+\varepsilon\end{pmatrix}$ has exact radicands $1,\varepsilon$ and is positive definite for every real $\varepsilon>0$. At finite precision, a sufficiently small $\varepsilon$ can disappear when $1+\varepsilon$ is stored, leaving a singular matrix instead. A computed tiny pivot, a tolerance or successful completion alone is therefore not a proof about the exact intended input; assess the stored data, scale and error. Moreover, routines such as LAPACK's DPOTRF read only one specified triangle: success does not check whether a separately supplied other triangle agrees with it.
<!-- bilingual-en:end -->

浮点计算还存在输入舍入和运算误差。例如 $A_\varepsilon=\begin{pmatrix}1&1\\1&1+\varepsilon\end{pmatrix}$ 的精确被开方数为 $1,\varepsilon$，对每个实数 $\varepsilon>0$ 都正定。但在有限精度下，足够小的 $\varepsilon$ 可能在保存 $1+\varepsilon$ 时消失，实际存储的矩阵反而奇异。因此，计算出的微小主元、一个容差值或程序成功结束，本身都不是关于精确预期输入的证明；还需考虑存储数据、尺度与误差。并且 LAPACK 的 DPOTRF 等程序只读取指定的一侧三角区域：成功结束不代表它检查了另一侧另行提供的数据是否对称一致。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Stanford EE263, *Cholesky Factorization*, PDF pp.2–5 and 8–9](https://ee263.stanford.edu/lectures/cholesky.pdf#page=2): real positive-diagonal factorization, converse, uniqueness and scalar algorithm. [UCLA, *Symmetric factorizations*, PDF p.3](https://math.ucla.edu/~njhu/notes/nla/lin-direct/symfact.pdf#page=3) supports the Hermitian form and construction from positive $LDL^*$ pivots.
- [MIT 18.06 Spring 2010 Problem Set 9 solutions, PDF p.1, problems 6.5.25–26](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/58e62cc93e3e8b6cb52e72fa5830fc4e_MIT18_06S10_pset9_s10_soln.pdf#page=1) and [[03_Positive Definite Matrices and Applications#3.3.2 配方就是 $LDL^T$|course Section 3.3.2]]: taking square roots of positive pivots and the local factor convention. The displayed example is multiplied out directly.
- [Demmel, *On Floating Point Errors in Cholesky*, LAPACK Working Note 14, PDF pp.1–2 and 5–6](https://netlib.org/lapack/lawnspdf/lawn14.pdf): input perturbation, roundoff and successful-completion boundaries near singularity. [LAPACK DPOTRF documentation](https://netlib.org/lapack/explore-html/d2/d09/group__potrf_ga84e90859b02139934b166e579dd211d4.html): selected-triangle storage and failure reporting. The $A_\varepsilon$ example is an explicit algebraic and rounding illustration, not a universal software tolerance rule.
<!-- bilingual-en:end -->

- Stanford EE263 讲义 PDF 第2–5、8–9页：支持实数正对角分解、反向、唯一性及标量算法；UCLA 第3页支持 Hermitian 情形与正 $LDL^*$ 主元构造。
- MIT 习题9题解 PDF 第1页的6.5.25–26题与课程 §3.3.2：核对正主元开方及本地因子约定；正文例子直接相乘验算。
- Demmel 的 LAPACK Working Note 14，第1–2、5–6页：支持近奇异时输入扰动、舍入与算法完成的边界；DPOTRF 官方文档支持单侧三角存储及失败报告。$A_\varepsilon$ 是明确的代数与舍入示例，不是通用软件容差规则。
