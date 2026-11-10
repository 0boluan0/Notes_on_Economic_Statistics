---
aliases:
- Positive pivots criterion
- Completing the square via LDLT
- LDL positive-definiteness criterion
student_os: knowledge-atom
atom_id: LA-SPD-013
atom_set: symmetric-positive-definite
atom_type: equivalence
status: source-checked
requires:
- '[[正定矩阵]]'
related:
- '[[Sylvester 正定判据]]'
leads_to:
- '[[Cholesky 正定判据]]'
contrasts_with:
- '[[LDL零主元边界]]'
part_of:
- '[[对称矩阵与正定二次型.canvas]]'
---

<!-- bilingual-en:start -->
# The positive-pivot criterion
<!-- bilingual-en:end -->
*LDL 正定判别*

<!-- bilingual-en:start -->
The positive-pivot criterion tests a symmetric quadratic expression by completing its squares. Elimination records the new linear combinations in a triangular matrix and the coefficients of their squares in a diagonal matrix. If every coefficient is positive and the change of coordinates is invertible, every nonzero input has a positive value.
<!-- bilingual-en:end -->

正主元判据通过配方来检验对称二次表达式。消元把新的线性组合记录在三角矩阵中，把各平方项的系数记录在对角矩阵中。如果系数全部为正，而且坐标变换可逆，每个非零输入就都得到正值。

<!-- bilingual-en:start -->
For example, let $x=(s,t)^T$, where $T$ denotes [[矩阵转置|transpose]], and consider the [[二次型|quadratic form]]
<!-- bilingual-en:end -->

例如令 $x=(s,t)^T$，$T$ 表示[[矩阵转置|转置]]，考虑[[二次型]]

$$A=\begin{pmatrix}4&2\\2&5\end{pmatrix},\qquad
x^TAx=4s^2+4st+5t^2
=4\left(s+\frac12t\right)^2+4t^2.$$

<!-- bilingual-en:start -->
The two squares can vanish together only when $t=0$ and then $s=0$. Thus this expression is strictly positive for every nonzero $x$, which is [[正定矩阵|positive definiteness]]. The numbers $4,4$ are the elimination pivots, not the original diagonal entries $4,5$. Subtracting half the first row from the second leaves the second pivot $5-\tfrac12\cdot2=4$.
<!-- bilingual-en:end -->

两个平方项同时为零，必须先有 $t=0$，继而 $s=0$。因此每个非零 $x$ 都得到严格正值，这就是[[正定矩阵|正定性]]。这里的 $4,4$ 是消元主元，不是原来的对角元 $4,5$：第二行减去第一行的一半后，第二个主元为 $5-\tfrac12\cdot2=4$。

<!-- bilingual-en:start -->
## What the factors mean
<!-- bilingual-en:end -->
*三个因子分别记录什么*

<!-- bilingual-en:start -->
Fix a finite positive integer $n$ and a [[实对称矩阵|real symmetric]] $n\times n$ matrix $A$. In an **unpivoted $LDL^T$ factorization**,
<!-- bilingual-en:end -->

设 $n$ 为有限正整数，$A$ 是 $n\times n$ [[实对称矩阵]]。**不换主元次序的 $LDL^T$ 分解**写为

$$A=LDL^T,\qquad D=\operatorname{diag}(d_1,\ldots,d_n).$$

<!-- bilingual-en:start -->
$L$ is [[三角矩阵|lower triangular]]: all entries above its main diagonal are zero. Its diagonal entries are fixed at $1$, hence “unit lower triangular.” $D$ is [[对角矩阵|diagonal]], with listed entries $d_i$ and zeros elsewhere. The $d_i$ are the scalar [[主元|pivots]] before any normalization of elimination rows. “Unpivoted” keeps the original variable order and performs no row exchanges; the right factor $L^T$ keeps the factorization symmetric.
<!-- bilingual-en:end -->

$L$ 是[[三角矩阵|下三角矩阵]]，主对角线上方全为零；对角元固定为 $1$，所以叫“单位下三角”。$D$ 是[[对角矩阵]]，列出的 $d_i$ 位于对角线上，其余元素为零。$d_i$ 是消元行尚未归一化时的标量[[主元]]。“不换主元次序”指保留原来的变量顺序、不交换行；右侧配上 $L^T$，使分解保持对称结构。

<!-- bilingual-en:start -->
The opening example has
<!-- bilingual-en:end -->

开头的例子给出

$$L=\begin{pmatrix}1&0\\1/2&1\end{pmatrix},\quad
D=\begin{pmatrix}4&0\\0&4\end{pmatrix},\quad
L^Tx=\binom{s+t/2}{t}.$$

> [!important] Exact positive-pivot criterion · 精确的正主元判据
> <!-- bilingual-en:start -->
> A real symmetric matrix is positive definite if and only if it admits this factorization with **every $d_i>0$**. A positive-definite matrix always admits it, and the unit-diagonal normalization makes its factors unique.
> <!-- bilingual-en:end -->
>
> 实对称矩阵正定，当且仅当存在上述分解且**每个 $d_i>0$**。正定矩阵一定存在这样的分解；把 $L$ 的对角元固定为 $1$ 后，两个因子唯一。

<!-- bilingual-en:start -->
## Why the signs decide the quadratic form
<!-- bilingual-en:end -->
*主元符号为什么能决定二次型*

<!-- bilingual-en:start -->
A unit lower triangular matrix has an [[逆矩阵|inverse]]: its equations can be solved successively without dividing by zero. Consequently $y=L^Tx$ runs through all vectors, and $x\ne0$ exactly when $y\ne0$. The factorization gives
<!-- bilingual-en:end -->

单位下三角矩阵有[[逆矩阵]]，因为逐个求解它的方程时不会除以零。因此 $y=L^Tx$ 能遍历所有向量，而且 $x\ne0$ 当且仅当 $y\ne0$。分解给出

$$x^TAx=(L^Tx)^TD(L^Tx)=\sum_{i=1}^n d_i y_i^2.$$

<!-- bilingual-en:start -->
If all $d_i>0$, a nonzero $y$ has at least one positive contribution, proving positivity. Conversely, take $y=e_i$, the column with $1$ in position $i$ and zeros elsewhere. The nonzero input $x=L^{-T}e_i$, where $L^{-T}=(L^T)^{-1}$, has quadratic value $d_i$. Thus positivity for every input forces every pivot to be positive. This argument applies to any factorization of the stated form; it does not infer the signs from a few trial vectors.
<!-- bilingual-en:end -->

若全部 $d_i>0$，非零 $y$ 至少贡献一个正项，因而总和正。反过来，取 $y=e_i$，即第 $i$ 个位置为 $1$、其余为零的列向量；令 $x=L^{-T}e_i$，其中 $L^{-T}=(L^T)^{-1}$。这个非零输入的二次型值就是 $d_i$，所以对所有输入正定迫使每个主元都正。论证适用于上述形式的任意分解，并非只试算几个向量。

> [!example] Recover a negative direction · 找回负方向
> <!-- bilingual-en:start -->
> If $D=\operatorname{diag}(2,-3)$, choose $y=e_2$ and $x=L^{-T}e_2$. Then $x^TAx=-3$. For $L=\begin{pmatrix}1&0\\2&1\end{pmatrix}$, this gives $x=(-2,1)^T$ and $A=\begin{pmatrix}2&4\\4&5\end{pmatrix}$. Direct substitution gives $8-16+5=-3$.
> <!-- bilingual-en:end -->
>
> 若 $D=\operatorname{diag}(2,-3)$，取 $y=e_2$ 和 $x=L^{-T}e_2$，便有 $x^TAx=-3$。例如 $L=\begin{pmatrix}1&0\\2&1\end{pmatrix}$ 时，得到 $x=(-2,1)^T$ 与 $A=\begin{pmatrix}2&4\\4&5\end{pmatrix}$；直接代入为 $8-16+5=-3$。

<!-- bilingual-en:start -->
## Why positive definiteness permits every elimination step
<!-- bilingual-en:end -->
*正定为什么保证消元可以逐步进行*

<!-- bilingual-en:start -->
For $n>1$, separate the first coordinate:
<!-- bilingual-en:end -->

当 $n>1$，把第一个坐标分开：

$$A=\begin{pmatrix}\alpha&b^T\\b&B\end{pmatrix},\qquad
S=B-\frac{bb^T}{\alpha}.$$

<!-- bilingual-en:start -->
Here $\alpha=a_{11}$ is a scalar, $b$ is an $(n-1)$-entry column and $B$ is the remaining square block. Positive definiteness gives $\alpha>0$. The remaining matrix $S$ is what is left after completing the first square: for a scalar $t$ and a column $u$,
<!-- bilingual-en:end -->

$\alpha=a_{11}$ 是一个数，$b$ 是含 $n-1$ 个元素的列，$B$ 是余下方块。正定使 $\alpha>0$。$S$ 是完成第一个平方项后留下的矩阵：对标量 $t$ 和列向量 $u$，

$$\begin{pmatrix}t&u^T\end{pmatrix}A\binom tu
=\alpha\left(t+\frac{b^Tu}{\alpha}\right)^2+u^TSu.$$

<!-- bilingual-en:start -->
For each nonzero $u$, choose $t=-b^Tu/\alpha$. The whole input is still nonzero, so $u^TSu>0$. Thus $S$ is positive definite as well. Repeating this argument reduces the dimension until the one-entry case. If $S=L_1D_1L_1^T$, the next assembly is
<!-- bilingual-en:end -->

对每个非零 $u$，选择 $t=-b^Tu/\alpha$；整个输入仍非零，因此 $u^TSu>0$。所以 $S$ 也正定。不断重复，维数逐渐减到一阶。若 $S=L_1D_1L_1^T$，则拼回

$$A=
\begin{pmatrix}1&0\\b/\alpha&L_1\end{pmatrix}
\begin{pmatrix}\alpha&0\\0&D_1\end{pmatrix}
\begin{pmatrix}1&b^T/\alpha\\0&L_1^T\end{pmatrix}.$$

<!-- bilingual-en:start -->
Every pivot produced this way is positive. The first pivot and first column below the diagonal are forced to be $\alpha$ and $b/\alpha$; the remaining factors are then forced recursively. This also proves uniqueness for positive-definite $A$.
<!-- bilingual-en:end -->

这样得到的每个主元都正。第一个主元与第一列的下方元素分别被强制为 $\alpha$ 和 $b/\alpha$，余下因子再由递推确定。这也证明了正定情形的唯一性。

<!-- bilingual-en:start -->
## Ordering, zero pivots and complex entries
<!-- bilingual-en:end -->
*排列、零主元与复数元素*

<!-- bilingual-en:start -->
More generally, if all [[顺序主子式|leading principal minors]] $\Delta_k=\det A_{1:k,1:k}$ are nonzero, unpivoted scalar elimination exists uniquely with nonzero pivots. Here $1:k$ selects the first $k$ rows and columns. The leading block of $L$ is unit triangular, so
<!-- bilingual-en:end -->

更一般地，若全部[[顺序主子式]] $\Delta_k=\det A_{1:k,1:k}$ 非零，不换序的标量消元就唯一存在，主元均非零。$1:k$ 表示前 $k$ 行与前 $k$ 列；$L$ 的对应顺序主块是单位三角矩阵，因此

$$\Delta_k=d_1\cdots d_k,\qquad
 d_k=\frac{\Delta_k}{\Delta_{k-1}},\qquad \Delta_0=1.$$

<!-- bilingual-en:start -->
The determinant identity uses the [[三角矩阵行列式|product of triangular diagonal entries]]. It connects positive pivots to [[Sylvester 正定判据|Sylvester's strict leading-minor criterion]]. A zero pivot instead needs the separate [[LDL零主元边界|zero-pivot analysis]]; it does not by itself prove that the whole matrix is singular. For example, $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ is invertible but cannot begin this scalar recurrence.
<!-- bilingual-en:end -->

行列式恒等式使用[[三角矩阵行列式|三角矩阵对角元的乘积]]，把正主元与 [[Sylvester 正定判据|Sylvester 的严格顺序主子式判据]]联系起来。零主元则需要另看 [[LDL零主元边界]]；它本身不说明整个矩阵奇异。例如 $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ 可逆，却不能开始这个标量递推。

<!-- bilingual-en:start -->
For a singular positive-semidefinite matrix, a factorization may exist while the usual division-based recurrence or uniqueness fails. Symmetric permutations exchange corresponding rows and columns together. They preserve positive definiteness, although the numerical pivots may change. General indefinite algorithms can also use $2\times2$ pivot blocks: then positivity must be checked for those whole blocks, not just their diagonal entries. Arbitrary row swaps alone do not preserve the symmetric factorization used here.
<!-- bilingual-en:end -->

奇异半正定矩阵可能存在分解，但通常依赖除法的递推或唯一性失效。对称置换同时交换对应的行和列，保持正定性，但各主元的数值可能改变。一般不定矩阵算法还可能使用 $2\times2$ 主元块，此时必须检查整个块的正定性，不能只看块的对角元。只任意交换行，不会保持这里使用的对称分解结构。

<!-- bilingual-en:start -->
For complex [[Hermitian 矩阵|Hermitian]] $A$, replace $T$ by conjugate transpose $*$: $A=LDL^*$, with $D$ real diagonal, and $z^*Az=\sum_i d_i|(L^*z)_i|^2$. Here $|w|^2=\overline w w$ is the nonnegative squared magnitude of a complex number. The same positive-pivot criterion and existence proof apply, using conjugation throughout. These are exact algebraic statements; a computed pivot near zero needs a numerical-error assessment, as discussed with the [[Cholesky 正定判据|Cholesky criterion]].
<!-- bilingual-en:end -->

对复 [[Hermitian 矩阵]]，把 $T$ 换为共轭转置 $*$：$A=LDL^*$，$D$ 为实对角矩阵，并有 $z^*Az=\sum_i d_i|(L^*z)_i|^2$。这里 $|w|^2=\overline w w$ 是复数模的非负平方。全程使用共轭后，同一个正主元判据与存在性证明仍成立。以上是精确代数结论；计算主元接近零时，需要评估数值误差，见 [[Cholesky 正定判据]]。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/06_Lecture_Transcripts/U3_S03_Lecture_Lecture_27_Positive_Definite_Matrices_and_Minima.pdf#page=13|MIT 18.06 Lecture 27 transcript, PDF pp.13–17]] and [[03_Positive Definite Matrices and Applications#3.3.2 配方就是 $LDL^T$|course Section 3.3.2]]: completing squares, symmetric elimination pivots and the leading-minor product. The displayed two-dimensional examples are recomputed directly.
- [Nicholas Hu, UCLA, *Symmetric factorizations*, PDF pp.1–3](https://math.ucla.edu/~njhu/notes/nla/lin-direct/symfact.pdf): scalar $LDL^*$, fixed-order nonzero-pivot existence/uniqueness, recursive blocks, symmetric permutations and $2\times2$ pivot blocks. The positive-definite recursion is expanded above to show why every division is legitimate.
<!-- bilingual-en:end -->

- MIT Lecture 27 讲稿 PDF 第13–17页与课程 §3.3.2：核对配方、对称消元主元及顺序主子式乘积；二维例子均直接复算。
- UCLA《Symmetric factorizations》PDF 第1–3页：核对标量 $LDL^*$、固定次序非零主元下的存在唯一性、递推块、对称置换及二阶主元块；正文展开正定递推，说明每次除法为什么合法。
