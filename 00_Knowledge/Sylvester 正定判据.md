---
aliases:
  - Sylvester's criterion
  - Leading principal minor test for positive definiteness
student_os: knowledge-atom
atom_id: LA-SPD-014
atom_set: symmetric-positive-definite
atom_type: equivalence
status: source-checked
requires:
  - "[[LDL正定判别]]"
  - "[[正定矩阵]]"
  - "[[Hermitian 矩阵]]"
related:
  - "[[半正定主子式判别]]"
  - "[[行列式.canvas]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

<!-- bilingual-en:start -->
# Sylvester's criterion
<!-- bilingual-en:end -->
*Sylvester 正定判据*

<!-- bilingual-en:start -->
Sylvester's criterion tests positive definiteness by computing the determinants of a nested sequence of upper-left blocks. It can save us from solving for every eigenvalue. The important restriction is that the matrix must be real symmetric or complex Hermitian, and **every** determinant in that sequence must be **strictly positive**.
<!-- bilingual-en:end -->

Sylvester 判据通过一串左上角嵌套方块的行列式来判断正定性，因此可以不先求出全部特征值。关键限制是：矩阵必须实对称或复 Hermitian，并且这串行列式中的**每一个**都要**严格为正**。

<!-- bilingual-en:start -->
For $A=\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$, the first determinant is $2$, and the full determinant is $2\cdot2-(-1)^2=3$. Both are positive, so $A$ is [[正定矩阵|positive definite]]: $x^TAx>0$ for every nonzero real column $x$. Here $T$ denotes [[矩阵转置|transpose]]. The conclusion can also be seen from $x^TAx=x_1^2+x_2^2+(x_1-x_2)^2$.
<!-- bilingual-en:end -->

对 $A=\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$，第一个行列式为 $2$，整个行列式为 $2\cdot2-(-1)^2=3$。两者都正，因此 $A$ [[正定矩阵|正定]]，即每个非零实列向量 $x$ 都满足 $x^TAx>0$；$T$ 表示[[矩阵转置|转置]]。也可以直接看到 $x^TAx=x_1^2+x_2^2+(x_1-x_2)^2$。

<!-- bilingual-en:start -->
## Which determinants are tested?
<!-- bilingual-en:end -->
*到底检查哪些行列式？*

<!-- bilingual-en:start -->
Let $A$ have size $n\times n$, with integer $n\ge1$. Assume $A=A^T$ with real entries, the [[实对称矩阵|real symmetric]] case, or $A=A^*$ with complex entries, the [[Hermitian 矩阵|Hermitian]] case; the star means conjugate transpose. Its [[顺序主子式|leading principal minors]] are the numbers
<!-- bilingual-en:end -->

设 $A$ 为 $n\times n$ 矩阵，整数 $n\ge1$。假设它具有实数元素且 $A=A^T$，即[[实对称矩阵|实对称]]；或具有复数元素且 $A=A^*$，即 [[Hermitian 矩阵|Hermitian]]，星号表示共轭转置。它的[[顺序主子式]]是下面这些数：

$$\Delta_k=\det A_{1:k,1:k},\qquad k=1,\ldots,n.$$

<!-- bilingual-en:start -->
The notation $1:k$ means retain the first $k$ rows and columns in the current order, then take their [[行列式|determinant]]. These determinants are real under the stated assumptions. Sylvester's criterion is
<!-- bilingual-en:end -->

$1:k$ 表示按当前排列保留前 $k$ 行、前 $k$ 列，再取[[行列式]]。在上述前提下，这些行列式都是实数。Sylvester 判据为

> [!important] All orders, strict signs · 全部阶数、严格正号
> $$A\succ0\quad\Longleftrightarrow\quad
> \Delta_1>0,\ \Delta_2>0,\ \ldots,\ \Delta_n>0.$$

<!-- bilingual-en:start -->
For a real two-by-two matrix $\begin{pmatrix}a&b\\b&c\end{pmatrix}$, the complete test is $a>0$ and $ac-b^2>0$. For a complex Hermitian matrix $\begin{pmatrix}a&b\\\overline b&c\end{pmatrix}$, where $a,c$ are real and $\overline b$ is the complex conjugate of $b$, replace $b^2$ by $|b|^2=b\overline b$: the test is $a>0$ and $ac-|b|^2>0$. With $i^2=-1$, for instance, $\begin{pmatrix}2&i\\-i&2\end{pmatrix}$ has leading minors $2,3$ and is positive definite. For $n=1$, the sole condition is $a_{11}>0$.
<!-- bilingual-en:end -->

实二阶矩阵 $\begin{pmatrix}a&b\\b&c\end{pmatrix}$ 的完整条件是 $a>0$ 且 $ac-b^2>0$。复 Hermitian 矩阵 $\begin{pmatrix}a&b\\\overline b&c\end{pmatrix}$ 中 $a,c$ 为实数，$\overline b$ 为 $b$ 的复共轭；此时把 $b^2$ 换成模平方 $|b|^2=b\overline b$，得到 $a>0$ 且 $ac-|b|^2>0$。例如，令 $i^2=-1$，则 $\begin{pmatrix}2&i\\-i&2\end{pmatrix}$ 的顺序主子式为 $2,3$，因此正定。$n=1$ 时只有条件 $a_{11}>0$。

<!-- bilingual-en:start -->
## Why positive definiteness forces every sign
<!-- bilingual-en:end -->
*正定为什么迫使每个符号都正*

<!-- bilingual-en:start -->
Take a nonzero vector with $k$ entries and append $n-k$ zeros. Its quadratic value under $A$ equals the value of its original coordinates under $A_{1:k,1:k}$. If $A$ is positive definite, each leading block is therefore positive definite too. The [[正定与半正定的谱判据|spectral criterion]] makes all eigenvalues of each block positive, and its [[行列式与特征值乘积|determinant is their product]]. Hence every $\Delta_k>0$.
<!-- bilingual-en:end -->

取一个有 $k$ 个分量的非零向量，在后面补上 $n-k$ 个零。它在 $A$ 下的二次型值，等于原来 $k$ 个坐标在 $A_{1:k,1:k}$ 下的值。因此，若 $A$ 正定，每个顺序主子矩阵也正定。由[[正定与半正定的谱判据]]，每个块的特征值都正；再由[[行列式与特征值乘积|行列式是特征值之积]]，得到每个 $\Delta_k>0$。

<!-- bilingual-en:start -->
## Why the signs are enough
<!-- bilingual-en:end -->
*这些符号为什么又足够*

<!-- bilingual-en:start -->
Prove the reverse implication by increasing the dimension. The one-entry case is immediate. Suppose the result is known for dimension $n-1$, and write the $n$-dimensional Hermitian matrix as
<!-- bilingual-en:end -->

用维数递增的方式证明反向。一阶情形显然成立。假设 $n-1$ 维已经证明，把 $n$ 维 Hermitian 矩阵写成

$$A=\begin{pmatrix}B&b\\b^*&c\end{pmatrix}.$$

<!-- bilingual-en:start -->
Here $B$ is the first $(n-1)\times(n-1)$ block, $b$ is a column with $n-1$ entries, and $c$ is real. All leading minors of $B$ are positive, so the induction hypothesis makes $B$ positive definite. It has an [[逆矩阵|inverse]]: $By=0$ for nonzero $y$ would imply $y^*By=0$, contradicting positivity. Its inverse is Hermitian as well, since taking conjugate transposes in the two inverse equations gives $(B^{-1})^*=B^{-1}$.
<!-- bilingual-en:end -->

$B$ 是左上角 $(n-1)\times(n-1)$ 块，$b$ 是含 $n-1$ 个元素的列，$c$ 为实数。$B$ 的全部顺序主子式都正，因此归纳假设说明 $B$ 正定。它有[[逆矩阵]]：若某个非零 $y$ 满足 $By=0$，便有 $y^*By=0$，与正定矛盾。其逆也为 Hermitian，因为对两侧逆矩阵等式取共轭转置，得到 $(B^{-1})^*=B^{-1}$。

<!-- bilingual-en:start -->
Set $s=c-b^*B^{-1}b$, a real scalar. [[分块矩阵乘法|Block multiplication]] gives
<!-- bilingual-en:end -->

令实数 $s=c-b^*B^{-1}b$。由[[分块矩阵乘法]]，

$$\begin{pmatrix}I&0\\-b^*B^{-1}&1\end{pmatrix}
\begin{pmatrix}B&b\\b^*&c\end{pmatrix}
=\begin{pmatrix}B&b\\0&s\end{pmatrix}.$$

<!-- bilingual-en:start -->
Here $I$ is the $(n-1)$-dimensional [[单位矩阵|identity matrix]], and the zero blocks have the matching sizes. The left multiplier is triangular with diagonal entries all $1$, so its determinant is $1$. Expanding the right-hand determinant along its last row gives $(\det B)s$. Consequently $\Delta_n=\Delta_{n-1}s$, and the assumed positive determinants give $s>0$.
<!-- bilingual-en:end -->

$I$ 是 $n-1$ 维[[单位矩阵]]，零块取相应尺寸。左乘矩阵为三角矩阵，对角元全为 $1$，所以行列式为 $1$；沿右侧矩阵的最后一行展开，行列式为 $(\det B)s$。因此 $\Delta_n=\Delta_{n-1}s$，假设中的两个正行列式说明 $s>0$。

<!-- bilingual-en:start -->
For a vector written as $\binom yt$, where $y$ has $n-1$ complex entries and $t$ is one complex scalar, completing the square yields
<!-- bilingual-en:end -->

把向量写成 $\binom yt$，其中 $y$ 有 $n-1$ 个复分量，$t$ 为一个复数。配方得到

$$\begin{pmatrix}y^*&\overline t\end{pmatrix}
A\binom yt
=(y+B^{-1}bt)^*B(y+B^{-1}bt)+s|t|^2.$$

<!-- bilingual-en:start -->
If $t\ne0$, the second term is positive and the first is nonnegative. If $t=0$ but $y\ne0$, the first term is positive. Every nonzero vector therefore has a positive value, finishing the induction. For real symmetric matrices, all bars disappear and $*$ becomes $T$; the same argument applies.
<!-- bilingual-en:end -->

若 $t\ne0$，第二项正，第一项非负；若 $t=0$ 而 $y\ne0$，第一项正。因此每个非零向量都给出正值，归纳证明完成。实对称情形只需去掉共轭，并把 $*$ 换成 $T$，论证完全相同。

<!-- bilingual-en:start -->
## How determinants connect to pivots
<!-- bilingual-en:end -->
*与主元乘积公式的联系*

<!-- bilingual-en:start -->
Repeated completion of squares is the [[LDL正定判别|symmetry-preserving factorization]] $A=LDL^*$ (real case $LDL^T$), without row exchanges. Here $L$ is lower triangular with diagonal entries $1$, and $D$ is diagonal with real pivots $d_k$. Nonzero leading minors permit this recurrence without division by zero. Each leading block of $L$ has determinant $1$, giving the identities
<!-- bilingual-en:end -->

重复配方就是不做行交换的[[LDL正定判别|对称消元分解]] $A=LDL^*$，实数情形为 $LDL^T$。$L$ 是对角元全为 $1$ 的下三角矩阵，$D$ 是以实主元 $d_k$ 为对角元的对角矩阵。非零顺序主子式保证递推不会除以零。$L$ 的每个顺序主块行列式都为 $1$，从而得到等式

$$\Delta_k=d_1\cdots d_k,\qquad
 d_k=\frac{\Delta_k}{\Delta_{k-1}},\qquad \Delta_0=1.$$

<!-- bilingual-en:start -->
Thus strictly positive leading minors correspond to strictly positive pivots. These statements refer to the symmetry-preserving factorization in the fixed ordering, not to arbitrary elimination after unrelated row exchanges.
<!-- bilingual-en:end -->

因此，全部顺序主子式严格为正，对应全部主元严格为正。这些说法针对固定排列下保持对称结构的分解，不能照搬到任意换行后的消元结果。

<!-- bilingual-en:start -->
## What a shortened check can miss
<!-- bilingual-en:end -->
*少检查条件会漏掉什么*

<!-- bilingual-en:start -->
A positive final determinant alone is insufficient: the [[对角矩阵|diagonal matrix]] $\operatorname{diag}(-1,-1)$, with these two diagonal entries and zeros elsewhere, has determinant $1$ but fails the first test. Positive diagonal entries alone are also insufficient: $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ has determinant $-3$. Without symmetry, $\begin{pmatrix}1&4\\0&1\end{pmatrix}$ even has both leading minors positive, yet its quadratic value at $(1,-1)^T$ is $-2$.
<!-- bilingual-en:end -->

只检查最后一个行列式为正不够：[[对角矩阵]] $\operatorname{diag}(-1,-1)$ 的对角元为这两个数、其余为零，行列式为 $1$，却通不过第一个条件。只检查对角元为正也不够：$\begin{pmatrix}1&2\\2&1\end{pmatrix}$ 的行列式为 $-3$。若没有对称性，$\begin{pmatrix}1&4\\0&1\end{pmatrix}$ 甚至两个顺序主子式都正，但在 $(1,-1)^T$ 上的二次型值为 $-2$。

<!-- bilingual-en:start -->
Do not replace every $>0$ by $\ge0$ to test semidefiniteness. For $\operatorname{diag}(0,-1)$ both leading minors are zero, although the second coordinate gives a negative quadratic value. The [[顺序主子式不推半正定|leading-minor counterexample]] and the [[半正定主子式判别|all-principal-minors criterion]] explain the different semidefinite test. These are exact algebraic statements; determinants rounded near zero require a separate numerical error assessment.
<!-- bilingual-en:end -->

判断半正定时，不能把所有 $>0$ 直接替换为 $\ge0$。对 $\operatorname{diag}(0,-1)$，两个顺序主子式都为零，但第二坐标方向给出负二次型值。[[顺序主子式不推半正定]]与[[半正定主子式判别]]分别解释反例和正确的全部主子式条件。以上是精确代数结论；行列式舍入到零附近时，还需另行评估数值误差。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [Givental, *Linear Algebra*, Chapter 3, “Sylvester's Rule,” PDF pp.31–33 / printed pp.131–133](https://math.berkeley.edu/~giventh/la/la_21_ch3.pdf#page=31): Hermitian leading minors, strict positive-definiteness equivalence in Corollary 2, and the real specialization. The block induction above supplies each algebraic step explicitly.
- [[01_Math/02_linear algebra/MIT_OCW_18.06SC_PDF/05_Session_Summaries/MIT18_06SCF11_Ses3.3sum.pdf#page=1|MIT 18.06SC Session 3.3, p.1]] and [[03_Positive Definite Matrices and Applications#主元判据为何成立|permanent course pivot proof]]: the two-dimensional test and the pivot-product identities with the no-row-exchange convention.
- [Princeton ORF523 Lecture 2, §2.3, PDF pp.9–10](https://www.princeton.edu/~aaa/Public/Teaching/ORF523/S16/ORF523_S16_Lec2_gh.pdf#page=9): strict leading-minor criterion and the contrasting all-principal-minors semidefinite criterion. The examples and block determinant/completion identities are recomputed directly.
<!-- bilingual-en:end -->

- Givental 第3章 Sylvester's Rule（PDF第31–33页／印刷第131–133页）：支持 Hermitian 顺序主子式、推论2的严格正定等价及实数特例；正文用分块归纳逐步展开代数证明。
- MIT Session 3.3 第1页与永久课程主元证明：核对二维判据、不换行约定及主元乘积恒等式。
- Princeton Lecture 2 §2.3（PDF第9–10页）：核对严格顺序主子式判据与全部主子式的半正定判据之别；例子和分块等式均独立计算。
