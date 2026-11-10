---
aliases:
- Zero-pivot boundary of LDLT factorization
- LDL零主元
student_os: knowledge-atom
atom_id: LA-SPD-027
atom_set: symmetric-positive-definite
atom_type: boundary
status: source-checked
requires:
  - "[[LDL正定判别]]"
  - "[[半正定矩阵]]"
related:
  - "[[Cholesky 正定判据]]"
  - "[[半正定零方向]]"
part_of:
  - "[[对称矩阵与正定二次型.canvas]]"
---

<!-- bilingual-en:start -->
# Zero pivots in LDL factorization
<!-- bilingual-en:end -->
*LDL 分解的零主元边界*

<!-- bilingual-en:start -->
An [[LDL正定判别|LDL factorization]] separates a symmetric matrix into a triangular factor and diagonal coefficients called **pivots**. Computing the next column usually means dividing by the current pivot. If it is zero, that division is unavailable. We must then examine the remaining entries: the factorization may still exist, may cease to be unique, or may require a different elimination scheme. A zero encountered by the algorithm alone does not prove that the original matrix is singular.
<!-- bilingual-en:end -->

[[LDL正定判别|LDL 分解]]把对称矩阵拆成三角因子与称为**主元**的对角系数。计算下一列时，通常要除以当前主元；若它为零，这个除法就不能使用。接下来必须检查剩余元素：分解可能仍存在，可能不再唯一，也可能需要改用另一种消元方式。算法遇到零，本身不证明原矩阵奇异。

<!-- bilingual-en:start -->
## See the obstruction in two dimensions
<!-- bilingual-en:end -->
*先看二维乘法中的障碍*

<!-- bilingual-en:start -->
Work first with [[实对称矩阵|real symmetric matrices]], meaning $A^T=A$, where $T$ denotes [[矩阵转置|transpose]]. A unit lower [[三角矩阵|triangular matrix]] has zeros above its diagonal and ones on it. In two dimensions, write its unknown lower entry as $\ell$, and write the [[对角矩阵|diagonal factor]] as $D=\operatorname{diag}(d_1,d_2)$. Direct multiplication gives
<!-- bilingual-en:end -->

先讨论[[实对称矩阵]]，即 $A^T=A$，其中 $T$ 表示[[矩阵转置|转置]]。单位下[[三角矩阵]]的对角线上方全为零，对角元全为一。二维时，把它下方的未知元素记作 $\ell$，把[[对角矩阵|对角因子]]写作 $D=\operatorname{diag}(d_1,d_2)$。直接相乘得到

$$
\begin{pmatrix}a&b\\b&c\end{pmatrix}
=\begin{pmatrix}1&0\\\ell&1\end{pmatrix}
\begin{pmatrix}d_1&0\\0&d_2\end{pmatrix}
\begin{pmatrix}1&\ell\\0&1\end{pmatrix}
=\begin{pmatrix}d_1&d_1\ell\\d_1\ell&d_1\ell^2+d_2\end{pmatrix}.
$$

<!-- bilingual-en:start -->
Thus $d_1=a$. If $a\ne0$, the next two entries are forced: $\ell=b/a$ and $d_2=c-b^2/a$. If $a=0$, return to the equation $a\ell=b$ instead of dividing. When $b\ne0$, it is impossible; when $b=0$, it imposes no restriction on $\ell$, and $d_2=c$. The failed division can therefore mean either an inconsistent equation or a free choice.
<!-- bilingual-en:end -->

所以 $d_1=a$。若 $a\ne0$，另外两个元素被确定为 $\ell=b/a$、$d_2=c-b^2/a$。若 $a=0$，应该回到方程 $a\ell=b$，而不是继续除法。$b\ne0$ 时方程无解；$b=0$ 时方程不限制 $\ell$，并有 $d_2=c$。同样是无法相除，背后可能是方程矛盾，也可能是留下自由选择。

> [!example] An existing but nonunique factorization · 存在却不唯一的分解
> <!-- bilingual-en:start -->
> Take the singular [[半正定矩阵|positive-semidefinite matrix]] $A=\operatorname{diag}(0,1)$. For $x=(x_1,x_2)^T$, its quadratic value is $x^TAx=x_2^2\ge0$, and it sends the nonzero vector $(1,0)^T$ to zero. For every real number $t$,
> <!-- bilingual-en:end -->
>
> 取奇异[[半正定矩阵]] $A=\operatorname{diag}(0,1)$。对 $x=(x_1,x_2)^T$，它的二次型值为 $x^TAx=x_2^2\ge0$，并把非零向量 $(1,0)^T$ 映成零。对每个实数 $t$，
>
> $$
> D=A,\qquad L_t=\begin{pmatrix}1&0\\t&1\end{pmatrix},\qquad L_tDL_t^T=A.
> $$
>
> <!-- bilingual-en:start -->
> Multiplication by $D$ erases the first column's contribution, so the free entry $t$ cannot change the product. Choosing $t=0$ selects one valid factorization; it does not prove uniqueness among all unit lower-triangular factors.
> <!-- bilingual-en:end -->
>
> 乘以 $D$ 时，第一列的贡献被抹去，所以自由元素 $t$ 不改变乘积。规定 $t=0$ 可以选出一个合法分解，却不说明在所有单位下三角因子中它是唯一的。

<!-- bilingual-en:start -->
The location of zero matters. For $A=\operatorname{diag}(1,0)$, the same equations force $d_1=1$, $\ell=0$, $d_2=0$, so this unit-lower/diagonal factorization is unique. The last pivot has no entries below it to compute. Likewise, a one-dimensional zero matrix has the unique factors $L=[1]$, $D=[0]$. A zero pivot can cause nonuniqueness; it need not do so.
<!-- bilingual-en:end -->

零出现的位置很重要。对 $A=\operatorname{diag}(1,0)$，同一组方程强制 $d_1=1$、$\ell=0$、$d_2=0$，所以这种“单位下三角加对角”的分解唯一。最后一个主元下方已经没有需要计算的元素。一维零矩阵也只有 $L=[1]$、$D=[0]$ 这一组因子。因此，零主元**可能**导致不唯一，并非必然如此。

<!-- bilingual-en:start -->
## How positive semidefiniteness permits continuation
<!-- bilingual-en:end -->
*半正定结构怎样允许继续分解*

<!-- bilingual-en:start -->
Let $A$ be a real symmetric positive-semidefinite $n\times n$ matrix, with finite integer $n\ge1$. When the current block has dimension greater than one, separate its first coordinate:
<!-- bilingual-en:end -->

设 $A$ 是实对称半正定 $n\times n$ 矩阵，$n\ge1$ 为有限整数。当当前方块的维数大于一，把它的第一个坐标分开：

$$A=\begin{pmatrix}a&b^T\\b&B\end{pmatrix}.$$

<!-- bilingual-en:start -->
Here $a$ is its first diagonal entry, $b$ is the column below it, and $B$ contains the remaining rows and columns. If $a=0$, let $e_1=(1,0,\ldots,0)^T$. Then $e_1^TAe_1=0$, and the [[半正定零方向|PSD zero-direction theorem]] gives $Ae_1=0$. Hence the entire first column and row vanish: $b=0$. We can set this pivot to zero, choose zero entries beneath it in $L$, and continue with $B$, which is still positive semidefinite.
<!-- bilingual-en:end -->

其中 $a$ 是第一个对角元，$b$ 是它下方的列，$B$ 包含其余行列。若 $a=0$，取 $e_1=(1,0,\ldots,0)^T$，则 $e_1^TAe_1=0$。由[[半正定零方向]]定理得到 $Ae_1=0$，所以整条首列与首行都为零，即 $b=0$。此时可以令该主元为零，选取 $L$ 中它下方的元素全为零，再对仍然半正定的 $B$ 继续。

<!-- bilingual-en:start -->
If $a>0$, ordinary elimination is valid. Its remaining block is $S=B-bb^T/a$, called the Schur complement. It stays positive semidefinite because, for every column $u$ of the remaining dimension, setting $s=-b^Tu/a$ gives
<!-- bilingual-en:end -->

若 $a>0$，普通消元可以进行，剩余方块为 $S=B-bb^T/a$，称为 Schur 补。它仍半正定：对剩余维度的任意列向量 $u$，取 $s=-b^Tu/a$，就有

$$u^TSu=\begin{pmatrix}s\\u\end{pmatrix}^{T}A\begin{pmatrix}s\\u\end{pmatrix}\ge0.$$

<!-- bilingual-en:start -->
Repeating these two cases gives an $LDL^T$ factorization with nonnegative diagonal $D$ for every such PSD matrix. This is an explicitly extended recurrence: at a zero pivot it skips a zero column, rather than evaluating $0/0$. If the matrix is [[正定矩阵|positive definite]], every remaining block is positive definite and every pivot is strictly positive, so the ordinary division-based recurrence avoids this obstacle.
<!-- bilingual-en:end -->

反复使用这两种情况，就能为每个这样的半正定矩阵构造对角元非负的 $LDL^T$ 分解。这是**明确扩展过的递推**：零主元处跳过零列，不是计算 $0/0$。若原矩阵[[正定矩阵|正定]]，每个剩余方块都正定，所有主元都严格为正，普通的除法递推就不会遇到这一障碍。

<!-- bilingual-en:start -->
For complex [[Hermitian 矩阵|Hermitian]] PSD matrices, replace $T$ by the conjugate transpose $*$ and use $A=LDL^*$ with real nonnegative $D$. The same argument uses $S=B-bb^*/a$ and $s=-b^*u/a$. No coordinate exchange is needed for this exact PSD construction; numerical decisions about tiny pivots require a separate rounding-error analysis.
<!-- bilingual-en:end -->

对复 [[Hermitian 矩阵|Hermitian]] 半正定矩阵，把 $T$ 换成共轭转置 $*$，使用 $A=LDL^*$，其中 $D$ 的元素为非负实数。同一证明改用 $S=B-bb^*/a$ 与 $s=-b^*u/a$。这个精确的半正定构造不需要交换坐标；计算机中很小的主元该怎样处理，还需要另外分析舍入误差。

<!-- bilingual-en:start -->
## A zero first pivot can occur in an invertible matrix
<!-- bilingual-en:end -->
*可逆矩阵也可能从零主元开始*

<!-- bilingual-en:start -->
Without the PSD assumption, a zero diagonal need not force its row to vanish. Consider
<!-- bilingual-en:end -->

没有半正定前提时，对角元为零不迫使所在行全零。考虑

$$C=\begin{pmatrix}0&1\\1&0\end{pmatrix},\qquad C^2=I_2.$$

<!-- bilingual-en:start -->
Here $I_2$ is the [[单位矩阵|identity matrix]], so $C$ is its own [[逆矩阵|inverse]]. Yet its first scalar pivot is zero, and the equation $0\ell=1$ shows that no unit lower-triangular $L$ and diagonal $D$ can give $C=LDL^T$ in this order. Exchanging both coordinates leaves this particular matrix unchanged. Treating the whole $2\times2$ matrix as one pivot block works: use $L=I_2$ and the block $D=C$.
<!-- bilingual-en:end -->

$I_2$ 为[[单位矩阵]]，所以 $C$ 自己就是其[[逆矩阵]]。但它的首个标量主元为零，方程 $0\ell=1$ 表明：在这一顺序下，不存在单位下三角 $L$ 和对角 $D$ 使 $C=LDL^T$。交换两个坐标，这个特定矩阵仍不变。把整个二阶矩阵当作一个主元块则可行：取 $L=I_2$、块 $D=C$。

<!-- bilingual-en:start -->
General symmetric or Hermitian indefinite algorithms can combine simultaneous row-and-column permutations with $1\times1$ or $2\times2$ pivot blocks. Their $D$ is block diagonal, so a block's sign must be assessed as a whole. The displayed block has zero diagonal entries but quadratic values $2$ and $-2$ at $(1,1)^T$ and $(1,-1)^T$. Reading only its two diagonal entries would miss both directions. This is a different procedure from the scalar-pivot criterion in [[LDL正定判别|the positive-definite LDL theorem]].
<!-- bilingual-en:end -->

一般对称或 Hermitian 不定矩阵的算法可以结合同行同列的置换与一阶、二阶主元块。此时 $D$ 是块对角矩阵，必须把每个方块作为整体判断符号。上面的方块虽有两个零对角元，在 $(1,1)^T$ 与 $(1,-1)^T$ 处的二次型值却为 $2$ 与 $-2$；只看两项对角元，会漏掉这两类方向。这与 [[LDL正定判别]]中的标量主元流程不同。

<!-- bilingual-en:start -->
## Sources and verification
<!-- bilingual-en:end -->
## 来源与核验

<!-- bilingual-en:start -->
- [N. Hu, UCLA, Symmetric factorizations, pp.1–2](https://math.ucla.edu/~njhu/notes/nla/lin-direct/symfact.pdf#page=1): scalar LDL* elimination and the symmetric-permutation/block-pivot alternative. The zero-pivot examples are multiplied out above.
- [Axler, Linear Algebra Done Right, fourth edition, §7C, results7.34–7.38, PDF265–266](https://linear.axler.net/LADR4e.pdf#page=265), and [[半正定零方向|the PSD zero-direction theorem]]: nonnegative spectral terms and the zero-column implication. The skip-zero PSD construction follows by the displayed block induction.
- [[LDL正定判别|The positive-definite LDL criterion]]: the strict-positive-pivot and fixed-order uniqueness comparison.
<!-- bilingual-en:end -->

- UCLA 的 Hu 讲义第1–2页：支持标量 LDL* 消元，以及对称置换、分块主元的替代流程；零主元例子按正文直接相乘核验。
- Axler 第4版§7C，7.34–7.38，PDF265–266页及[[半正定零方向]]：核对非负谱项与零列推论；跳过零列的半正定构造由正文的分块归纳推出。
- [[LDL正定判别]]：对照严格正主元与固定顺序下的唯一性。
